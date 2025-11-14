import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
from lightgbm import LGBMRegressor
import lightgbm as lgb
import joblib

from src.print_result import print_report
from src.features_diff import build_features_with_diff


# -------------------------------------------------------------------
#  Construction de X et y (goal diff)
# -------------------------------------------------------------------
def build_Xy_goal_diff(train_csv: str, y_supp_csv: str):
    """
    Construit X (features enrichies) et y (GOAL_DIFF_HOME_AWAY).

    - train_csv : CSV fusionné, ex: data/processed/train_merged.csv
      (doit contenir une colonne 'ID')
    - y_supp_csv : Y_train_supp_aligned.csv (colonnes ['ID', 'GOAL_DIFF_HOME_AWAY'])
    """
    X_raw = pd.read_csv(train_csv)
    if "ID" not in X_raw.columns:
        raise ValueError("Le fichier train_csv doit contenir une colonne 'ID'.")

    ids = X_raw["ID"].values

    # Features avec colonnes home/away + diff
    X = build_features_with_diff(X_raw, drop_id_cols=True)

    # Sécurité si jamais ID reste
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    # On ne garde que les features numériques
    X = X.select_dtypes(include=["number"]).copy()
    print(f"[debug] Features numériques après filtrage: {X.shape}")

    # Cible = goal diff
    y_supp = pd.read_csv(y_supp_csv)
    if not {"ID", "GOAL_DIFF_HOME_AWAY"}.issubset(y_supp.columns):
        raise ValueError(
            "y_supp_csv doit contenir les colonnes 'ID' et 'GOAL_DIFF_HOME_AWAY'."
        )

    y_supp = y_supp.set_index("ID").loc[ids]
    y = y_supp["GOAL_DIFF_HOME_AWAY"].astype(float).values

    return X, y, ids


# -------------------------------------------------------------------
#  Conversion goal diff -> classes
# -------------------------------------------------------------------
def goal_diff_to_class(diff: np.ndarray) -> np.ndarray:
    """
    Convertit un goal diff en classe:
        > 0 → 0 (HOME_WINS)
        = 0 → 1 (DRAW)
        < 0 → 2 (AWAY_WINS)
    """
    diff = np.asarray(diff)
    cls = np.zeros_like(diff, dtype=int)
    cls[diff < 0] = 2
    cls[diff == 0] = 1
    return cls


def goal_diff_to_class_with_band(diff: np.ndarray, band: float = 0.5) -> np.ndarray:
    """
    Version avec bande autour de 0 pour mieux capter les matchs nuls.

      - if diff >  band  → 0 (HOME)
      - if |diff| <= band → 1 (DRAW)
      - if diff < -band  → 2 (AWAY)
    """
    diff = np.asarray(diff)
    cls = np.zeros_like(diff, dtype=int)
    cls[diff < -band] = 2          # AWAY
    cls[np.abs(diff) <= band] = 1  # DRAW
    return cls


# -------------------------------------------------------------------
#  Entraînement + évaluation (focus sur les nuls)
# -------------------------------------------------------------------
def train_lgbm_goal_diff_draw_focus(
    train_csv: str,
    y_supp_csv: str,
    model_out: str,
    valid_size: float = 0.1667,
    holdout_size: float = 0.1667,
    random_state: int = 42,
):
    # 1) Chargement X, y (goal diff)
    X, y_diff, ids = build_Xy_goal_diff(train_csv, y_supp_csv)

    # Labels de classe dérivés du goal diff (0/1/2) pour le stratify
    y_cls = goal_diff_to_class(y_diff)

    assert len(X) == len(y_diff) == len(y_cls), "Longueurs incohérentes entre X et y."

    # 2) Split train / valid / holdout
    print("[info] Split train / valid / holdout ...")

    X_train_valid, X_hold, y_train_valid_diff, y_hold_diff, y_train_valid_cls, y_hold_cls = train_test_split(
        X,
        y_diff,
        y_cls,
        test_size=holdout_size,
        random_state=random_state,
        stratify=y_cls,
    )

    valid_ratio_within_train_valid = valid_size / (1.0 - holdout_size)

    X_tr, X_va, y_tr_diff, y_va_diff, y_tr_cls, y_va_cls = train_test_split(
        X_train_valid,
        y_train_valid_diff,
        y_train_valid_cls,
        test_size=valid_ratio_within_train_valid,
        random_state=random_state,
        stratify=y_train_valid_cls,
    )

    print(
        f"[info] Tailles : train={len(X_tr)} | valid={len(X_va)} | holdout={len(X_hold)}"
    )

    # 3) Modèle LightGBM Regressor sur l'écart de buts
    print("[info] Entraînement du LightGBM Regressor sur l'écart de buts ...")

    reg = LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )

    reg.fit(
        X_tr,
        y_tr_diff,
        eval_set=[(X_va, y_va_diff)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100),
        ],
    )

    best_it = reg.best_iteration_
    if best_it is None:
        best_it = reg.n_estimators

    # 4) Prédiction & RMSE goal diff
    y_tr_pred_diff = reg.predict(X_tr, num_iteration=best_it)
    y_va_pred_diff = reg.predict(X_va, num_iteration=best_it)
    y_hold_pred_diff = reg.predict(X_hold, num_iteration=best_it)

    rmse_tr = np.sqrt(mean_squared_error(y_tr_diff, y_tr_pred_diff))
    rmse_va = np.sqrt(mean_squared_error(y_va_diff, y_va_pred_diff))
    rmse_hold = np.sqrt(mean_squared_error(y_hold_diff, y_hold_pred_diff))

    print(
        f"[RMSE] train={rmse_tr:.4f} | valid={rmse_va:.4f} | holdout={rmse_hold:.4f}"
    )

    # 5) Tuning du band pour MAXIMISER le F1 de la classe "Draw" (1)
    print("[info] Tuning du seuil pour les matchs nuls (band) en max F1(classe 1) ...")
    candidate_bands = np.linspace(0.1, 2.5, 25)
    best_band = None
    best_score = -1.0

    for b in candidate_bands:
        y_va_pred_cls_tmp = goal_diff_to_class_with_band(y_va_pred_diff, band=b)

        # F1 pour la classe 1 (match nul) vs le reste
        y_true_bin = (y_va_cls == 1).astype(int)
        y_pred_bin = (y_va_pred_cls_tmp == 1).astype(int)
        f1_draw = f1_score(y_true_bin, y_pred_bin)

        if f1_draw > best_score:
            best_score = f1_draw
            best_band = b

    print(f"[tuning] Meilleur band = {best_band:.3f} | F1 Draw (val) = {best_score:.4f}")

    # 6) Applique ce band sur les 3 splits et calcule les accuracies globales
    y_tr_pred_cls = goal_diff_to_class_with_band(y_tr_pred_diff, band=best_band)
    y_va_pred_cls = goal_diff_to_class_with_band(y_va_pred_diff, band=best_band)
    y_hold_pred_cls = goal_diff_to_class_with_band(y_hold_pred_diff, band=best_band)

    acc_tr = accuracy_score(y_tr_cls, y_tr_pred_cls)
    acc_va = accuracy_score(y_va_cls, y_va_pred_cls)
    acc_hold = accuracy_score(y_hold_cls, y_hold_pred_cls)

    # 7) Matrice de confusion & rapport sur le hold-out (jeu le plus honnête)
    cm = confusion_matrix(y_hold_cls, y_hold_pred_cls)
    clf_report = classification_report(
        y_hold_cls,
        y_hold_pred_cls,
        digits=3,
        target_names=["Home (0)", "Draw (1)", "Away (2)"],
    )

    # 8) Top features les plus importantes
    importances = reg.feature_importances_
    feature_names = np.array(X.columns)
    order = np.argsort(importances)[::-1]
    top_features = feature_names[order[:20]].tolist()

    # 9) Impression du rapport via print_report (même format que ton script actuel)
    print_report(
        train_acc=acc_tr,
        val_acc=acc_va,
        hold_acc=acc_hold,
        cm=cm,
        clf_report=clf_report,
        top_features=top_features,
        X=X,
        X_tr_sel=X_tr,
        X_va_sel=X_va,
        X_ho_sel=X_hold,
    )

    # 10) Sauvegarde du modèle + band
    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": reg,
            "band": best_band,
            "feature_columns": X.columns.tolist(),
        },
        out_path,
    )
    print(f"\n[ok] Modèle LightGBM goal_diff + band (focus draws) sauvegardé dans {out_path.resolve()}")


# -------------------------------------------------------------------
#  CLI
# -------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne un LightGBM Regressor sur GOAL_DIFF_HOME_AWAY, "
            "puis convertit en classes (Home/Draw/Away) via un band optimisé "
            "pour le F1 de la classe 'Draw' (1)."
        )
    )

    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="CSV des features train fusionnées (ex: data/processed/train_merged.csv)",
    )
    parser.add_argument(
        "--y-supp-csv",
        type=str,
        required=True,
        help="Y_train_supp_aligned.csv contenant GOAL_DIFF_HOME_AWAY",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        required=True,
        help="Chemin de sortie du modèle LightGBM (pkl)",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.1667,
        help="Proportion pour la validation (par défaut 0.1667 ≈ 1/6).",
    )
    parser.add_argument(
        "--holdout-size",
        type=float,
        default=0.1667,
        help="Proportion pour le hold-out (par défaut 0.1667 ≈ 1/6).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed pour le split et le modèle.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train_lgbm_goal_diff_draw_focus(
        train_csv=args.train_csv,
        y_supp_csv=args.y_supp_csv,
        model_out=args.model_out,
        valid_size=args.valid_size,
        holdout_size=args.holdout_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()