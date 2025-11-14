#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import lightgbm as lgb

from src.print_result import print_report
from src.features_diff import add_interaction_features, build_features_with_diff


def build_Xy_multiclass_with_diff(train_csv: str, y_csv: str):
    """
    Construit:
      - X : features numériques + colonnes diff (home-away)
      - y : classes 0=Home, 1=Draw, 2=Away
    """
    # 1) Chargement X brut
    X_raw = pd.read_csv(train_csv)
    if "ID" not in X_raw.columns:
        raise ValueError("Le fichier train_csv doit contenir une colonne 'ID'.")

    ids = X_raw["ID"].values

    # 2) Ajout des features diff (home/away + diff_*)
    X = build_features_with_diff(X_raw, drop_id_cols=True)

    # Sécurité si jamais "ID" traîne encore
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    # On garde uniquement les colonnes numériques (LightGBM + stable)
    X = X.select_dtypes(include=["number"]).copy()
    print(f"[debug] Features numériques après filtrage: {X.shape}")

    # 3) Chargement du y (HOME_WINS, DRAW, AWAY_WINS)
    y_raw = pd.read_csv(y_csv)
    expected_cols = {"ID", "HOME_WINS", "DRAW", "AWAY_WINS"}
    if not expected_cols.issubset(y_raw.columns):
        raise ValueError(
            f"y_csv doit contenir les colonnes {expected_cols}, trouvé {y_raw.columns}"
        )

    # Alignement sur les IDs de X
    y_raw = y_raw.set_index("ID").loc[ids]

    # Passer du one-hot [HOME_WINS, DRAW, AWAY_WINS] -> classe 0/1/2
    y_onehot = y_raw[["HOME_WINS", "DRAW", "AWAY_WINS"]].values
    y = y_onehot.argmax(axis=1)  # 0=Home, 1=Draw, 2=Away

    return X, y, ids


def split_train_valid_holdout(X, y, valid_size=0.1667, holdout_size=0.1667, random_state=42):
    # 1) Séparer un holdout final
    X_train_valid, X_hold, y_train_valid, y_hold = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=random_state,
        stratify=y,
    )

    # 2) Séparer train / valid à l'intérieur
    valid_ratio_within_train_valid = valid_size / (1.0 - holdout_size)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_ratio_within_train_valid,
        random_state=random_state,
        stratify=y_train_valid,
    )

    print(
        f"[split] train={X_tr.shape}, valid={X_va.shape}, holdout={X_hold.shape}"
    )
    return X_tr, X_va, X_hold, y_tr, y_va, y_hold

def make_lgbm_multiclass(random_state=42):
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        boosting_type="dart",     
        class_weight="balanced",   
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=200,    
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
    )
    return clf

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


def train_lgbm_multiclass_diff(
    train_csv: str,
    y_csv: str,
    model_out: str,
    valid_size: float = 0.1667,
    holdout_size: float = 0.1667,
    random_state: int = 42,
):
    # 1) Build X, y
    X, y, ids = build_Xy_multiclass_with_diff(train_csv, y_csv)
    X = add_interaction_features(X)

    # 2) Split
    X_tr, X_va, X_hold, y_tr, y_va, y_hold = split_train_valid_holdout(
        X, y, valid_size=valid_size, holdout_size=holdout_size, random_state=random_state
    )

    # 3) Modèle LGBM
    clf = make_lgbm_multiclass(random_state=random_state)

    # 4) Fit avec early stopping
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100),
        ],
    )

    best_it = clf.best_iteration_
    if best_it is None:
        best_it = clf.n_estimators

    # 5) Prédiction
    y_tr_pred = clf.predict(X_tr, num_iteration=best_it)
    y_va_pred = clf.predict(X_va, num_iteration=best_it)
    y_hold_pred = clf.predict(X_hold, num_iteration=best_it)

    # 6) Scores
    acc_tr = accuracy_score(y_tr, y_tr_pred)
    acc_va = accuracy_score(y_va, y_va_pred)
    acc_hold = accuracy_score(y_hold, y_hold_pred)

    # Matrice de confusion sur holdout
    cm = confusion_matrix(y_hold, y_hold_pred)

    target_names = ["Home (0)", "Draw (1)", "Away (2)"]
    clf_report = classification_report(y_hold, y_hold_pred, target_names=target_names)


    
    # 7) Top features
    importances = clf.feature_importances_
    feature_names = np.array(X.columns)
    order = np.argsort(importances)[::-1]
    top_features = feature_names[order[:20]].tolist()

    # 8) Impression via print_report (comme ton script actuel)
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

    # 9) Sauvegarde du modèle
    from pathlib import Path
    import joblib

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "best_iteration": best_it,
            "feature_columns": X.columns.tolist(),
            "class_mapping": {0: "HOME", 1: "DRAW", 2: "AWAY"},
        },
        out_path,
    )
    print(f"\n[ok] Modèle LightGBM multiclass + diff sauvegardé dans {out_path.resolve()}")

# -------------------------------------------------------------------
#  CLI
# -------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne un LightGBMClassifier multiclass (Home/Draw/Away) "
            "avec features diff et interactions, sans focus draw."
        )
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="CSV train fusionné (ex: data/processed/train_merged.csv)",
    )
    parser.add_argument(
        "--y-csv",
        type=str,
        required=True,
        help="Y_train aligné (avec colonnes ID, HOME_WINS, DRAW, AWAY_WINS)",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        required=True,
        help="Chemin de sortie du modèle .pkl/.joblib",
    )
    parser.add_argument(
        "--valid-size", type=float, default=0.1667, help="Proportion validation"
    )
    parser.add_argument(
        "--holdout-size", type=float, default=0.1667, help="Proportion holdout"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Seed aléatoire"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_lgbm_multiclass_diff(
        train_csv=args.train_csv,
        y_csv=args.y_csv,
        model_out=args.model_out,
        valid_size=args.valid_size,
        holdout_size=args.holdout_size,
        random_state=args.random_state,
    )
