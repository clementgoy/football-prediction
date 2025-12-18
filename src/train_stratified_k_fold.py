from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

from print_result import print_report


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

X_PATH = PROCESSED / "train_merged.csv"
Y_ONEHOT_PATH = PROCESSED / "y_train_aligned.csv"


def info(msg: str) -> None:
    print(f"\n[info] {msg}")


def ok(msg: str) -> None:
    print(f"[ok] {msg}")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge X et y_onehot aligné sur les IDs de train."""
    if not X_PATH.exists():
        raise FileNotFoundError(f"Introuvable: {X_PATH}")
    if not Y_ONEHOT_PATH.exists():
        raise FileNotFoundError(f"Introuvable: {Y_ONEHOT_PATH}")

    info("Chargement X …")
    X = pd.read_csv(X_PATH, low_memory=False)
    ok(f"X: {X.shape[0]} lignes × {X.shape[1]} colonnes")

    info("Chargement y one-hot …")
    y1 = pd.read_csv(Y_ONEHOT_PATH, low_memory=False)
    ok(f"y_onehot: {y1.shape[0]} lignes × {y1.shape[1]} colonnes")

    return X, y1


def prepare_features_labels(
    X: pd.DataFrame,
    y_onehot: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Index]:
    """
    Aligne X et y, encode la target en classes {0,1,2}
    et garde seulement les features numériques.
    """
    need = ["ID", "HOME_WINS", "DRAW", "AWAY_WINS"]
    if not set(need).issubset(y_onehot.columns):
        raise ValueError("y_onehot doit contenir ID, HOME_WINS, DRAW, AWAY_WINS")

    merged = X.merge(y_onehot[need], on="ID", how="inner")
    ok(f"Alignement X↔y: {merged.shape[0]} lignes")

    # target : argmax sur les 3 colonnes one-hot → 0,1,2
    y_cls = merged[["HOME_WINS", "DRAW", "AWAY_WINS"]].values.argmax(axis=1)

    # features : colonnes numériques uniquement
    feature_cols_all = [
        c for c in merged.columns
        if c not in ("ID", "HOME_WINS", "DRAW", "AWAY_WINS")
    ]
    num_cols = merged[feature_cols_all].select_dtypes(include=[np.number]).columns
    dropped = len(feature_cols_all) - len(num_cols)
    if dropped > 0:
        info(f"Colonnes non numériques écartées: {dropped}")

    X_num = merged[num_cols].copy()

    # Imputation simple (remplacement des NaN par 0.0)
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_num),
        columns=num_cols,
        index=X_num.index,
    )

    return X_imp, y_cls, merged["ID"]


def make_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Construit le RandomForest avec les hyperparamètres choisis."""
    rf = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    return rf


def train_rf_with_stratified_kfold(
    X: pd.DataFrame,
    y: np.ndarray,
    ids: pd.Index,
    *,
    random_state: int = 42,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    1) Garde un hold-out stratifié (20 %) de côté.
    2) Fait une cross-validation StratifiedKFold sur les 80 % restants.
    3) Entraîne un modèle final sur les 80 %.
    4) Évalue sur le hold-out.
    5) Sauvegarde modèle + métriques.
    """

    # 1) Split global en train_cv / hold-out
    info("Split global train_cv / hold-out (StratifiedShuffleSplit, test_size=0.2) …")
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=random_state,
    )
    (train_cv_idx, hold_idx), = sss.split(X, y)

    X_cv = X.iloc[train_cv_idx]
    y_cv = y[train_cv_idx]
    X_ho = X.iloc[hold_idx]
    y_ho = y[hold_idx]

    ok(
        f"Train-CV: {X_cv.shape[0]} lignes, "
        f"Hold-out: {X_ho.shape[0]} lignes"
    )

    # 2) Cross-validation StratifiedKFold sur train_cv
    info(
        f"Cross-validation StratifiedKFold sur {n_splits} folds "
        "(sur le bloc train_cv uniquement) …"
    )
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_train_accs = []
    fold_val_accs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_cv, y_cv), start=1):
        X_tr = X_cv.iloc[tr_idx]
        y_tr = y_cv[tr_idx]
        X_va = X_cv.iloc[va_idx]
        y_va = y_cv[va_idx]

        ok(
            f"[Fold {fold_idx}/{n_splits}] "
            f"train={X_tr.shape[0]} | val={X_va.shape[0]}"
        )

        clf = make_random_forest(random_state=random_state + fold_idx)
        clf.fit(X_tr, y_tr)

        # train acc
        y_tr_pred = clf.predict(X_tr)
        acc_tr = accuracy_score(y_tr, y_tr_pred)

        # val acc
        y_va_pred = clf.predict(X_va)
        acc_va = accuracy_score(y_va, y_va_pred)

        fold_train_accs.append(acc_tr)
        fold_val_accs.append(acc_va)

        ok(
            f"[Fold {fold_idx}] "
            f"train_acc={acc_tr:.4f} | val_acc={acc_va:.4f}"
        )

    mean_train_acc = float(np.mean(fold_train_accs))
    std_train_acc = float(np.std(fold_train_accs))
    mean_val_acc = float(np.mean(fold_val_accs))
    std_val_acc = float(np.std(fold_val_accs))

    info(
        "Résumé StratifiedKFold:\n"
        f"  train_acc  mean={mean_train_acc:.4f} ± {std_train_acc:.4f}\n"
        f"  val_acc    mean={mean_val_acc:.4f} ± {std_val_acc:.4f}"
    )

    # 3) Entraînement du modèle final sur le bloc train_cv complet
    info("Entraînement du modèle final sur tout le bloc train_cv …")
    final_clf = make_random_forest(random_state=random_state)
    final_clf.fit(X_cv, y_cv)

    y_cv_pred = final_clf.predict(X_cv)
    final_train_acc = accuracy_score(y_cv, y_cv_pred)
    ok(f"Final RF sur train_cv: train_accuracy = {final_train_acc:.4f}")

    # 4) Évaluation sur le hold-out
    y_ho_pred = final_clf.predict(X_ho)
    hold_acc = accuracy_score(y_ho, y_ho_pred)
    ok(f"Final RF sur hold-out: hold_accuracy = {hold_acc:.4f}")

    clf_rep_ho = classification_report(
        y_ho,
        y_ho_pred,
        target_names=["HOME_WINS", "DRAW", "AWAY_WINS"],
        digits=4,
    )
    cm_ho = confusion_matrix(y_ho, y_ho_pred)
    cm_ho_list = cm_ho.tolist()

    # 5) Top features
    if hasattr(final_clf, "feature_importances_"):
        importances = np.asarray(final_clf.feature_importances_)
        feature_names = np.array(X.columns)
        order = np.argsort(importances)[::-1]
        top_features = feature_names[order].tolist()
    else:
        top_features = list(X.columns)

    # 6) Sauvegardes
    tag = "rf_stratkfold"
    model_path = MODELS / f"{tag}.joblib"
    joblib.dump(final_clf, model_path)
    ok(f"Modèle sauvegardé: {model_path}")

    meta: Dict[str, Any] = {
        "model": "rf",
        "weight_scheme": "balanced_subsample",
        "cv": {
            "n_splits": n_splits,
            "train_acc_mean": mean_train_acc,
            "train_acc_std": std_train_acc,
            "val_acc_mean": mean_val_acc,
            "val_acc_std": std_val_acc,
            "fold_train_accs": fold_train_accs,
            "fold_val_accs": fold_val_accs,
        },
        "final_train_accuracy": final_train_acc,
        "hold_accuracy": hold_acc,
        "n_train_cv": int(X_cv.shape[0]),
        "n_holdout": int(X_ho.shape[0]),
        "n_features": int(X.shape[1]),
        "random_state": random_state,
        "confusion_matrix_holdout": cm_ho_list,
        "classification_report_holdout": clf_rep_ho,
    }

    (MODELS / f"{tag}_metrics.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    (MODELS / f"{tag}_features.json").write_text(
        json.dumps({"feature_names": list(X.columns)}, indent=2),
        encoding="utf-8",
    )

    # 7) Appel à la fonction de print custom
    print_report(
        train_acc=final_train_acc,
        val_acc=mean_val_acc,      # on affiche la moyenne CV comme "val"
        hold_acc=hold_acc,
        cm=cm_ho,
        clf_report=clf_rep_ho,
        top_features=top_features,
        X=X,
        X_tr_sel=X_cv,
        X_va_sel=X_cv,             # pas de val unique : on met train_cv
        X_ho_sel=X_ho,
    )

    # 8) Fichier best.json (ici on suppose que ce modèle devient le "best")
    best_for_json = {
        "key": tag,
        "acc": float(hold_acc),
        "model_path": str(model_path),
        "meta": meta,
    }
    (MODELS / "best.json").write_text(
        json.dumps(best_for_json, indent=2),
        encoding="utf-8",
    )

    ok(
        f"Meilleur modèle mis à jour : {tag} "
        f"(hold_acc={hold_acc:.4f}) → {model_path}"
    )

    return {
        "clf": final_clf,
        "cv_val_accuracy_mean": mean_val_acc,
        "final_train_accuracy": final_train_acc,
        "hold_accuracy": hold_acc,
        "model_path": str(model_path),
        "meta": meta,
    }


def main() -> None:
    X_raw, y_onehot = load_data()
    X, y, ids = prepare_features_labels(X_raw, y_onehot)

    info("--- Entraînement RandomForest avec StratifiedKFold + hold-out ---")
    _ = train_rf_with_stratified_kfold(
        X,
        y,
        ids,
        random_state=42,
        n_splits=5,
    )

    ok("Terminé. Regarde le dossier 'models/' (rf_stratkfold*.json + .joblib).")


if __name__ == "__main__":
    main()
