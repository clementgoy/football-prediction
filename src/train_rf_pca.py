from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.print_result import print_report


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# CHANGED: Use the PCA-augmented dataset
X_PATH = PROCESSED / "train_merged_pca.csv"
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

    info("Chargement X (avec PCA) …")
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
    """Aligne X et y, encode la target en classes {0,1,2} et garde seulement les features numériques."""
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
    X_imp = X_num.fillna(0.0)

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
        class_weight="balanced_subsample",  # pour aider sur la classe DRAW
    )
    return rf


def train_random_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    ids: pd.Index,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Split train/val, entraine un RandomForest (sans sample_weight) et sauvegarde le modèle + métriques."""

    # Split simple, stratifié
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    (tr_idx, va_idx), = sss.split(X, y)

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    ok(f"Split train/val: {X_tr.shape[0]} train, {X_va.shape[0]} val")

    # Modèle
    clf = make_random_forest(random_state=random_state)

    info("Entraînemen RandomForest (sans poids) …")
    clf.fit(X_tr, y_tr)

    # Accuracy validation
    y_va_pred = clf.predict(X_va)
    val_acc = accuracy_score(y_va, y_va_pred)
    ok(f"RandomForest: val_accuracy = {val_acc:.4f}")

    # Accuracy train (pour vérifier l'overfit)
    y_tr_pred = clf.predict(X_tr)
    train_acc = accuracy_score(y_tr, y_tr_pred)
    ok(f"RandomForest: train_accuracy = {train_acc:.4f}")

    # Rapport + matrice de confusion
    clf_rep = classification_report(
        y_va,
        y_va_pred,
        target_names=["HOME_WINS", "DRAW", "AWAY_WINS"],
        digits=4,
    )
    cm = confusion_matrix(y_va, y_va_pred)
    cm_list = cm.tolist()

    # Top features (si dispo)
    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_)
        feature_names = np.array(X.columns)
        order = np.argsort(importances)[::-1]
        top_features = feature_names[order].tolist()
    else:
        top_features = list(X.columns)

    # Sauvegardes
    tag = "rf_pca" # CHANGED
    model_path = MODELS / f"{tag}.joblib"
    joblib.dump(clf, model_path)
    ok(f"Modèle sauvegardé: {model_path}")

    meta = {
        "model": "rf_pca", # CHANGED
        "weight_scheme": "none",
        "val_accuracy": float(val_acc),
        "train_accuracy": float(train_acc),
        "n_train": int(X_tr.shape[0]),
        "n_val": int(X_va.shape[0]),
        "n_features": int(X.shape[1]),
        "random_state": random_state,
        "confusion_matrix": cm_list,
        "classification_report": clf_rep,
    }
    (MODELS / f"{tag}_metrics.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    (MODELS / f"{tag}_features.json").write_text(
        json.dumps({"feature_names": list(X.columns)}, indent=2),
        encoding="utf-8",
    )

    # Appel à la fonction de print custom
    print_report(
        train_acc=train_acc,
        val_acc=val_acc,
        hold_acc=val_acc,       # pas de vrai hold-out séparé ici
        cm=cm,
        clf_report=clf_rep,
        top_features=top_features,
        X=X,
        X_tr_sel=X_tr,
        X_va_sel=X_va,
        X_ho_sel=X_va,          # on réutilise la val comme "hold"
    )

    return {
        "clf": clf,
        "val_accuracy": val_acc,
        "train_accuracy": train_acc,
        "model_path": str(model_path),
        "meta": meta,
    }


def main() -> None:
    X_raw, y_onehot = load_data()
    X, y, ids = prepare_features_labels(X_raw, y_onehot)

    info("--- Entraînement RandomForest avec poids = none et PCA ---")
    _ = train_random_forest(X, y, ids, random_state=42)

    ok("Terminé. Regarde le dossier 'models/' (rf_pca.joblib + json).")


if __name__ == "__main__":
    main()
