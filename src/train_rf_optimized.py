from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Import robuste de ta fonction de print
try:
    from .print_result import print_report
except ImportError:  # si tu exécutes depuis src/ directement
    from print_result import print_report


# -------------------------------------------------------------------
# Chemins
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_X_PATH = PROC / "train_merged.csv"
Y_PATH = PROC / "y_train_aligned.csv"

MODEL_OUT_PATH = MODELS_DIR / "random_forest.pkl"
FEATS_PATH = MODELS_DIR / "rf_feature_importances.csv"
METRICS_PATH = MODELS_DIR / "rf_metrics.json"


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
@dataclass
class TrainConfig:
    holdout_fraction: float = 0.2     # portion pour le hold-out
    val_fraction: float = 0.2         # portion pour la validation (sur le reste)
    random_state: int = 42

    # RF de sélection de features
    fs_n_estimators: int = 400
    fs_max_depth: int = 18
    fs_max_features: float = 0.6
    fs_top_k: int = 800               # nb de features à garder

    # RF finale
    rf_n_estimators: int = 1200
    rf_max_depth: int = 18
    rf_max_features: float = 0.4
    rf_min_samples_leaf: int = 4
    rf_min_samples_split: int = 6
    rf_class_weight: str = "balanced_subsample"


cfg = TrainConfig()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def info(msg: str) -> None:
    print(f"\n[info] {msg}")


def ok(msg: str) -> None:
    print(f"[ok] {msg}")


def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Charge X et y, fusionne sur ID et renvoie (X_features, y_labels)."""
    if not TRAIN_X_PATH.exists():
        raise FileNotFoundError(f"Fichier X introuvable: {TRAIN_X_PATH}")
    if not Y_PATH.exists():
        raise FileNotFoundError(f"Fichier y introuvable: {Y_PATH}")

    info("Chargement des données...")
    X = pd.read_csv(TRAIN_X_PATH, low_memory=False)
    y_df = pd.read_csv(Y_PATH, low_memory=False)

    # Merge sur ID
    merged = y_df[["ID", "HOME_WINS", "DRAW", "AWAY_WINS"]].merge(X, on="ID", how="left")

    # Cible : argmax sur [HOME_WINS, DRAW, AWAY_WINS] → 0/1/2
    y = merged[["HOME_WINS", "DRAW", "AWAY_WINS"]].values.argmax(axis=1)

    # Features : on enlève ID + cibles
    feats = merged.drop(columns=["HOME_WINS", "DRAW", "AWAY_WINS", "ID"], errors="ignore")

    # On garde uniquement les colonnes numériques (normalement tout est déjà numérisé)
    X_num = feats.select_dtypes(include=[np.number]).copy()

    info(f"Échantillons: {len(X_num)}, features numériques: {X_num.shape[1]}")
    return X_num, y


def select_top_features(X: pd.DataFrame, y: np.ndarray, cfg: TrainConfig) -> List[str]:
    """Sélectionne les top features via RandomForest et sauvegarde le CSV d'importances."""
    info("Sélection des features les plus importantes (RandomForest)...")

    rf_fs = RandomForestClassifier(
        n_estimators=cfg.fs_n_estimators,
        max_depth=cfg.fs_max_depth,
        max_features=cfg.fs_max_features,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        min_samples_split=cfg.rf_min_samples_split,
        n_jobs=-1,
        random_state=cfg.random_state,
        class_weight=cfg.rf_class_weight,
    )
    rf_fs.fit(X, y)

    importances = pd.DataFrame(
        {"feature": X.columns, "importance": rf_fs.feature_importances_}
    ).sort_values("importance", ascending=False)

    # On garde les top_k
    if cfg.fs_top_k is not None and cfg.fs_top_k < len(importances):
        importances_top = importances.head(cfg.fs_top_k).copy()
    else:
        importances_top = importances.copy()

    importances_top.to_csv(FEATS_PATH, index=False)
    ok(f"Importances sauvegardées dans {FEATS_PATH.name} ({len(importances_top)} features gardées)")

    return importances_top["feature"].tolist()


def make_splits(
    X: pd.DataFrame, y: np.ndarray, cfg: TrainConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Split en train / val / hold-out de manière stratifiée."""
    info("Découpage train / validation / hold-out...")

    # D'abord on isole le hold-out
    X_trva, X_ho, y_trva, y_ho = train_test_split(
        X,
        y,
        test_size=cfg.holdout_fraction,
        random_state=cfg.random_state,
        stratify=y,
    )

    # Puis on découpe train / val sur le reste
    val_size = cfg.val_fraction / (1.0 - cfg.holdout_fraction)  # ex: 0.2 / 0.8 = 0.25
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trva,
        y_trva,
        test_size=val_size,
        random_state=cfg.random_state,
        stratify=y_trva,
    )

    info(
        f"Tailles: train={len(X_tr)} | val={len(X_va)} | hold-out={len(X_ho)} "
        f"(features={X.shape[1]})"
    )
    return X_tr, X_va, X_ho, y_tr, y_va, y_ho


def build_model(cfg: TrainConfig) -> RandomForestClassifier:
    """Construit une RandomForest avec les hyperparamètres optimisés."""
    rf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        max_features=cfg.rf_max_features,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        min_samples_split=cfg.rf_min_samples_split,
        bootstrap=True,
        n_jobs=-1,
        random_state=cfg.random_state,
        class_weight=cfg.rf_class_weight,
    )
    return rf


# -------------------------------------------------------------------
# Main training pipeline
# -------------------------------------------------------------------
def main() -> None:
    info("Entraînement RandomForest optimisée ")

    # 1) Chargement des données
    X, y = load_data()

    # 2) Sélection de features via RF
    top_features = select_top_features(X, y, cfg)
    X_sel = X[top_features].copy()

    # 3) Découpage train / val / holdout
    X_tr, X_va, X_ho, y_tr, y_va, y_ho = make_splits(X_sel, y, cfg)

    # 4) Entraînement sur le train uniquement (pour les métriques)
    info("Entraînement du modèle sur le set d'entraînement...")
    rf = build_model(cfg)
    rf.fit(X_tr, y_tr)

    # 5) Évaluation train / val / hold-out
    y_tr_pred = rf.predict(X_tr)
    y_va_pred = rf.predict(X_va)
    y_ho_pred = rf.predict(X_ho)

    train_acc = accuracy_score(y_tr, y_tr_pred)
    val_acc = accuracy_score(y_va, y_va_pred)
    hold_acc = accuracy_score(y_ho, y_ho_pred)

    cm = confusion_matrix(y_ho, y_ho_pred)
    clf_rep = classification_report(y_ho, y_ho_pred, digits=3)

    # 6) Rapport formaté
    print_report(
        train_acc=train_acc,
        val_acc=val_acc,
        hold_acc=hold_acc,
        cm=cm,
        clf_report=clf_rep,
        top_features=top_features,
        X=X,                 
        X_tr_sel=X_tr,
        X_va_sel=X_va,
        X_ho_sel=X_ho,
    )

    # 7) Réentraînement sur train + val pour le modèle final
    info("Réentraînement sur train + val pour le modèle final...")
    X_final = pd.concat([X_tr, X_va], axis=0)
    y_final = np.concatenate([y_tr, y_va])

    rf_final = build_model(cfg)
    rf_final.fit(X_final, y_final)

    # 8) Sauvegarde du modèle
    joblib.dump(rf_final, MODEL_OUT_PATH)
    ok(f"Modèle final sauvegardé dans {MODEL_OUT_PATH}")

    # 9) Sauvegarde des métriques
    metrics = {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "holdout_accuracy": float(hold_acc),
        "n_samples": {
            "train": int(len(X_tr)),
            "val": int(len(X_va)),
            "holdout": int(len(X_ho)),
        },
        "n_features_total": int(X.shape[1]),
        "n_features_used": int(len(top_features)),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ok(f"Métriques sauvegardées dans {METRICS_PATH}")

    info("Entraînement terminé")


if __name__ == "__main__":
    main()