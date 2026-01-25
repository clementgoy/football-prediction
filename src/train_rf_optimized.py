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

try:
    from .print_result import print_report
except ImportError:
    from print_result import print_report


BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_X_PATH = PROC / "train_merged.csv"
Y_PATH = PROC / "y_train_aligned.csv"

MODEL_OUT_PATH = MODELS_DIR / "random_forest.pkl"
FEATS_PATH = MODELS_DIR / "rf_feature_importances.csv"
METRICS_PATH = MODELS_DIR / "rf_metrics.json"

@dataclass
class TrainConfig:
    holdout_fraction: float = 0.2
    val_fraction: float = 0.2
    random_state: int = 42

    fs_n_estimators: int = 400
    fs_max_depth: int = 18
    fs_max_features: float = 0.6
    fs_top_k: int = 800              

    rf_n_estimators: int = 1200
    rf_max_depth: int = 18
    rf_max_features: float = 0.4
    rf_min_samples_leaf: int = 4
    rf_min_samples_split: int = 6
    rf_class_weight: str = "balanced_subsample"


cfg = TrainConfig()

def info(msg: str) -> None:
    print(f"\n[info] {msg}")


def ok(msg: str) -> None:
    print(f"[ok] {msg}")


def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
    if not TRAIN_X_PATH.exists():
        raise FileNotFoundError(f"Je trouve pas le fichier X : {TRAIN_X_PATH}, bizarre...")
    if not Y_PATH.exists():
        raise FileNotFoundError(f"Je trouve pas le fichier y : {Y_PATH}, bizarre...")

    info("On charge les données (ça va vite)...")
    X = pd.read_csv(TRAIN_X_PATH, low_memory=False)
    y_df = pd.read_csv(Y_PATH, low_memory=False)

    # On colle les réponses (y) avec les données (X) grâce à l'ID
    merged = y_df[["ID", "HOME_WINS", "DRAW", "AWAY_WINS"]].merge(X, on="ID", how="left")

    # On transforme les 3 colonnes de résultats en une seule (0, 1, 2)
    y = merged[["HOME_WINS", "DRAW", "AWAY_WINS"]].values.argmax(axis=1)

    # On enlève ce qui sert pas à l'entraînement
    feats = merged.drop(columns=["HOME_WINS", "DRAW", "AWAY_WINS", "ID"], errors="ignore")

    # On garde que les colonnes avec des chiffres
    X_num = feats.select_dtypes(include=[np.number]).copy()

    info(f"C'est bon ! On a {len(X_num)} lignes et {X_num.shape[1]} features numériques.")
    return X_num, y


def select_top_features(X: pd.DataFrame, y: np.ndarray, cfg: TrainConfig) -> List[str]:
    info("Sélection des meilleures features (celles qui servent vraiment)...")

    # On lance un premier forêt "brouillon" pour voir quelles stats sont importantes
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

    # On trie les features par importance
    importances = pd.DataFrame(
        {"feature": X.columns, "importance": rf_fs.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Si on en a trop, on coupe pour garder que le top_k
    if cfg.fs_top_k is not None and cfg.fs_top_k < len(importances):
        importances_top = importances.head(cfg.fs_top_k).copy()
    else:
        importances_top = importances.copy()

    # On sauvegarde la liste
    importances_top.to_csv(FEATS_PATH, index=False)
    ok(f"Liste des features importantes sauvegardée dans {FEATS_PATH.name} ({len(importances_top)} retenues)")

    return importances_top["feature"].tolist()


def make_splits(
    X: pd.DataFrame, y: np.ndarray, cfg: TrainConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    info("Découpage en 3 morceaux : train / validation / hold-out (pour être sûr)...")

    # On coupe d'abord un morceau 'hold-out' qu'on met de côté pour la fin
    X_trva, X_ho, y_trva, y_ho = train_test_split(
        X,
        y,
        test_size=cfg.holdout_fraction,
        random_state=cfg.random_state,
        stratify=y,
    )

    # Ensuite on coupe le reste en train et validation
    val_size = cfg.val_fraction / (1.0 - cfg.holdout_fraction)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trva,
        y_trva,
        test_size=val_size,
        random_state=cfg.random_state,
        stratify=y_trva,
    )

    info(
        f"C'est coupé ! Train={len(X_tr)} | Val={len(X_va)} | Hold-out={len(X_ho)} "
        f"(et y'a {X.shape[1]} colonnes)"
    )
    return X_tr, X_va, X_ho, y_tr, y_va, y_ho


def build_model(cfg: TrainConfig) -> RandomForestClassifier:
    # Création du modèle avec tous les paramètres "optimisés"
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


def main() -> None:
    info("Lancement de l'entraînement RandomForest (version optimisée) !")

    X, y = load_data()

    # On choisit les meilleures colonnes
    top_features = select_top_features(X, y, cfg)
    X_sel = X[top_features].copy()

    # On coupe
    X_tr, X_va, X_ho, y_tr, y_va, y_ho = make_splits(X_sel, y, cfg)

    info("Premier entraînement sur le set d'entraînement...")
    rf = build_model(cfg)
    rf.fit(X_tr, y_tr)

    # On regarde ce que ça donne
    y_tr_pred = rf.predict(X_tr)
    y_va_pred = rf.predict(X_va)
    y_ho_pred = rf.predict(X_ho)

    train_acc = accuracy_score(y_tr, y_tr_pred)
    val_acc = accuracy_score(y_va, y_va_pred)
    hold_acc = accuracy_score(y_ho, y_ho_pred)

    cm = confusion_matrix(y_ho, y_ho_pred)
    clf_rep = classification_report(y_ho, y_ho_pred, digits=3)

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

    info("Réentraînement sur TOUT (train + val) pour le modèle final...")
    X_final = pd.concat([X_tr, X_va], axis=0)
    y_final = np.concatenate([y_tr, y_va])

    rf_final = build_model(cfg)
    rf_final.fit(X_final, y_final)

    joblib.dump(rf_final, MODEL_OUT_PATH)
    ok(f"Modèle final sauvegardé dans {MODEL_OUT_PATH} (c'est celui-là qu'on utilisera)")

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
    ok(f"Les scores sont sauvegardés dans {METRICS_PATH}")

    info("Tout est fini ! On croise les doigts.")


if __name__ == "__main__":
    main()