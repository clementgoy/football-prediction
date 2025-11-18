from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Chemins
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TEST_X_PATH = PROC / "test_merged.csv"
MODEL_PATH = MODELS_DIR / "random_forest.pkl"
FEATS_PATH = MODELS_DIR / "rf_feature_importances.csv"
SUBMISSION_PATH = MODELS_DIR / "submission_rf_optimized.csv"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def info(msg: str) -> None:
    print(f"\n[info] {msg}")


def ok(msg: str) -> None:
    print(f"[ok] {msg}")


def load_expected_feature_list() -> List[str]:
    if not FEATS_PATH.exists():
        raise FileNotFoundError(
            f"{FEATS_PATH} introuvable. Entraîne d'abord le modèle optimisé."
        )
    imp = pd.read_csv(FEATS_PATH, low_memory=False)
    if "feature" not in imp.columns:
        raise ValueError(f"{FEATS_PATH} doit contenir une colonne 'feature'.")
    feats = imp["feature"].tolist()
    if not feats:
        raise ValueError("Liste de features vide dans rf_feature_importances.csv.")
    return feats


def load_test_data() -> Tuple[np.ndarray, pd.DataFrame]:
    if not TEST_X_PATH.exists():
        raise FileNotFoundError(f"Fichier test introuvable: {TEST_X_PATH}")

    info("Chargement des données de test...")
    df = pd.read_csv(TEST_X_PATH, low_memory=False)
    if "ID" not in df.columns:
        raise ValueError("La table de test doit contenir la colonne ID.")

    ids = df["ID"].values

    # On garde seulement les colonnes numériques, sans ID
    feats = df.drop(columns=["ID"], errors="ignore")
    X_num = feats.select_dtypes(include=[np.number]).copy()

    info(f"Test: {len(X_num)} lignes, {X_num.shape[1]} features numériques.")
    return ids, X_num


def main() -> None:
    info("Prédiction avec la RandomForest optimisée")

    # 1) Features attendues
    feature_list = load_expected_feature_list()
    info(f"{len(feature_list)} features attendues (sélectionnées à l'entraînement).")

    # 2) Données de test
    ids, X_test_raw = load_test_data()

    # On aligne les colonnes sur la liste de features utilisée à l'entraînement
    X_test = X_test_raw.reindex(columns=feature_list, fill_value=0.0)

    # 3) Chargement du modèle
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")
    info(f"Chargement du modèle depuis {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # 4) Prédiction des probabilités
    info("Prédiction des probabilités...")
    proba = model.predict_proba(X_test)  # (n_samples, 3)

    # 5) Conversion en prédiction one-hot (0/1) comme dans l'exemple du challenge
    classes = np.argmax(proba, axis=1)  # 0 = home, 1 = draw, 2 = away

    y_onehot = np.zeros_like(proba, dtype=int)
    y_onehot[np.arange(len(classes)), classes] = 1

    # 6) Construction du fichier de soumission au format attendu (0/1)
    sub = pd.DataFrame(
        {
            "ID": ids,
            "HOME_WINS": y_onehot[:, 0],
            "DRAW": y_onehot[:, 1],
            "AWAY_WINS": y_onehot[:, 2],
        }
    )

    sub.to_csv(SUBMISSION_PATH, index=False)
    ok(f"Fichier de soumission sauvegardé dans {SUBMISSION_PATH}")

    info("Prédiction terminée ")


if __name__ == "__main__":
    main()