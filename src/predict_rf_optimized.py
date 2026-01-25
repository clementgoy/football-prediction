from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TEST_X_PATH = PROC / "test_merged.csv"
MODEL_PATH = MODELS_DIR / "random_forest.pkl"
FEATS_PATH = MODELS_DIR / "rf_feature_importances.csv"
SUBMISSION_PATH = MODELS_DIR / "submission_rf_optimized.csv"


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
        raise FileNotFoundError(f"Je trouve pas le fichier test : {TEST_X_PATH}")

    info("Chargement des données de test (pour voir ce qu'on vaut)...")
    df = pd.read_csv(TEST_X_PATH, low_memory=False)
    if "ID" not in df.columns:
        raise ValueError("Y'a pas de colonne ID dans le test, c'est grave !")

    ids = df["ID"].values

    feats = df.drop(columns=["ID"], errors="ignore")
    # On garde que les chifres
    X_num = feats.select_dtypes(include=[np.number]).copy()

    info(f"Test chargé : {len(X_num)} lignes à prédire.")
    return ids, X_num


def main() -> None:
    info("Lancement des prédictions (RandomForest optimisée)...")

    feature_list = load_expected_feature_list()
    info(f"Le modèle attend {len(feature_list)} features précises.")

    ids, X_test_raw = load_test_data()

    # On remplit les trous (si y'a des features manquantes dans le test, on met 0)
    X_test = X_test_raw.reindex(columns=feature_list, fill_value=0.0)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Le modèle est pas là : {MODEL_PATH}")
    info(f"Chargement du modèle {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    info("Calcul des probabilités...")
    proba = model.predict_proba(X_test)  

    classes = np.argmax(proba, axis=1)  

    y_onehot = np.zeros_like(proba, dtype=int)
    y_onehot[np.arange(len(classes)), classes] = 1

    sub = pd.DataFrame(
        {
            "ID": ids,
            "HOME_WINS": y_onehot[:, 0],
            "DRAW": y_onehot[:, 1],
            "AWAY_WINS": y_onehot[:, 2],
        }
    )

    sub.to_csv(SUBMISSION_PATH, index=False)
    ok(f"C'est bon ! Fichier prêt à être envoyé : {SUBMISSION_PATH}")

    info("Terminé.")


if __name__ == "__main__":
    main()