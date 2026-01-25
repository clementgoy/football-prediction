from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"

TEST_PATH = PROCESSED / "test_merged.csv"

MODEL_TAG = "rf_none" 
MODEL_PATH = MODELS / f"{MODEL_TAG}.joblib"
FEATURES_PATH = MODELS / f"{MODEL_TAG}_features.json"

SUBMISSION_PATH = PROCESSED / f"submission_{MODEL_TAG}.csv"


def info(msg: str) -> None:
    print(f"[info] {msg}")


def ok(msg: str) -> None:
    print(f"[ok] {msg}")


def load_model_and_features():
    """Charge le modèle entraîné et la liste de features utilisées."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Je trouve pas le modèle : {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Je trouve pas le fichier de features : {FEATURES_PATH}")

    info(f"On charge le modèle depuis {MODEL_PATH} …")
    clf = joblib.load(MODEL_PATH)

    info(f"On charge la liste des features depuis {FEATURES_PATH} …")
    feat_data = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    feature_names = feat_data["feature_names"]

    ok(f"C'est tout bon ! Modèle et features chargés ({len(feature_names)} colonnes).")
    return clf, feature_names


def load_test_data() -> pd.DataFrame:
    """Charge le fichier test_merged.csv."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Je trouve pas le test : {TEST_PATH}")

    info("On charge le fichier de test …")
    df_test = pd.read_csv(TEST_PATH, low_memory=False)
    ok(f"Test chargé : {df_test.shape[0]} lignes × {df_test.shape[1]} colonnes")

    if "ID" not in df_test.columns:
        raise ValueError("Y'a pas de colonne ID, c'est pas normal !")

    return df_test


def prepare_test_features(df_test: pd.DataFrame, feature_names):
    # Aligne les colonnes du test sur celles utilisées à l'entraînement
    info("On prépare les colonnes pour le test …")

    X_test = df_test.copy()

    # Ajout des colonnes manquantes (si y'en a une qui manque dans le test, on met 0)
    missing_cols = [c for c in feature_names if c not in X_test.columns]
    if missing_cols:
        info(f"Oups, il manque {len(missing_cols)} colonnes. On les rajoute avec des 0.")
        for c in missing_cols:
            X_test[c] = 0.0

    X_test = X_test[feature_names].copy()

    # On remplace les trous par des 0 (comme à l'entraînement)
    X_test = X_test.fillna(0.0)

    ok(f"Features prêtes ! On a bien {X_test.shape[0]} lignes × {X_test.shape[1]} colonnes.")
    return X_test


def make_submission(df_test: pd.DataFrame, clf, X_test: pd.DataFrame):
    # Fait les prédictions
    # Ici on utilise predict() directement qui renvoie la classe (0, 1 ou 2)
    info("Calcul des prédictions …")
    y_pred = clf.predict(X_test)

    # Conversion en one-hot
    n_samples = len(y_pred)
    one_hot = np.zeros((n_samples, 3), dtype=int)
    for i, cls in enumerate(y_pred):
        if cls not in (0, 1, 2):
            raise ValueError(f"Euh, le modèle a prédit {cls}, je connais pas cette classe.")
        one_hot[i, cls] = 1

    # On construit le tableau final
    sub = pd.DataFrame({
        "ID": df_test["ID"].values,
        "HOME_WINS": one_hot[:, 0],
        "DRAW":      one_hot[:, 1],
        "AWAY_WINS": one_hot[:, 2],
    })

    ok("Soumission générée (c'est des 0 et des 1).")
    return sub


def main():
    clf, feature_names = load_model_and_features()
    df_test = load_test_data()
    X_test = prepare_test_features(df_test, feature_names)

    submission_df = make_submission(df_test, clf, X_test)

    info(f"Enregistrement de la soumission dans {SUBMISSION_PATH} …")
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    ok(f"Fichier de soumission créé : {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
