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
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Fichier de features introuvable: {FEATURES_PATH}")

    info(f"Chargement du modèle depuis {MODEL_PATH} …")
    clf = joblib.load(MODEL_PATH)

    info(f"Chargement des features depuis {FEATURES_PATH} …")
    feat_data = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    feature_names = feat_data["feature_names"]

    ok(f"Modèle et features chargés ({len(feature_names)} colonnes).")
    return clf, feature_names


def load_test_data() -> pd.DataFrame:
    """Charge le fichier test_merged.csv."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Introuvable: {TEST_PATH}")

    info("Chargement de test_merged.csv …")
    df_test = pd.read_csv(TEST_PATH, low_memory=False)
    ok(f"Test: {df_test.shape[0]} lignes × {df_test.shape[1]} colonnes")

    if "ID" not in df_test.columns:
        raise ValueError("La colonne 'ID' doit exister dans test_merged.csv")

    return df_test


def prepare_test_features(df_test: pd.DataFrame, feature_names):
    
    # Aligne les colonnes du test sur celles utilisées à l'entraînement, ajoute les colonnes manquantes (remplies à 0.0), ignore les colonnes en plus, remplace les NaN par 0.0.
    
    info("Préparation des features du test …")

    X_test = df_test.copy()

    # Ajout des colonnes manquantes
    missing_cols = [c for c in feature_names if c not in X_test.columns]
    if missing_cols:
        info(f"Ajout de {len(missing_cols)} colonnes manquantes (remplies à 0.0).")
        for c in missing_cols:
            X_test[c] = 0.0

    X_test = X_test[feature_names].copy()

    X_test = X_test.fillna(0.0)

    ok(f"Features test préparées: {X_test.shape[0]} lignes × {X_test.shape[1]} colonnes")
    return X_test


def make_submission(df_test: pd.DataFrame, clf, X_test: pd.DataFrame):

    # Fait les prédictions, convertit en one-hot et génère le DataFrame

    info("Prédiction des classes …")
    y_pred = clf.predict(X_test)

    # Conversion en one-hot
    n_samples = len(y_pred)
    one_hot = np.zeros((n_samples, 3), dtype=int)
    for i, cls in enumerate(y_pred):
        if cls not in (0, 1, 2):
            raise ValueError(f"Classe prédite inattendue: {cls}")
        one_hot[i, cls] = 1

    # Construction du DataFrame 
    sub = pd.DataFrame({
        "ID": df_test["ID"].values,
        "HOME_WINS": one_hot[:, 0],
        "DRAW":      one_hot[:, 1],
        "AWAY_WINS": one_hot[:, 2],
    })

    ok("Soumission générée (one-hot 0/1).")
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
