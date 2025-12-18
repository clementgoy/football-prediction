from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer


# ======================
# Paths
# ======================
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

OUTPUTS.mkdir(parents=True, exist_ok=True)

X_TEST_PATH = PROCESSED / "test_merged.csv"
MODEL_PATH = MODELS / "rf_stratkfold.joblib"
OUT_PATH = OUTPUTS / "y_test_rf_stratkfold.csv"


def info(msg: str) -> None:
    print(f"\n[info] {msg}")


def ok(msg: str) -> None:
    print(f"[ok] {msg}")


# ======================
# Load test data
# ======================
def load_test_data() -> pd.DataFrame:
    if not X_TEST_PATH.exists():
        raise FileNotFoundError(f"Fichier test introuvable : {X_TEST_PATH}")

    info("Chargement des features test …")
    X_test = pd.read_csv(X_TEST_PATH, low_memory=False)
    ok(f"X_test: {X_test.shape[0]} lignes × {X_test.shape[1]} colonnes")

    return X_test


# ======================
# Prepare features
# ======================
def prepare_features(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "ID" not in X.columns:
        raise ValueError("La colonne ID est manquante dans le test set")

    ids = X["ID"].copy()

    # garder uniquement les colonnes numériques (comme à l'entraînement)
    X_num = X.drop(columns=["ID"], errors="ignore")
    X_num = X_num.select_dtypes(include=[np.number])

    info(f"Features numériques utilisées : {X_num.shape[1]}")

    # même stratégie d’imputation que pour le train
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_num),
        columns=X_num.columns,
        index=X_num.index,
    )

    return X_imp, ids


# ======================
# Predict
# ======================
def main() -> None:
    # 1. Charger le modèle
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    info("Chargement du modèle RandomForest (StratifiedKFold)…")
    clf = joblib.load(MODEL_PATH)
    ok("Modèle chargé")

    # 2. Charger les données test
    X_test_raw = load_test_data()

    # 3. Préparer les features
    X_test, ids = prepare_features(X_test_raw)

    # 4. Prédiction (classes 0,1,2)
    info("Prédiction sur le test set …")
    y_pred = clf.predict(X_test)

    # 5. Conversion en one-hot (format Challenge Data)
    y_out = pd.DataFrame({
        "ID": ids.values,
        "HOME_WINS": (y_pred == 0).astype(int),
        "DRAW": (y_pred == 1).astype(int),
        "AWAY_WINS": (y_pred == 2).astype(int),
    })

    # vérification sécurité
    assert (y_out[["HOME_WINS", "DRAW", "AWAY_WINS"]].sum(axis=1) == 1).all()

    # 6. Sauvegarde
    y_out.to_csv(OUT_PATH, index=False)
    ok(f"Fichier de soumission généré : {OUT_PATH}")
    ok(f"{len(y_out)} lignes écrites")


if __name__ == "__main__":
    main()
