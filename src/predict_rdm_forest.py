from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib

# ----------------- chemins depuis src/ -----------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TEST_X_PATH = PROC / "test_merged.csv"
MODEL_PATH  = MODELS_DIR / "random_forest.pkl"
FEATS_PATH  = MODELS_DIR / "rf_feature_importances.csv"

# sortie
OUT_PATH = PROC / "submission.csv"

# -> IMPORTANT: one-hot (1/0) au lieu de probabilités
HARD_LABELS = True

# (optionnel) CSV modèle pour imposer l'ordre de colonnes
TEMPLATE_PATH: Optional[Path] = None
# TEMPLATE_PATH = BASE_DIR / "Data" / "Y_test_random_sEE2QeA.csv"

def info(msg: str) -> None:
    print(f"[info] {msg}")

def ok(msg: str) -> None:
    print(f"[ok] {msg}")

def _encode_categoricals(df: pd.DataFrame, max_cardinality: int = 50) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    keep, drop = [], []
    for c in cat_cols:
        nun = int(df[c].nunique(dropna=True))
        (keep if nun <= max_cardinality else drop).append(c)
    if drop:
        info(f"Dropping high-cardinality categoricals: {drop[:10]}{' ...' if len(drop)>10 else ''}")
    df_ = df.drop(columns=drop, errors="ignore")
    if keep:
        enc = pd.get_dummies(df_[keep], drop_first=True, dummy_na=False)
        rest = df_.drop(columns=keep, errors="ignore")
        out = pd.concat([rest, enc], axis=1)
    else:
        out = df_
    return out

def load_expected_feature_list() -> List[str]:
    if not FEATS_PATH.exists():
        raise FileNotFoundError(f"{FEATS_PATH} introuvable. Entraîne d'abord le modèle.")
    imp = pd.read_csv(FEATS_PATH, low_memory=False)
    if "feature" not in imp.columns:
        raise ValueError(f"{FEATS_PATH} doit contenir la colonne 'feature'.")
    feats = imp["feature"].tolist()
    if not feats:
        raise ValueError("Liste de features vide dans rf_feature_importances.csv.")
    return feats

def build_X_test(test_df: pd.DataFrame, expected_features: List[str], max_cardinality: int = 50) -> np.ndarray:
    if "ID" not in test_df.columns:
        raise ValueError("test_merged.csv doit contenir la colonne 'ID'.")
    feats = test_df.drop(columns=["ID"], errors="ignore")
    num = feats.select_dtypes(include=[np.number])
    non_num = feats.drop(columns=num.columns, errors="ignore")
    if not non_num.empty:
        non_num_enc = _encode_categoricals(non_num, max_cardinality=max_cardinality)
        feats_final = pd.concat([num, non_num_enc], axis=1)
    else:
        feats_final = num
    X_aligned = feats_final.reindex(columns=expected_features, fill_value=0.0)
    X_np = X_aligned.fillna(0.0).to_numpy(dtype=np.float32)
    ok(f"X_test: {X_np.shape} (features alignées: {len(expected_features)})")
    return X_np

def load_template_header(template_path: Path) -> Optional[List[str]]:
    try:
        df = pd.read_csv(template_path, nrows=0)
        return df.columns.tolist()
    except Exception as e:
        info(f"Template non lisible ({e}) → ordre par défaut.")
        return None

def main() -> None:
    # 1) charger test
    if not TEST_X_PATH.exists():
        raise FileNotFoundError(f"Fichier test introuvable: {TEST_X_PATH}")
    test = pd.read_csv(TEST_X_PATH, low_memory=False)
    ok(f"test_merged.csv: {test.shape}")

    # 2) charger modèle
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    ok(f"Modèle chargé: {MODEL_PATH.name}")

    # 3) features attendues
    expected_features = load_expected_feature_list()

    # 4) construire X_test
    X_test = build_X_test(test, expected_features, max_cardinality=50)

    # 5) prédire
    if not hasattr(model, "predict_proba"):
        raise ValueError("Le modèle ne supporte pas predict_proba().")
    proba = model.predict_proba(X_test)  # (n, 3)
    if proba.shape[1] != 3:
        raise ValueError(f"Le modèle ne renvoie pas 3 classes (shape={proba.shape}).")

    # 6) construire DataFrame de sortie
    if HARD_LABELS:
        # argmax -> one-hot
        idx = proba.argmax(axis=1)
        onehot = np.zeros_like(proba, dtype=np.int8)
        onehot[np.arange(len(idx)), idx] = 1
        sub = pd.DataFrame({
            "ID": test["ID"].values,
            "HOME_WINS": onehot[:, 0],
            "DRAW":      onehot[:, 1],
            "AWAY_WINS": onehot[:, 2],
        })
    else:
        # (fallback) probabilités
        sub = pd.DataFrame({
            "ID": test["ID"].values,
            "HOME_WINS": proba[:, 0],
            "DRAW":      proba[:, 1],
            "AWAY_WINS": proba[:, 2],
        })

    # 7) option: imposer l'ordre du template
    if TEMPLATE_PATH is not None and TEMPLATE_PATH.exists():
        cols = load_template_header(TEMPLATE_PATH)
        needed = ["ID","HOME_WINS","DRAW","AWAY_WINS"]
        if cols and set(needed).issubset(cols):
            keep = [c for c in cols if c in sub.columns]
            sub = sub.reindex(columns=keep)

    # 8) écrire CSV (valeurs int 0/1)
    sub = sub.astype({"HOME_WINS": "int8", "DRAW": "int8", "AWAY_WINS": "int8"})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)
    ok(f"Submission écrite: {OUT_PATH}")

if __name__ == "__main__":
    main()
