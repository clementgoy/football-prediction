from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import numpy as np
import pandas as pd
import joblib

# ---------- chemins (depuis src/) ----------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

TEST_X_PATH = PROC / "test_merged.csv"

OUT_DIR = PROC

HARD_LABELS = True        
MAX_CARD = 50          

def info(msg: str) -> None:
    print(f"[info] {msg}")

def ok(msg: str) -> None:
    print(f"[ok] {msg}")

# ---------- utils ----------
def _encode_categoricals(df: pd.DataFrame, max_cardinality: int = 50) -> pd.DataFrame:
    """One-hot des colonnes catégorielles de faible cardinalité ; drop le reste."""
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        return df
    keep, drop = [], []
    for c in cat_cols:
        nun = int(df[c].nunique(dropna=True))
        (keep if nun <= max_cardinality else drop).append(c)
    if drop:
        info(f"Drop catégorielles haute cardinalité: {drop[:10]}{' ...' if len(drop)>10 else ''}")
    df2 = df.drop(columns=drop, errors="ignore")
    if keep:
        enc = pd.get_dummies(df2[keep], drop_first=True, dummy_na=False)
        rest = df2.drop(columns=keep, errors="ignore")
        return pd.concat([rest, enc], axis=1)
    return df2

def load_test() -> pd.DataFrame:
    if not TEST_X_PATH.exists():
        raise FileNotFoundError(f"Introuvable: {TEST_X_PATH}")
    df = pd.read_csv(TEST_X_PATH, low_memory=False)
    if "ID" not in df.columns:
        raise ValueError("test_merged.csv doit contenir la colonne 'ID'.")
    ok(f"test_merged: {df.shape}")
    return df

def find_model_tags() -> List[str]:
    """Repère tous les modèles disponibles via leurs fichiers *_features.json."""
    tags = []
    for p in sorted(MODELS.glob("*_features.json")):
        tag = p.name.replace("_features.json", "")
        # on ne garde que ceux qui ont aussi le .joblib correspondant
        if (MODELS / f"{tag}.joblib").exists():
            tags.append(tag)
    if not tags:
        raise SystemError(
            "Aucun modèle détecté dans 'models/'.\n"
            "Attendu: fichiers comme rf_linear025.joblib + rf_linear025_features.json"
        )
    ok(f"Modèles détectés: {tags}")
    return tags

def load_expected_features(tag: str) -> List[str]:
    feat_file = MODELS / f"{tag}_features.json"
    meta = json.loads(feat_file.read_text(encoding="utf-8"))
    feats = meta.get("feature_names")
    if not feats:
        raise ValueError(f"{feat_file} ne contient pas 'feature_names'.")
    return feats

def build_X_for_model(test_df: pd.DataFrame, expected_features: List[str]) -> np.ndarray:
    feats = test_df.drop(columns=["ID"], errors="ignore")

    # Sépare numérique / non-numérique
    num = feats.select_dtypes(include=[np.number])
    non_num = feats.drop(columns=num.columns, errors="ignore")

    if not non_num.empty:
        non_num_enc = _encode_categoricals(non_num, max_cardinality=MAX_CARD)
        feats_final = pd.concat([num, non_num_enc], axis=1)
    else:
        feats_final = num

    # réalignement exact:
    X_aligned = feats_final.reindex(columns=expected_features, fill_value=0.0)
    X_np = X_aligned.fillna(0.0).to_numpy(dtype=np.float32)
    return X_np

def predict_one_model(tag: str, test_df: pd.DataFrame) -> pd.DataFrame:
    # 1) features attendues
    expected = load_expected_features(tag)
    X_test = build_X_for_model(test_df, expected)
    ok(f"[{tag}] X_test shape: {X_test.shape}")

    # 2) modèle
    model_path = MODELS / f"{tag}.joblib"
    clf = joblib.load(model_path)
    if not hasattr(clf, "predict_proba"):
        raise ValueError(f"{tag} ne supporte pas predict_proba().")

    # 3) proba -> 1/0
    proba = clf.predict_proba(X_test)
    if proba.shape[1] != 3:
        raise ValueError(f"{tag}: le modèle ne renvoie pas 3 classes (shape={proba.shape}).")

    if HARD_LABELS:
        idx = proba.argmax(axis=1)
        onehot = np.zeros_like(proba, dtype=np.int8)
        onehot[np.arange(len(idx)), idx] = 1
        sub = pd.DataFrame({
            "ID": test_df["ID"].values,
            "HOME_WINS": onehot[:, 0],
            "DRAW":      onehot[:, 1],
            "AWAY_WINS": onehot[:, 2],
        })
    else:
        sub = pd.DataFrame({
            "ID": test_df["ID"].values,
            "HOME_WINS": proba[:, 0],
            "DRAW":      proba[:, 1],
            "AWAY_WINS": proba[:, 2],
        })

    # types propres pour Challenge Data
    sub = sub.astype({"HOME_WINS": "int8", "DRAW": "int8", "AWAY_WINS": "int8"})
    return sub

def main() -> None:
    test = load_test()
    tags = find_model_tags()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        info(f"Prédiction avec le modèle: {tag}")
        sub = predict_one_model(tag, test)
        out_path = OUT_DIR / f"submission_{tag}.csv"
        sub.to_csv(out_path, index=False)
        ok(f"Fichier écrit: {out_path}")

    ok("Terminé. Un CSV par modèle est dispo dans data/processed/.")

if __name__ == "__main__":
    main()
