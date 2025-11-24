#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

CLASS_NAMES_TRAIN = ["HOME_WINS", "DRAW", "AWAY_WINS"]
SUBMIT_COLS = ["HOME", "DRAW", "AWAY"]                  

def parse_args():
    p = argparse.ArgumentParser(description="Predict with LightGBM model trained by train_lgbm.py.")
    p.add_argument("--test-csv", required=True, help="CSV test fusionné contenant la colonne ID.")
    p.add_argument("--model", default="outputs/models/lgbm.pkl", help="Chemin du modèle LightGBM (joblib/pkl).")
    p.add_argument("--out-csv", required=True, help="Chemin du CSV de soumission à écrire.")
    p.add_argument("--id-col", default="ID", help="Nom de la colonne identifiant (défaut: ID).")
    p.add_argument("--submit-onehot", action="store_true",
                   help="Écrit un one-hot (classe argmax) au lieu des probabilités.")
    p.add_argument("--alpha-draw", type=float, default=1.0,
                   help="Facteur multiplicatif sur la proba DRAW (ré-étalonnage facultatif).")
    return p.parse_args()

def coerce_numeric_like_train(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == id_col:
            continue
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    num_cols = [c for c in out.columns if c != id_col]
    out[num_cols] = out[num_cols].astype("float32")
    return out

def get_model_feature_names(model) -> list:
    if hasattr(model, "booster_") and model.booster_ is not None:
        names = list(model.booster_.feature_name())
        if names and all(isinstance(x, str) for x in names):
            return names
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    raise RuntimeError("Impossible de récupérer les noms de features du modèle (booster_.feature_name()).")

def align_features_to_model(feats: pd.DataFrame, model_feature_names: list) -> pd.DataFrame:
    X = feats.copy()
    for c in model_feature_names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[model_feature_names]
    return X

def apply_draw_alpha(proba: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 1.0:
        return proba
    p = proba.copy()
    p[:, 1] *= alpha
    p = p / p.sum(axis=1, keepdims=True)
    return p

def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[err] Modèle introuvable: {model_path}", file=sys.stderr)
        sys.exit(2)
    model = joblib.load(model_path)

    test_df = pd.read_csv(args.test_csv)
    if args.id_col not in test_df.columns:
        raise ValueError(f"Colonne identifiante '{args.id_col}' introuvable dans {args.test_csv}")
    ids = test_df[args.id_col].copy()

    test_df = coerce_numeric_like_train(test_df, args.id_col)
    feats = test_df.drop(columns=[args.id_col], errors="ignore")

    model_feats = get_model_feature_names(model)
    X = align_features_to_model(feats, model_feats)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Le modèle chargé ne supporte pas predict_proba().")
    proba = model.predict_proba(X)

    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if set(classes) != {0, 1, 2}:
            raise RuntimeError(f"Classes inattendues dans le modèle: {classes} (attendu: [0,1,2])")

    proba = apply_draw_alpha(proba, args.alpha_draw)

    if args.submit_onehot:
        argmax = proba.argmax(axis=1)
        onehot = np.zeros_like(proba, dtype=int)
        onehot[np.arange(len(argmax)), argmax] = 1
        submit_body = pd.DataFrame(onehot, columns=SUBMIT_COLS)
    else:
        submit_body = pd.DataFrame(proba, columns=SUBMIT_COLS)

    submit = pd.concat([ids.reset_index(drop=True), submit_body], axis=1)
    submit.columns = [args.id_col] + SUBMIT_COLS

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submit.to_csv(out_path, index=False)
    print(f"[ok] Soumission écrite -> {out_path.resolve()}")

if __name__ == "__main__":
    main()
