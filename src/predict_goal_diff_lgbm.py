#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.features_diff import build_features_with_diff

SUBMIT_COLS = ["HOME_WINS", "DRAW", "AWAY_WINS"]


def goal_diff_to_onehot(diff: np.ndarray) -> np.ndarray:
    """
    Transforme un goal diff en one-hot [HOME,DRAW,AWAY].

    On utilise juste le signe :
        > 0 -> [1,0,0]
        = 0 -> [0,1,0]
        < 0 -> [0,0,1]
    """
    diff = np.asarray(diff)
    n = diff.shape[0]
    proba = np.zeros((n, 3), dtype=float)

    home_mask = diff > 0
    draw_mask = diff == 0
    away_mask = diff < 0

    proba[home_mask, 0] = 1.0
    proba[draw_mask, 1] = 1.0
    proba[away_mask, 2] = 1.0

    return proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=str, required=True,
                        help="CSV de test fusionné (ex: data/processed/test_merged.csv)")
    parser.add_argument("--model", type=str, required=True,
                        help="Modèle LightGBM Regressor entraîné (pkl)")
    parser.add_argument("--out-csv", type=str, required=True,
                        help="Chemin de sortie de la soumission")
    parser.add_argument("--id-col", type=str, default="ID",
                        help="Nom de la colonne identifiant (par défaut: ID)")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv)
    if args.id_col not in test_df.columns:
        raise ValueError(f"La colonne id '{args.id_col}' n'est pas présente dans le test CSV.")

    ids = test_df[args.id_col].copy()

    # Même pipeline de features que pour le train
    X_test = build_features_with_diff(test_df, drop_id_cols=True)
    if args.id_col in X_test.columns:
        X_test = X_test.drop(columns=[args.id_col])

    X_test = X_test.select_dtypes(include=["number"]).copy()
    print(f"[debug] Features numériques après filtrage: {X_test.shape}")

    # Chargement du modèle
    reg = joblib.load(args.model)

    # Prédiction du goal diff
    diff_pred = reg.predict(X_test)

    # Conversion en one-hot
    proba = goal_diff_to_onehot(diff_pred)

    # Construction de la soumission
    submit_body = pd.DataFrame(proba, columns=SUBMIT_COLS)
    submit = pd.concat([ids.reset_index(drop=True), submit_body], axis=1)
    submit.columns = [args.id_col] + SUBMIT_COLS

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submit.to_csv(out_path, index=False)
    print(f"[ok] Soumission écrite -> {out_path.resolve()}")


if __name__ == "__main__":
    main()