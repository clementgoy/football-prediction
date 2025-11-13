#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from lightgbm import LGBMRegressor
import joblib

from src.features_diff import build_features_with_diff


def build_Xy_goal_diff(train_csv: str, y_supp_csv: str):
    """
    Construit X (features enrichies) et y (GOAL_DIFF_HOME_AWAY).

    - train_csv : CSV fusionné, ex: data/processed/train_merged.csv
      (doit contenir une colonne 'ID')
    - y_supp_csv : Y_train_supp_aligned.csv (colonnes ['ID', 'GOAL_DIFF_HOME_AWAY'])
    """
    X_raw = pd.read_csv(train_csv)
    if "ID" not in X_raw.columns:
        raise ValueError("Le fichier train_csv doit contenir une colonne 'ID'.")

    ids = X_raw["ID"].values

    X = build_features_with_diff(X_raw, drop_id_cols=True)

    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    X = X.select_dtypes(include=["number"]).copy()
    print(f"[debug] Features numériques après filtrage: {X.shape}")
    
    y_supp = pd.read_csv(y_supp_csv)
    if not {"ID", "GOAL_DIFF_HOME_AWAY"}.issubset(y_supp.columns):
        raise ValueError("y_supp_csv doit contenir les colonnes 'ID' et 'GOAL_DIFF_HOME_AWAY'.")

    y_supp = y_supp.set_index("ID").loc[ids]
    y = y_supp["GOAL_DIFF_HOME_AWAY"].astype(float).values

    return X, y, ids


def goal_diff_to_class(diff: np.ndarray) -> np.ndarray:
    """
    Convertit un goal diff en classe:
        > 0 → 0 (HOME_WINS)
        = 0 → 1 (DRAW)
        < 0 → 2 (AWAY_WINS)
    """
    diff = np.asarray(diff)
    cls = np.zeros_like(diff, dtype=int)
    cls[diff < 0] = 2
    cls[diff == 0] = 1
    return cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, required=True,
                        help="CSV des features train fusionnées (ex: data/processed/train_merged.csv)")
    parser.add_argument("--y-supp-csv", type=str, required=True,
                        help="Y_train_supp.csv contenant GOAL_DIFF_HOME_AWAY")
    parser.add_argument("--model-out", type=str, required=True,
                        help="Chemin de sortie du modèle LightGBM (pkl)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Part de validation (par défaut 0.2)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, y, ids = build_Xy_goal_diff(args.train_csv, args.y_supp_csv)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    print(f"[info] X_train: {X_tr.shape}, X_val: {X_va.shape}")

    reg = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.random_state,
        n_jobs=-1,
    )

    reg.fit(X_tr, y_tr)

    y_va_pred = reg.predict(X_va)
    rmse = mean_squared_error(y_va, y_va_pred, squared=False)
    print(f"\n[VAL] RMSE (goal diff): {rmse:.4f}")

    y_va_cls_true = goal_diff_to_class(y_va)
    y_va_cls_pred = goal_diff_to_class(y_va_pred)

    acc = accuracy_score(y_va_cls_true, y_va_cls_pred)
    print(f"[VAL] Accuracy (via goal diff → résultat): {acc:.4f}")

    cm = confusion_matrix(y_va_cls_true, y_va_cls_pred)
    print("\n[VAL] Confusion matrix (0=HOME,1=DRAW,2=AWAY):")
    print(cm)

    print("\n[VAL] Classification report:")
    print(classification_report(y_va_cls_true, y_va_cls_pred, digits=3))

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, out_path)
    print(f"\n[ok] Modèle LightGBM Regressor sauvegardé dans {out_path.resolve()}")


if __name__ == "__main__":
    main()