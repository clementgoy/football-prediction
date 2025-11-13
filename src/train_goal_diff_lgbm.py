#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from lightgbm import LGBMRegressor
import joblib

from src.print_result import print_report
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

    # Features avec colonnes home/away + diff
    X = build_features_with_diff(X_raw, drop_id_cols=True)

    # Sécurité si jamais ID reste
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    # On ne garde que les features numériques
    X = X.select_dtypes(include=["number"]).copy()
    print(f"[debug] Features numériques après filtrage: {X.shape}")

    # Cible
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
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="CSV des features train fusionnées (ex: data/processed/train_merged.csv)",
    )
    parser.add_argument(
        "--y-supp-csv",
        type=str,
        required=True,
        help="Y_train_supp.csv contenant GOAL_DIFF_HOME_AWAY",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        required=True,
        help="Chemin de sortie du modèle LightGBM (pkl)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Part de validation (par défaut 0.2)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # 1) Chargement X, y
    X, y, ids = build_Xy_goal_diff(args.train_csv, args.y_supp_csv)

    # 2) Split train / val
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    print(f"[info] X_train: {X_tr.shape}, X_val: {X_va.shape}")

    # 3) Modèle LGBM sur le goal diff
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

    # 4) Performance RMSE sur le goal diff (régression)
    y_va_pred = reg.predict(X_va)
    mse_va = mean_squared_error(y_va, y_va_pred)
    rmse = np.sqrt(mse_va)
    print(f"\n[VAL] RMSE (goal diff): {rmse:.4f}")

    # 5) On transforme en classes pour réutiliser print_report (0/1/2)
    y_tr_pred = reg.predict(X_tr)

    y_tr_cls_true = goal_diff_to_class(y_tr)
    y_tr_cls_pred = goal_diff_to_class(y_tr_pred)
    y_va_cls_true = goal_diff_to_class(y_va)
    y_va_cls_pred = goal_diff_to_class(y_va_pred)

    train_acc = accuracy_score(y_tr_cls_true, y_tr_cls_pred)
    val_acc = accuracy_score(y_va_cls_true, y_va_cls_pred)
    # Pas de hold-out ici → on peut mettre None ou np.nan
    hold_acc = None

    cm = confusion_matrix(y_va_cls_true, y_va_cls_pred)
    clf_report = classification_report(y_va_cls_true, y_va_cls_pred, digits=3)

    # 6) Top features à partir des importances LGBM
    importances = reg.feature_importances_
    feature_names = np.array(X.columns)
    order = np.argsort(importances)[::-1]
    top_features = feature_names[order[:20]].tolist()

    print_report(
        train_acc=train_acc,
        val_acc=val_acc,
        hold_acc=hold_acc,
        cm=cm,
        clf_report=clf_report,
        top_features=top_features,
        X=X,
        X_tr_sel=X_tr,
        X_va_sel=X_va,
        X_ho_sel=X_va, 
    )

    # 8) Sauvegarde du modèle
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, out_path)
    print(f"\n[ok] Modèle LightGBM Regressor sauvegardé dans {out_path.resolve()}")


if __name__ == "__main__":
    main()