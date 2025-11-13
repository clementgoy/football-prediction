#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from .features import build_features


def build_features_with_diff(df: pd.DataFrame, drop_id_cols: bool = True) -> pd.DataFrame:
    """
    Construit les features de base (via src.features.build_features)
    puis ajoute des features de type "home - away" pour chaque paire.

    On ne calcule les différences QUE pour les colonnes numériques afin
    d'éviter les erreurs du type 'str' - 'str'.
    """
    # 1) Features de base
    X = build_features(df, drop_id_cols=drop_id_cols).copy()

    # 2) Liste des colonnes numériques
    numeric_cols = set(X.select_dtypes(include="number").columns)

    # 3) Ajout des features différentielles home-away
    cols = list(X.columns)
    for c in cols:
        if not c.startswith("home_"):
            continue

        suffix = c[len("home_"):]
        away_col = "away_" + suffix

        if away_col not in X.columns:
            continue

        if c not in numeric_cols or away_col not in numeric_cols:
            continue

        diff_name = f"diff_{suffix}"
        X[diff_name] = X[c] - X[away_col]

    return X