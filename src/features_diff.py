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

def add_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute quelques features d'interaction simples, interprétables.
    """
    X = X.copy()

    # Exemples: différences déjà présentes (diff_team...) → ratios / densités
    def safe_div(a, b, eps=1e-3):
        return a / (b.abs() + eps)

    # Ratio tirs cadrés / buts (diff)
    if {"diff_teamTEAM_SHOTS_ON_TARGET_season_sum",
        "diff_teamTEAM_GOALS_season_sum"}.issubset(X.columns):
        X["int_diff_shots_on_target_per_goal"] = safe_div(
            X["diff_teamTEAM_SHOTS_ON_TARGET_season_sum"],
            X["diff_teamTEAM_GOALS_season_sum"],
        )

    # Possession * tirs totaux (mesure de domination)
    if {"diff_teamTEAM_BALL_POSSESSION_season_average",
        "diff_teamTEAM_SHOTS_TOTAL_season_sum"}.issubset(X.columns):
        X["int_diff_possession_x_shots_total"] = (
            X["diff_teamTEAM_BALL_POSSESSION_season_average"]
            * X["diff_teamTEAM_SHOTS_TOTAL_season_sum"]
        )

    # Cartons / fautes (discipline)
    if {"diff_teamTEAM_YELLOWCARDS_season_sum",
        "diff_teamTEAM_FOULS_season_sum"}.issubset(X.columns):
        X["int_diff_cards_per_foul"] = safe_div(
            X["diff_teamTEAM_YELLOWCARDS_season_sum"],
            X["diff_teamTEAM_FOULS_season_sum"],
        )

    # Tirs dans la surface / tirs totaux (qualité des occasions)
    if {"diff_teamTEAM_SHOTS_INSIDEBOX_season_sum",
        "diff_teamTEAM_SHOTS_TOTAL_season_sum"}.issubset(X.columns):
        X["int_diff_shots_insidebox_ratio"] = safe_div(
            X["diff_teamTEAM_SHOTS_INSIDEBOX_season_sum"],
            X["diff_teamTEAM_SHOTS_TOTAL_season_sum"],
        )

    print(f"[debug] Features après interactions: {X.shape}")
    return X