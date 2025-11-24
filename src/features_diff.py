import pandas as pd
from .features import build_features


def build_features_with_diff(df: pd.DataFrame, drop_id_cols: bool = True) -> pd.DataFrame:

    X = build_features(df, drop_id_cols=drop_id_cols).copy()

    numeric_cols = set(X.select_dtypes(include="number").columns)

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

    X = X.copy()

    def safe_div(a, b, eps=1e-3):
        return a / (b.abs() + eps)

    if {"diff_teamTEAM_SHOTS_ON_TARGET_season_sum",
        "diff_teamTEAM_GOALS_season_sum"}.issubset(X.columns):
        X["int_diff_shots_on_target_per_goal"] = safe_div(
            X["diff_teamTEAM_SHOTS_ON_TARGET_season_sum"],
            X["diff_teamTEAM_GOALS_season_sum"],
        )

    if {"diff_teamTEAM_BALL_POSSESSION_season_average",
        "diff_teamTEAM_SHOTS_TOTAL_season_sum"}.issubset(X.columns):
        X["int_diff_possession_x_shots_total"] = (
            X["diff_teamTEAM_BALL_POSSESSION_season_average"]
            * X["diff_teamTEAM_SHOTS_TOTAL_season_sum"]
        )

    if {"diff_teamTEAM_YELLOWCARDS_season_sum",
        "diff_teamTEAM_FOULS_season_sum"}.issubset(X.columns):
        X["int_diff_cards_per_foul"] = safe_div(
            X["diff_teamTEAM_YELLOWCARDS_season_sum"],
            X["diff_teamTEAM_FOULS_season_sum"],
        )

    if {"diff_teamTEAM_SHOTS_INSIDEBOX_season_sum",
        "diff_teamTEAM_SHOTS_TOTAL_season_sum"}.issubset(X.columns):
        X["int_diff_shots_insidebox_ratio"] = safe_div(
            X["diff_teamTEAM_SHOTS_INSIDEBOX_season_sum"],
            X["diff_teamTEAM_SHOTS_TOTAL_season_sum"],
        )

    print(f"[debug] Features après interactions: {X.shape}")
    return X