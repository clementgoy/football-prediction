import pandas as pd

"""Features are already standardized in the challenge data.
Functions to build features from raw data.
"""
def build_features(df: pd.DataFrame, drop_id_cols=True):
    cols = list(df.columns)
    id_like = [c for c in cols if "MATCH" in c or "GAME" in c or "TEAM_ID" in c or "PLAYER_ID" in c or "LEAGUE_ID" in c]
    if drop_id_cols:
        df = df.drop(columns=[c for c in id_like if c in df.columns], errors="ignore")
    return df