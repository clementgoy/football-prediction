import pandas as pd
from .features import build_features

def build_Xy(X_raw: pd.DataFrame, y_raw: pd.DataFrame):
    X = build_features(X_raw)
    # y: convertir one-hot -> classe (0,1,2) si besoin
    if {"HOME","DRAW","AWAY"}.issubset(set(y_raw.columns)):
        y = y_raw[["HOME","DRAW","AWAY"]].values.argmax(axis=1)
    else:
        raise ValueError("y_train doit contenir les colonnes HOME/DRAW/AWAY (one-hot)")
    return X, y