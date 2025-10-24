# src/data_loading.py
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict

# ---------- CONFIG PAR DÉFAUT (peut être surchargé par base.yaml) ----------
DEFAULT_PATHS = {
    "train_home": "data/train_home_team_statistics_df.csv",
    "train_away": "data/train_away_team_statistics_df.csv",
    "test_home":  "data/test_home_team_statistics_df.csv",
    "test_away":  "data/test_away_team_statistics_df.csv",
    "y_train":    "data/Y_train_1rknArQ.csv",  # contient id + [home, draw, away]
}

TARGET_COL_CANDIDATES = ["home","Home","HOME","draw","Draw","DRAW","away","Away","AWAY"]

# ---------------------------------------------------------------------------

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_csv(path)

def _find_join_key(dfA: pd.DataFrame, dfB: pd.DataFrame) -> str:
    """
    Détecte automatiquement la clé commune (match/game id).
    Heuristique :
    - colonnes communes
    - priorité aux noms contenant 'game'/'match' ou 'id'
    - unicité raisonnable
    """
    commons = [c for c in dfA.columns if c in dfB.columns]
    # élimine les features classiques
    bad = set(k for k in commons if k.lower().startswith("team_"))
    commons = [c for c in commons if c not in bad]

    # propose dans l’ordre de probabilité
    priority = []
    for c in commons:
        cl = c.lower()
        score = 0
        if "game" in cl or "match" in cl: score += 3
        if "id" in cl: score += 2
        if dfA[c].nunique() > 10: score += 1
        priority.append((score, c))
    if not priority:
        # dernier recours : première colonne commune
        if not commons:
            raise ValueError("Aucune colonne commune pour la jointure.")
        return commons[0]
    priority.sort(reverse=True)
    return priority[0][1]

def _suffix_features(df: pd.DataFrame, suffix: str, exclude: List[str]) -> pd.DataFrame:
    ren = {c: f"{c}{suffix}" for c in df.columns if c not in exclude}
    return df.rename(columns=ren)

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    # supprime les colonnes entièrement vides / constantes
    nunique = df.nunique(dropna=False)
    drop = nunique[nunique <= 1].index.tolist()
    if drop:
        df = df.drop(columns=drop)
    return df

def _load_X(
    home_path: str, away_path: str, split_name: str = "train"
) -> Tuple[pd.DataFrame, str]:
    home = _read_csv(home_path)
    away = _read_csv(away_path)

    key = _find_join_key(home, away)

    home_s = _suffix_features(home, "_home", exclude=[key])
    away_s = _suffix_features(away, "_away", exclude=[key])

    X = home_s.merge(away_s, on=key, how="inner")
    X = _clean_cols(X)

    # standardise le nom de l'id pour la suite
    if key != "id":
        X = X.rename(columns={key: "id"})
    return X, "id"

def _load_y(y_path: str, id_col: str = "id") -> pd.DataFrame:
    y = _read_csv(y_path)

    # normalisation des noms de colonnes
    cols = {c.lower(): c for c in y.columns}
    # alias de l'id
    y_id = None
    for candidate in ["id", "row_id", "match_id", "game_id"]:
        if candidate in cols:
            y_id = cols[candidate]
            break
    if y_id is None:
        # si l'id n'est pas là, on suppose que la première colonne est l'id
        y_id = y.columns[0]

    # cible: trouver home/draw/away (insensible à la casse)
    ycols = {}
    for k in y.columns:
        kl = k.lower()
        if "home" in kl and "prob" not in kl:
            ycols["home"] = k
        elif "draw" in kl:
            ycols["draw"] = k
        elif "away" in kl:
            ycols["away"] = k
    if set(ycols.keys()) != {"home","draw","away"}:
        raise ValueError(
            f"Impossible de détecter les colonnes cible home/draw/away dans {y_path}. Colonnes trouvées: {y.columns.tolist()}"
        )

    y = y.rename(columns={y_id: "id", ycols["home"]: "home", ycols["draw"]: "draw", ycols["away"]: "away"})
    # on calcule une classe 0/1/2 pour l'entraînement multi-classes
    y["target"] = np.argmax(y[["home","draw","away"]].values, axis=1)
    return y[["id","home","draw","away","target"]]

# ---------------------- API PUBLIQUE ----------------------------------------

def load_train(paths: Dict[str,str] | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    p = {**DEFAULT_PATHS, **(paths or {})}
    X, id_col = _load_X(p["train_home"], p["train_away"], "train")
    y = _load_y(p["y_train"], id_col)
    df = X.merge(y, on="id", how="inner")
    y_target = df["target"].astype(int)
    X = df.drop(columns=["target","home","draw","away"])
    return X, y_target

def load_test(paths: Dict[str,str] | None = None) -> pd.DataFrame:
    p = {**DEFAULT_PATHS, **(paths or {})}
    X, id_col = _load_X(p["test_home"], p["test_away"], "test")
    return X  # contient la colonne 'id'

def get_feature_target_names() -> List[str]:
    # utilitaire si besoin
    return ["home","draw","away"]
