# src/predict.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd


def load_feature_list(path: str) -> list[str]:
    """Lit la liste des features (une par ligne) écrite pendant le train."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        feats = [ln.strip() for ln in f if ln.strip()]
    return feats


def align_columns(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """
    Aligne df sur feature_list :
      - ajoute les features manquantes (remplies à 0)
      - supprime les colonnes en trop
      - respecte l'ordre des colonnes du train
    """
    out = df.copy()
    missing = [c for c in feature_list if c not in out.columns]
    for c in missing:
        out[c] = 0.0
    out = out[feature_list]
    return out


def main(config_path: str):
    # chemins
    test_path = "data/processed/test_merged.csv"
    model_path = "outputs/models/model.joblib"
    feat_path = "outputs/logs/features.txt"
    out_dir = "outputs/submissions"
    out_csv = os.path.join(out_dir, "submission.csv")

    # vérifs
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable: {model_path}. Entraîne d'abord avec `make train`."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Fichier test introuvable: {test_path}")

    # charge test
    df = pd.read_csv(test_path)
    if "ID" not in df.columns:
        raise ValueError("'ID' manquant dans data/processed/test_merged.csv")

    ids = df["ID"].values

    # features numériques uniquement, sans ID
    X = df.drop(columns=["ID"]).select_dtypes(include=["number"]).copy()
    X = X.astype("float32").fillna(0.0)

    # charge liste de features du train (pour aligner)
    feature_list = load_feature_list(feat_path)
    if not feature_list:
        # fallback (moins sûr) : toutes les colonnes numériques actuelles
        feature_list = X.columns.tolist()

    X = align_columns(X, feature_list)

    # charge modèle
    model = joblib.load(model_path)

    # proba dans l'ordre des classes du modèle (attendues 0,1,2 = home,draw,away)
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Le modèle ne supporte pas predict_proba.")
    proba = model.predict_proba(X)  # (N, n_classes)

    # classes_: on veut [0,1,2] -> [home,draw,away]
    classes = list(getattr(model, "classes_", range(proba.shape[1])))

    def idx_of(c):
        try:
            return classes.index(c)
        except ValueError:
            return None

    idx_home = idx_of(0)
    idx_draw = idx_of(1)
    idx_away = idx_of(2)

    if None in (idx_home, idx_draw, idx_away):
        # fallback: tri des classes si elles sont numériques mais désordonnées
        try:
            order = np.argsort(classes)
            proba = proba[:, order]
            idx_home, idx_draw, idx_away = 0, 1, 2
        except Exception as e:
            raise RuntimeError(f"Impossible d'aligner les classes du modèle: {classes}") from e

    sub = pd.DataFrame(
        {
            "id": ids,  # la plateforme attend souvent 'id'
            "home": proba[:, idx_home],
            "draw": proba[:, idx_draw],
            "away": proba[:, idx_away],
        }
    )

    os.makedirs(out_dir, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"[predict] ok -> {out_csv} | shape={sub.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
