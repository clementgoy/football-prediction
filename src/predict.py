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
    # ajoute manquantes
    missing = [c for c in feature_list if c not in out.columns]
    for c in missing:
        out[c] = 0.0
    # garde uniquement la liste et dans le bon ordre
    out = out[feature_list]
    return out


def main(config_path: str):
    # chemins
    test_path = "data/x_test_clean.csv"
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
    Xtest = pd.read_csv(test_path)
    if "id" not in Xtest.columns:
        raise ValueError("'id' manquant dans data/x_test_clean.csv")
    ids = Xtest["id"].values

    # charge liste de features du train (pour aligner)
    feature_list = load_feature_list(feat_path)
    if not feature_list:
        # fallback: toutes les colonnes numériques sauf 'id'
        feature_list = (
            Xtest.drop(columns=["id"])
            .select_dtypes(include=["number"])
            .columns.tolist()
        )

    # ne garder que numérique puis aligner
    Xnum = Xtest.select_dtypes(include=["number"]).copy()
    if "id" in Xnum.columns:
        Xnum = Xnum.drop(columns=["id"])
    Xnum = Xnum.astype("float32").fillna(0.0)
    Xnum = align_columns(Xnum, feature_list)

    # charge modèle
    model = joblib.load(model_path)

    # proba dans l'ordre des classes du modèle
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Le modèle ne supporte pas predict_proba.")

    proba = model.predict_proba(Xnum)  # shape (N, n_classes)
    classes = list(getattr(model, "classes_", range(proba.shape[1])))

    # On veut l'ordre: [home, draw, away] = classes 0,1,2 (cf. train: argmax([home,draw,away]))
    def col_idx(c):
        try:
            return classes.index(c)
        except ValueError:
            # si les classes ne sont pas 0/1/2, on normalise au mieux
            return None

    idx_home = col_idx(0)
    idx_draw = col_idx(1)
    idx_away = col_idx(2)

    # sécurité si l'ordre n'est pas standard
    if None in (idx_home, idx_draw, idx_away):
        # réindexe via tri des classes si besoin
        order = np.argsort(classes)  # suppose classes triables et correspondent à 0,1,2
        proba = proba[:, order]
        idx_home, idx_draw, idx_away = 0, 1, 2

    sub = pd.DataFrame(
        {
            "id": ids,
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
