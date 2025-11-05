# src/train.py
import os, json, argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib


def load_train_processed(path: str = "data/processed/train_merged.csv"):
    df = pd.read_csv(path)
    # Vérifs de base
    need_y = {"y_home_win", "y_draw", "y_away_win"}
    if "ID" not in df.columns:
        raise ValueError(f"'ID' manquant dans {path}")
    if not need_y.issubset(set(df.columns)):
        raise ValueError(
            f"Colonnes cibles manquantes dans {path}. Attendu: {sorted(need_y)}"
        )

    # Cible 0/1/2 depuis one-hot
    y = np.argmax(df[["y_home_win", "y_draw", "y_away_win"]].values, axis=1).astype(int)

    # Features = toutes colonnes sauf targets
    X = df.drop(columns=["y_home_win", "y_draw", "y_away_win"]).copy()

    # On garde ID séparément (pas une feature)
    if "ID" in X.columns:
        X = X.set_index("ID", drop=True)  # index = ID, n'entre pas dans le modèle

    # Numérique uniquement, types & NaN
    Xnum = X.select_dtypes(include=["number"]).astype("float32").fillna(0.0)

    return Xnum, y


def main(config_path: str):
    # Chargement
    Xnum, y = load_train_processed()

    # Split
    Xtr, Xva, ytr, yva = train_test_split(
        Xnum, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modèle
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, random_state=42
    )
    model.fit(Xtr, ytr)

    # Métriques
    pred = model.predict(Xva)
    acc = accuracy_score(yva, pred)

    # Sauvegardes
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    joblib.dump(model, "outputs/models/model.joblib")

    # log simple + infos utiles
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "val_accuracy": float(acc),
        "n_train": int(Xtr.shape[0]),
        "n_valid": int(Xva.shape[0]),
        "n_features": int(Xnum.shape[1]),
        "train_path": "data/processed/train_merged.csv",
        "model": "HistGradientBoostingClassifier",
        "params": {"max_iter": 300, "learning_rate": 0.05, "random_state": 42},
    }
    with open("outputs/logs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # sauvegarde la liste des features utilisées (pour aligner au moment du predict)
    with open("outputs/logs/features.txt", "w") as f:
        for c in Xnum.columns:
            f.write(c + "\n")

    print(
        f"[train] ok | val_acc={acc:.4f} | n_features={Xnum.shape[1]} | "
        f"train={Xtr.shape[0]} valid={Xva.shape[0]}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
