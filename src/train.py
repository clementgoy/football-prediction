# src/train.py
import os, json, argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib


def load_X_y(
    x_path: str = "data/x_train_clean.csv",
    y_path: str = "data/y_train_aligned.csv",
):
    # X: doit contenir une colonne 'id' + features numériques
    X = pd.read_csv(x_path)
    if "id" not in X.columns:
        raise ValueError(f"'id' manquant dans {x_path}")

    # y: colonnes ['id', 'home','draw','away']
    y = pd.read_csv(y_path)
    expected = {"id", "home", "draw", "away"}
    if not expected.issubset(set(y.columns)):
        raise ValueError(
            f"Colonnes attendues dans {y_path}: {sorted(expected)} ; trouvées: {y.columns.tolist()}"
        )

    # garder l’intersection des ids (sécurité)
    ids = pd.Series(sorted(set(X["id"]).intersection(set(y["id"]))), name="id")
    X = ids.to_frame().merge(X, on="id", how="left")
    y = ids.to_frame().merge(y, on="id", how="left")

    # cible 0/1/2 depuis les probas
    y_target = np.argmax(y[["home", "draw", "away"]].values, axis=1).astype(int)

    return X, y_target


def main(config_path: str):
    # Chargement
    X, y = load_X_y()

    # Sélection numérique + drop id
    feature_cols = [c for c in X.columns if c != "id"]
    Xnum = X[feature_cols].select_dtypes(include=["number"]).copy()

    # Harmonisation des types + NaN
    Xnum = Xnum.astype("float32")
    Xnum = Xnum.fillna(0.0)

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
        "x_path": "data/x_train_clean.csv",
        "y_path": "data/y_train_aligned.csv",
        "model": "HistGradientBoostingClassifier",
        "params": {"max_iter": 300, "learning_rate": 0.05, "random_state": 42},
    }
    with open("outputs/logs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # petit aperçu des features pour debug reproductible
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
