import os, json, argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

from src.build_dataset import build_Xy
from src.utils import set_seeds


def load_train_processed():
    set_seeds(42)

    X_raw = pd.read_csv("data/processed/train_merged.csv")
    y_raw = pd.read_csv("data/processed/y_train_aligned.csv")

    bad = [c for c in X_raw.columns if c in ("HOME_WINS","DRAW","AWAY_WINS")]
    assert not bad, f"Leakage: {bad} found in X_raw!"

    X, y = build_Xy(X_raw, y_raw)
    
    X = X.select_dtypes(include=["number"]).astype("float32").fillna(0.0)

    return X, y


def main(config_path: str):
    Xnum, y = load_train_processed()

    Xtr, Xva, ytr, yva = train_test_split(
        Xnum, y, test_size=0.2, random_state=42, stratify=y
    )

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, random_state=42
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xva)
    acc = accuracy_score(yva, pred)

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    joblib.dump(model, "outputs/models/model.joblib")

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

    with open("outputs/logs/features.txt", "w") as f:
        for c in Xnum.columns:
            f.write(c + "\n")

    #val_acc : accuracy on validation set
    #n_features : number of features used for training
    #train : number of training samples
    #valid : number of validation samples
    print(
        f"[train] ok | val_acc={acc:.4f} | n_features={Xnum.shape[1]} | "
        f"train={Xtr.shape[0]} valid={Xva.shape[0]}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
