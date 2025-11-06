# src/train.py (extrait corrigé)
import os, json, argparse
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import VarianceThreshold
import numpy as np, joblib


from src.utils import set_seeds
from src.build_dataset import build_Xy
from src.print_result import print_report

def load_train_processed():
    set_seeds(42)
    X_raw = pd.read_csv("data/processed/train_merged.csv")
    y_raw = pd.read_csv("data/processed/y_train_aligned.csv")

    # Anti-fuite
    bad = [c for c in X_raw.columns if c in ("HOME_WINS","DRAW","AWAY_WINS")]
    assert not bad, f"Leakage: {bad} found in X_raw!"

    X, y = build_Xy(X_raw, y_raw)
    X = X.select_dtypes(include=["number"]).astype("float32").fillna(0.0)
    return X, y

def main(config_path: str):
    X, y = load_train_processed()

    # 70 / 20 / 10 (train / val / hold-out)
    X_tmp, X_ho, y_tmp, y_ho = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
    X_tr,  X_va, y_tr, y_va   = train_test_split(X_tmp, y_tmp, test_size=0.2222, random_state=42, stratify=y_tmp)

    corr = X_tr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if (upper[c] > 0.995).any()]
    X_tr = X_tr.drop(columns=drop_cols); X_va = X_va.drop(columns=drop_cols); X_ho = X_ho.drop(columns=drop_cols)

    # === Sélecteur de features ===
    vt = VarianceThreshold(threshold=1e-4)
    vt.fit(X_tr)
    X_tr_sel = vt.transform(X_tr)
    X_va_sel = vt.transform(X_va)
    X_ho_sel = vt.transform(X_ho)

    # Noms des features conservées (utile pour predict)
    support = vt.get_support()
    selected_features = np.array(X_tr.columns)[support].tolist()

    # === Modèle ===
    hgb_params = dict(
        max_iter=800,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=200,
        l2_regularization=5.0,
        early_stopping=True,
        random_state=42,
    )

    model = HistGradientBoostingClassifier(**hgb_params)
    w = compute_sample_weight("balanced", y_tr)
    model.fit(X_tr_sel, y_tr, sample_weight=w)

    # === Scores et diagnostics ===
    train_acc = accuracy_score(y_tr, model.predict(X_tr_sel))
    val_acc   = accuracy_score(y_va, model.predict(X_va_sel))
    hold_pred = model.predict(X_ho_sel)
    hold_acc  = accuracy_score(y_ho, hold_pred)

    cm = confusion_matrix(y_ho, hold_pred)
    clf_report = classification_report(y_ho, hold_pred, digits=3)

    imp = permutation_importance(model, X_va_sel, y_va, n_repeats=5, random_state=42)
    top = np.argsort(-imp.importances_mean)[:30]
    top_features = [selected_features[i] for i in top]

    print_report(train_acc, val_acc, hold_acc, cm, clf_report, top_features, X, X_tr_sel, X_va_sel, X_ho_sel)

    # === Sauvegardes ===
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    artifact = {
        "model": model,
        "selector": vt,
        "drop_cols": drop_cols,          
        "features": selected_features,
    }
    joblib.dump(artifact, "outputs/models/model.joblib")


    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "hold_accuracy": float(hold_acc),
        "n_train": int(X_tr_sel.shape[0]),
        "n_valid": int(X_va_sel.shape[0]),
        "n_holdout": int(X_ho_sel.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "n_features_selected": int(X_tr_sel.shape[1]),
        "train_path": "data/processed/train_merged.csv",
        "model": "HistGradientBoostingClassifier",
        "params": hgb_params,
    }
    with open("outputs/logs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("outputs/logs/features.txt", "w") as f:
        for c in selected_features:
            f.write(c + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
