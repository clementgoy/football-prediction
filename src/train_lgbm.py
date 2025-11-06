#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import lightgbm as lgb  # pour les callbacks
import joblib

from src.print_result import print_report
from src.build_dataset import build_Xy

def load_train_processed(train_csv, y_csv):
    """
    - Charge X_raw, y_raw
    - Vérifie l'absence de fuite (colonnes cibles dans X)
    - Aligne par ID via build_Xy (ton helper)
    - Garde NUMÉRIQUES uniquement, cast float32, NaN -> 0.0
    - Renvoie (X, y, feat_cols, class_names)
    """
    X_raw = pd.read_csv(train_csv)
    y_raw = pd.read_csv(y_csv)

    if all(c in y_raw.columns for c in ["HOME_WINS", "DRAW", "AWAY_WINS"]):
        class_names = ["HOME_WINS", "DRAW", "AWAY_WINS"]
    else:
        raise ValueError(f"Colonnes de y non reconnues: {y_raw.columns.tolist()}")

    bad = [c for c in X_raw.columns if c in class_names]
    assert not bad, f"Leakage: {bad} found in X_raw!"

    X_aligned, y = build_Xy(X_raw, y_raw)  
    drop_ids = [c for c in ["ID","match_id","MatchID"] if c in X_aligned.columns]
    if drop_ids:
        X_aligned = X_aligned.drop(columns=drop_ids)

    X_aligned = X_aligned.select_dtypes(include=["number"]).astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat_cols = X_aligned.columns.tolist()

    if isinstance(y, pd.DataFrame):
        if all(c in y.columns for c in class_names):
            y = y[class_names].values.argmax(axis=1)
        else:
            y = y.values.squeeze()
    elif isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y).astype(int).ravel()

    return X_aligned, y, feat_cols, class_names

def main():
    parser = argparse.ArgumentParser(description="LightGBM avec validation + hold-out + print_report.")
    parser.add_argument("--train-csv", default="data/processed/train_merged.csv")
    parser.add_argument("--y-csv",     default="data/processed/y_train_aligned.csv")
    parser.add_argument("--model-out", default="outputs/models/lgbm.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size-valid", type=float, default=0.20)
    parser.add_argument("--test-size-hold",  type=float, default=0.20)
    parser.add_argument("--early-stopping", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    X, y, feat_cols, class_names = load_train_processed(args.train_csv, args.y_csv)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=args.test_size_valid, random_state=args.seed, stratify=y
    )
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X_tr, y_tr, test_size=args.test_size_hold, random_state=args.seed, stratify=y_tr
    )

    classes = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    clf = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        learning_rate=0.05,        
        n_estimators=5000,          
        num_leaves=31,             
        max_depth=10,          
        min_data_in_leaf=200,       
        min_gain_to_split=0.05,
        subsample=0.7,
        subsample_freq=1,           
        colsample_bytree=0.6, 
        reg_alpha=0.5,            
        reg_lambda=5.0, 
        max_bin=255,
        random_state=args.seed,
        class_weight=class_weight,
        n_jobs=-1
    ).set_params(force_col_wise=True)

    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
        lgb.log_evaluation(period=0)
    ]

    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=callbacks
    )

    p_va = clf.predict_proba(X_va)  
    p_ho = clf.predict_proba(X_ho) 

    def tune_draw_bias(p, y_true, alphas=(0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3)):
        best_a, best_acc = 1.0, -1.0
        for a in alphas:
            p_adj = p.copy()
            p_adj[:, 1] *= a          
            p_adj = p_adj / p_adj.sum(axis=1, keepdims=True)
            acc = (p_adj.argmax(axis=1) == y_true).mean()
            if acc > best_acc:
                best_acc, best_a = acc, a
        return best_a, best_acc

    alpha, acc_val_adj = tune_draw_bias(p_va, y_va)
    print(f"Facteur optimal pour DRAW (validation) : {alpha:.2f} → acc={acc_val_adj:.4f}")

    p_ho_adj = p_ho.copy()
    p_ho_adj[:, 1] *= alpha
    p_ho_adj = p_ho_adj / p_ho_adj.sum(axis=1, keepdims=True)
    yhat_ho = p_ho_adj.argmax(axis=1)

    yhat_tr = clf.predict(X_tr)
    yhat_va = clf.predict(X_va)
    yhat_ho = clf.predict(X_ho)

    train_acc = accuracy_score(y_tr, yhat_tr)
    val_acc   = accuracy_score(y_va, yhat_va)
    hold_acc  = accuracy_score(y_ho, yhat_ho)

    cm      = confusion_matrix(y_ho, yhat_ho, labels=[0,1,2])
    clf_rep = classification_report(y_ho, yhat_ho, target_names=class_names)

    importances = getattr(clf, "feature_importances_", None)
    if importances is not None:
        order = np.argsort(importances)[::-1]
        top_features = [f"{feat_cols[i]}  (gain={int(importances[i])})" for i in order[:200]]
    else:
        top_features = feat_cols

    print_report(
        train_acc=train_acc,
        val_acc=val_acc,
        hold_acc=hold_acc,
        cm=cm,
        clf_report=clf_rep,
        top_features=top_features,
        X=X,              
        X_tr_sel=X_tr,
        X_va_sel=X_va,
        X_ho_sel=X_ho
    )

    joblib.dump(clf, args.model_out)
    print(f"\n Modèle sauvegardé dans {args.model_out}")

if __name__ == "__main__":
    main()
