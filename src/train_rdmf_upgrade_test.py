from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
import joblib

from print_result import print_report

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

X_PATH = PROCESSED / "train_merged.csv"
Y_ONEHOT_PATH = PROCESSED / "y_train_aligned.csv"        
Y_SUPP_PATH   = PROCESSED / "y_train_supp_aligned.csv" 

def info(msg: str) -> None:
    print(f"\n[info] {msg}")

def ok(msg: str) -> None:
    print(f"[ok] {msg}")

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    if not X_PATH.exists():
        raise FileNotFoundError(f"Je trouve pas X : {X_PATH}")
    if not Y_ONEHOT_PATH.exists():
        raise FileNotFoundError(f"Je trouve pas Y : {Y_ONEHOT_PATH}")

    info("On charge les données X ...")
    X = pd.read_csv(X_PATH, low_memory=False)
    ok(f"X : {X.shape[0]} lignes × {X.shape[1]} colonnes")

    info("On charge les données Y (one-hot) ...")
    y1 = pd.read_csv(Y_ONEHOT_PATH, low_memory=False)
    ok(f"y_onehot : {y1.shape[0]} lignes × {y1.shape[1]} colonnes")

    y_supp = None
    if Y_SUPP_PATH.exists():
        y_supp = pd.read_csv(Y_SUPP_PATH, low_memory=False)
        ok(f"y_supp trouvé (cool, on a la différence de buts) : {y_supp.shape[0]} lignes")
    else:
        info("Pas de y_supp, tant pis, on fera sans pondération par buts.")

    return X, y1, y_supp

def prepare_features_labels(
    X: pd.DataFrame,
    y_onehot: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Index]:
    need = ["ID", "HOME_WINS", "DRAW", "AWAY_WINS"]
    if not set(need).issubset(y_onehot.columns):
        raise ValueError("Il manque des colonnes dans y_onehot (ID, HOME_WINS...)")

    merged = X.merge(y_onehot[need], on="ID", how="inner")
    ok(f"Fusion OK : {merged.shape[0]} lignes")

    # classes (0,1,2)
    y_cls = merged[["HOME_WINS", "DRAW", "AWAY_WINS"]].values.argmax(axis=1)

    # features = colonnes numériques uniquement
    feature_cols_all = [c for c in merged.columns if c not in ("ID", "HOME_WINS", "DRAW", "AWAY_WINS")]
    num_cols = merged[feature_cols_all].select_dtypes(include=[np.number]).columns
    dropped = len(feature_cols_all) - len(num_cols)
    if dropped > 0:
        info(f"On a viré {dropped} colonnes qui n'étaient pas des chiffres.")

    X_num = merged[num_cols].copy()

    # On remplace les trous par des 0 
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    X_imp = pd.DataFrame(imputer.fit_transform(X_num), columns=num_cols, index=X_num.index)

    return X_imp, y_cls, merged["ID"]

def build_sample_weights(
    ids: pd.Index,
    y_supp: Optional[pd.DataFrame],
    scheme: str = "linear025"
) -> np.ndarray:
    
    # linear025 = 1 + 0.25 * |écart|
    # exp015 = exp(0.15 * |écart|), plafonné

    w = np.ones(len(ids), dtype=np.float32)
    if y_supp is None or scheme == "none":
        return w

    if "GOAL_DIFF_HOME_AWAY" not in y_supp.columns:
        info("Pas de GOAL_DIFF dans y_supp, donc pas de poids.")
        return w

    tmp = pd.DataFrame({"ID": ids}).merge(
        y_supp[["ID", "GOAL_DIFF_HOME_AWAY"]],
        on="ID", how="left"
    )
    diff = tmp["GOAL_DIFF_HOME_AWAY"].fillna(0).abs().to_numpy(np.float32)

    if scheme == "linear025":
        w = 1.0 + 0.25 * diff
    elif scheme == "exp015":
        w = np.exp(0.15 * diff, dtype=np.float64)
    
    # On limite les poids
    w = np.clip(w, 0.5, 20.0).astype(np.float32)
    return w

def make_candidates(random_state: int = 42) -> Dict[str, Any]:
    # On teste 3 modèles différents pour voir le meilleur :
    # rf = RandomForest (classique, valeur sûre)
    # et = ExtraTrees (plus rapide, plus aléatoire, parfois meilleur)
    # hgb = HistGradientBoosting

    rf = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",  # Pour aider à trouver les matchs nuls
    )

    et = ExtraTreesClassifier(
        n_estimators=900,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )

    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_depth=None,            
        max_leaf_nodes=63, 
        min_samples_leaf=50,  
        l2_regularization=0.0,
        max_bins=255,      
        early_stopping=True,
        validation_fraction=0.1, 
        random_state=random_state,
    )

    return {"rf": rf, "et": et, "hgb": hgb}

def train_and_select(
    X: pd.DataFrame,
    y: np.ndarray,
    ids: pd.Index,
    y_supp: Optional[pd.DataFrame],
    weight_scheme: str = "linear025",
    random_state: int = 42,
) -> Dict[str, Any]:
    # On coupe en train/val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    (tr_idx, va_idx), = sss.split(X, y)

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    w_all = build_sample_weights(ids, y_supp, scheme=weight_scheme)
    w_tr = w_all[tr_idx]

    models = make_candidates(random_state)
    results = []

    for key, clf in models.items():
        info(f"Test du modèle '{key}' (avec pondération : {weight_scheme}) ...")
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        pred_va = clf.predict(X_va)
        acc = accuracy_score(y_va, pred_va)
        ok(f"Résultat pour {key}: accuracy = {acc:.4f}")

        rep = classification_report(
            y_va, pred_va,
            target_names=["HOME_WINS","DRAW","AWAY_WINS"],
            digits=4
        )
        cm = confusion_matrix(y_va, pred_va).tolist()

        # On sauvegarde tout
        tag = f"{key}_{weight_scheme}"
        model_path = MODELS / f"{tag}.joblib"
        joblib.dump(clf, model_path)

        meta = {
            "model": key,
            "weight_scheme": weight_scheme,
            "val_accuracy": acc,
            "n_train": int(X_tr.shape[0]),
            "n_val": int(X_va.shape[0]),
            "n_features": int(X.shape[1]),
            "random_state": random_state,
            "confusion_matrix": cm,
            "classification_report": rep,
        }
        (MODELS / f"{tag}_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (MODELS / f"{tag}_features.json").write_text(json.dumps({"feature_names": list(X.columns)}, indent=2), encoding="utf-8")

        results.append({
            "key": key,
            "acc": acc,
            "model_path": str(model_path),
            "meta": meta,
            "clf": clf,
        })

    # On prend le meilleur des 3
    best = max(results, key=lambda r: r["acc"])
    best_clf = best["clf"]
    ok(f"Le vainqueur est : {best['key']} (score={best['acc']:.4f}) -> {best['model_path']}")

    # Juste pour info, on regarde le score sur le train
    y_tr_pred = best_clf.predict(X_tr)
    train_acc = accuracy_score(y_tr, y_tr_pred)

    y_va_pred = best_clf.predict(X_va)
    val_acc = accuracy_score(y_va, y_va_pred)

    hold_acc = val_acc
    cm_hold = confusion_matrix(y_va, y_va_pred)
    clf_rep_hold = classification_report(
        y_va, y_va_pred,
        target_names=["HOME_WINS","DRAW","AWAY_WINS"],
        digits=4
    )

    # Si le modèle nous donne l'importance des features, on regarde
    if hasattr(best_clf, "feature_importances_"):
        importances = np.asarray(best_clf.feature_importances_)
        feature_names = np.array(X.columns)
        order = np.argsort(importances)[::-1]
        top_features = feature_names[order].tolist()
    else:
        top_features = list(X.columns)

    # Affichage du rapport complet
    print_report(
        train_acc=train_acc,
        val_acc=val_acc,
        hold_acc=hold_acc,
        cm=cm_hold,
        clf_report=clf_rep_hold,
        top_features=top_features,
        X=X,
        X_tr_sel=X_tr,
        X_va_sel=X_va,
        X_ho_sel=X_va, 
    )

    # On prépare le JSON final
    best_for_json = dict(best)        
    best_for_json.pop("clf", None)    
    (MODELS / "best.json").write_text(
        json.dumps(best_for_json, indent=2),
        encoding="utf-8"
    )

    return best

def main() -> None:
    X_raw, y_onehot, y_supp = load_data()
    X, y, ids = prepare_features_labels(X_raw, y_onehot)

    for scheme in ["none", "linear025", "exp015"]:
        info(f"On teste avec le schéma de poids : {scheme} ---")
        best = train_and_select(X, y, ids, y_supp, weight_scheme=scheme, random_state=42)

    ok("Tout est fini. Va voir dans 'models/' !")

if __name__ == "__main__":
    main()

