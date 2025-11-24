# src/train_tabular_baseline.py
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

# ---------- chemins (robustes au placement dans src/) ----------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

X_PATH = PROCESSED / "train_merged.csv"
Y_ONEHOT_PATH = PROCESSED / "y_train_aligned.csv"          # colonnes: ID, HOME_WINS, DRAW, AWAY_WINS
Y_SUPP_PATH   = PROCESSED / "y_train_supp_aligned.csv"     # colonnes: ID, GOAL_DIFF_HOME_AWAY (optionnel)

def info(msg: str) -> None:
    print(f"\n[info] {msg}")

def ok(msg: str) -> None:
    print(f"[ok] {msg}")

# ---------- chargement ----------
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    if not X_PATH.exists():
        raise FileNotFoundError(f"Introuvable: {X_PATH}")
    if not Y_ONEHOT_PATH.exists():
        raise FileNotFoundError(f"Introuvable: {Y_ONEHOT_PATH}")

    info("Chargement X …")
    X = pd.read_csv(X_PATH, low_memory=False)
    ok(f"X: {X.shape[0]} lignes × {X.shape[1]} colonnes")

    info("Chargement y one-hot …")
    y1 = pd.read_csv(Y_ONEHOT_PATH, low_memory=False)
    ok(f"y_onehot: {y1.shape[0]} lignes × {y1.shape[1]} colonnes")

    y_supp = None
    if Y_SUPP_PATH.exists():
        y_supp = pd.read_csv(Y_SUPP_PATH, low_memory=False)
        ok(f"y_supp: {y_supp.shape[0]} lignes × {y_supp.shape[1]} colonnes")
    else:
        info("Pas de y_supp (OK) — pondérations par écart de buts désactivées.")

    return X, y1, y_supp

# ---------- préparation ----------
def prepare_features_labels(
    X: pd.DataFrame,
    y_onehot: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Index]:
    need = ["ID", "HOME_WINS", "DRAW", "AWAY_WINS"]
    if not set(need).issubset(y_onehot.columns):
        raise ValueError("y_onehot doit contenir ID, HOME_WINS, DRAW, AWAY_WINS")

    merged = X.merge(y_onehot[need], on="ID", how="inner")
    ok(f"Alignement X↔y: {merged.shape[0]} lignes")

    # classes (0,1,2)
    y_cls = merged[["HOME_WINS", "DRAW", "AWAY_WINS"]].values.argmax(axis=1)

    # features = colonnes numériques uniquement (sécurise le modèle)
    feature_cols_all = [c for c in merged.columns if c not in ("ID", "HOME_WINS", "DRAW", "AWAY_WINS")]
    num_cols = merged[feature_cols_all].select_dtypes(include=[np.number]).columns
    dropped = len(feature_cols_all) - len(num_cols)
    if dropped > 0:
        info(f"Colonnes non numériques écartées: {dropped}")

    X_num = merged[num_cols].copy()

    # imputation simple à 0.0 (les forêts y sont peu sensibles)
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    X_imp = pd.DataFrame(imputer.fit_transform(X_num), columns=num_cols, index=X_num.index)

    return X_imp, y_cls, merged["ID"]

def build_sample_weights(
    ids: pd.Index,
    y_supp: Optional[pd.DataFrame],
    scheme: str = "linear025"
) -> np.ndarray:
    """
    schémas:
      - 'none'       : poids = 1
      - 'linear025'  : 1 + 0.25 * |écart|
      - 'exp015'     : exp(0.15 * |écart|), plafonné
    """
    w = np.ones(len(ids), dtype=np.float32)
    if y_supp is None or scheme == "none":
        return w

    if "GOAL_DIFF_HOME_AWAY" not in y_supp.columns:
        info("y_supp sans GOAL_DIFF_HOME_AWAY → pondérations ignorées.")
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
    w = np.clip(w, 0.5, 20.0).astype(np.float32)
    return w

# ---------- modèles ----------
def make_candidates(random_state: int = 42) -> Dict[str, Any]:
    """
    Trois candidats simples, mais efficaces sur tabulaire:
      - rf  : RandomForest (réglages robustes)
      - et  : ExtraTrees (plus de randomisation, souvent très bon)
      - hgb : HistGradientBoosting (boosting d’arbres, early stopping)
    """
    rf = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",  # équilibre classes rares (ex: DRAW)
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

    # NB: HGB gère bien les NA mais on a déjà imputé 0.0 (simple).
    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_depth=None,            # la profondeur effective est gérée par max_leaf_nodes
        max_leaf_nodes=63,         # feuilles pas trop nombreuses (biais/variance)
        min_samples_leaf=50,       # régularisation simple
        l2_regularization=0.0,
        max_bins=255,              # bins fins (tabulaire souvent ok)
        early_stopping=True,
        validation_fraction=0.1,   # HGB garde une partie du train pour le stop (indépendant de notre split)
        random_state=random_state,
    )

    return {"rf": rf, "et": et, "hgb": hgb}

# ---------- entraînement & sélection ----------
def train_and_select(
    X: pd.DataFrame,
    y: np.ndarray,
    ids: pd.Index,
    y_supp: Optional[pd.DataFrame],
    weight_scheme: str = "linear025",
    random_state: int = 42,
) -> Dict[str, Any]:
    # split simple & stratifié (pas de K-fold comme demandé)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    (tr_idx, va_idx), = sss.split(X, y)

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # poids = pondération écart de buts (optionnelle)
    w_all = build_sample_weights(ids, y_supp, scheme=weight_scheme)
    w_tr = w_all[tr_idx]

    models = make_candidates(random_state)
    results = []

    for key, clf in models.items():
        info(f"Entraînement modèle '{key}' (poids: {weight_scheme}) …")
        # RF / ET: sample_weight OK; HGB: aussi OK
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        pred = clf.predict(X_va)
        acc = accuracy_score(y_va, pred)
        ok(f"{key}: val_accuracy = {acc:.4f}")

        rep = classification_report(
            y_va, pred,
            target_names=["HOME_WINS","DRAW","AWAY_WINS"],
            digits=4
        )
        cm = confusion_matrix(y_va, pred).tolist()

        # Sauvegardes
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

        results.append({"key": key, "acc": acc, "model_path": str(model_path), "meta": meta})

    best = max(results, key=lambda r: r["acc"])
    ok(f"Meilleur: {best['key']} (acc={best['acc']:.4f}) → {best['model_path']}")

    # Pointeur vers le best
    (MODELS / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best

def main() -> None:
    X_raw, y_onehot, y_supp = load_data()
    X, y, ids = prepare_features_labels(X_raw, y_onehot)

    # essaie ces trois schémas sans complexifier
    for scheme in ["none", "linear025", "exp015"]:
        info(f"--- Schéma de poids: {scheme} ---")
        best = train_and_select(X, y, ids, y_supp, weight_scheme=scheme, random_state=42)

    ok("Terminé. Regarde le dossier 'models/' (json + joblib).")

if __name__ == "__main__":
    main()
