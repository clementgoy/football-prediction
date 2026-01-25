# Encore une fois (comme pour merge_data_streaming) ce fichier a été crée dans l'optique d'alléger l'entrainement afin qu'il puisse être supporté par mon pc perso et que je puisse avancer même pendant les vacances.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json, time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

X_PATH = PROCESSED / "train_merged.csv"
Y_ONEHOT_PATH = PROCESSED / "y_train_aligned.csv"
Y_SUPP_PATH   = PROCESSED / "y_train_supp_aligned.csv"

N_FEATURES_MAX = 600          
USE_HGB = False               
RF_TREES = 350                 
ET_TREES = 450                 
RANDOM_STATE = 42

def info(msg: str) -> None:
    print(f"\n[info] {msg}")

def ok(msg: str) -> None:
    print(f"[ok] {msg}")

def tic() -> float:
    return time.perf_counter()

def toc(t0: float, label: str = "done") -> None:
    dt = time.perf_counter() - t0
    ok(f"{label} in {dt:.1f}s")

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    if not X_PATH.exists():
        raise FileNotFoundError(f"Fichier X introuvable : {X_PATH}")
    if not Y_ONEHOT_PATH.exists():
        raise FileNotFoundError(f"Fichier Y introuvable : {Y_ONEHOT_PATH}")

    info("Chargement X (optimisé pour ma petite RAM)...")
    t0 = tic()
    X: Optional[pd.DataFrame] = None

    try:
        # On essaie d'utiliser PyArrow
        import pyarrow
        X = pd.read_csv(
            X_PATH,
            engine="pyarrow",
            memory_map=True
        )
        ok("Ouverture avec PyArrow réussie (c'est le top !)")
    except Exception as _:
        # Si PyArrow plante
        info("PyArrow marche pas -> On lit par petits bouts (chunks) pour pas saturer la mémoire")
        chunks = []
        for chunk in pd.read_csv(
            X_PATH,
            chunksize=20_000, # On lit 20 000 lignes à la fois
            low_memory=False
        ):
            # ASTUCE : Pour gagner de la place, on réduit la taille des nombres
            # float64 -> float32 
            # int64 -> int optimisé
            for c in chunk.columns:
                col = chunk[c]
                if pd.api.types.is_float_dtype(col):
                    chunk[c] = col.astype("float32")
                elif pd.api.types.is_integer_dtype(col):
                    chunk[c] = pd.to_numeric(col, downcast="integer")
                
            chunks.append(chunk)
        # On recolle tous les morceaux à la fin
        X = pd.concat(chunks, axis=0, ignore_index=True)

    toc(t0, f"X chargé : {X.shape[0]} lignes × {X.shape[1]} colonnes")

    info("Chargement y (one-hot) ...")
    t0 = tic()
    y1 = pd.read_csv(Y_ONEHOT_PATH, low_memory=False)
    toc(t0, f"y_onehot : {y1.shape[0]} lignes")

    y_supp = None
    if Y_SUPP_PATH.exists():
        y_supp = pd.read_csv(Y_SUPP_PATH, low_memory=False)
        ok(f"y_supp chargé (avec la différence de buts)")
    else:
        info("Pas de différence de buts dispo, on fera sans.")

    return X, y1, y_supp

def prepare_features_labels(
    X: pd.DataFrame,
    y_onehot: pd.DataFrame,
    n_features_max: int = N_FEATURES_MAX,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Index, List[str]]:
    need = ["ID", "HOME_WINS", "DRAW", "AWAY_WINS"]
    if not set(need).issubset(y_onehot.columns):
        raise ValueError("Il manque des colonnes dans y_onehot (ID, HOME_WINS...)")

    merged = X.merge(y_onehot[need], on="ID", how="inner")
    ok(f"Fusion X <-> y OK : {merged.shape[0]} lignes")

    y_cls = merged[["HOME_WINS", "DRAW", "AWAY_WINS"]].values.argmax(axis=1)

    # features numériques uniquement
    feat_cols = [c for c in merged.columns if c not in ("ID", "HOME_WINS", "DRAW", "AWAY_WINS")]
    num_cols = merged[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    dropped = len(feat_cols) - len(num_cols)
    if dropped > 0:
        info(f"On a viré {dropped} colonnes pas numériques.")

    X_num = merged[num_cols].copy()

    # Reductions de type encore une fois pour la RAM (float64 -> float32)
    for c in X_num.columns:
        if np.issubdtype(X_num[c].dtype, np.floating):
            X_num[c] = X_num[c].astype(np.float32, copy=False)
        elif np.issubdtype(X_num[c].dtype, np.integer):
            X_num[c] = pd.to_numeric(X_num[c], downcast="integer")

    # Remplissage des vides (NaN) par 0
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    X_imp = pd.DataFrame(imputer.fit_transform(X_num), columns=num_cols, index=X_num.index)
    X_imp = X_imp.astype(np.float32)  

    # Sélection des features pour alléger le modèle (VarianceThreshold manuel)
    if n_features_max is not None and X_imp.shape[1] > n_features_max:
        info(f"Y'a trop de colonnes ({X_imp.shape[1]}), on garde que les {n_features_max} plus 'tordues' (variance élevée).")
        # on calcule la variance colonne par colonne
        vars_ = X_imp.var(axis=0).to_numpy()
        # On trie pour garder celles qui bougent le plus
        top_idx = np.argsort(vars_)[::-1][:n_features_max]
        keep_cols = [X_imp.columns[i] for i in top_idx]
        X_imp = X_imp[keep_cols]
    else:
        keep_cols = list(X_imp.columns)

    ok(f"Features prêtes pour l'entraînement : {X_imp.shape[0]} lignes × {X_imp.shape[1]} colonnes")
    return X_imp, y_cls, merged["ID"], keep_cols

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

def make_candidates(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    rf = RandomForestClassifier(
        n_estimators=RF_TREES,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )

    et = ExtraTreesClassifier(
        n_estimators=ET_TREES,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )

    models = {"rf": rf, "et": et}

    if USE_HGB:
        hgb = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.06,
            max_depth=None,
            max_leaf_nodes=31,    
            min_samples_leaf=80,  
            l2_regularization=0.0,
            max_bins=64,           
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        )
        models["hgb"] = hgb

    return models

# entrainement et selection
def train_and_select(
    X: pd.DataFrame,
    y: np.ndarray,
    ids: pd.Index,
    y_supp: Optional[pd.DataFrame],
    weight_scheme: str = "linear025",
    random_state: int = RANDOM_STATE,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:

    # Découpage train/val simple
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    (tr_idx, va_idx), = sss.split(X, y)

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    w_all = build_sample_weights(ids, y_supp, scheme=weight_scheme)
    w_tr = w_all[tr_idx]

    models = make_candidates(random_state)
    results = []

    for key, clf in models.items():
        info(f"Test du modèle '{key}' avec poids '{weight_scheme}' ...")
        t0 = tic()
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        toc(t0, f"Entraînement {key} réussi")

        pred = clf.predict(X_va)
        acc = accuracy_score(y_va, pred)
        ok(f"Résultat {key}: accuracy = {acc:.4f}")

        rep = classification_report(
            y_va, pred,
            target_names=["HOME_WINS","DRAW","AWAY_WINS"],
            digits=4
        )
        cm = confusion_matrix(y_va, pred).tolist()

        # On sauvegarde le modèle
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

        # Liste de features réellement utilisées 
        feats_to_save = list(X.columns) if feature_names is None else feature_names
        (MODELS / f"{tag}_features.json").write_text(
            json.dumps({"feature_names": feats_to_save}, indent=2), encoding="utf-8"
        )

        results.append({"key": key, "acc": acc, "model_path": str(model_path), "meta": meta})

    # On garde le meilleur
    best = max(results, key=lambda r: r["acc"])
    ok(f"Le meilleur est : {best['key']} (score={best['acc']:.4f}) -> {best['model_path']}")

    (MODELS / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best

def main() -> None:
    X_raw, y_onehot, y_supp = load_data()
    X, y, ids, kept_features = prepare_features_labels(X_raw, y_onehot, n_features_max=N_FEATURES_MAX)

    for scheme in ["none", "linear025"]:
        info(f"--- Test du schéma : {scheme} ---")
        _ = train_and_select(X, y, ids, y_supp,
                             weight_scheme=scheme,
                             random_state=RANDOM_STATE,
                             feature_names=kept_features)

    ok("Tout est fini. Les modèles sont dans le dossier 'models/'.")

if __name__ == "__main__":
    main()
