from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib



BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_X_PATH = PROC / "train_merged.csv"
Y_ALIGNED_PATH = PROC / "y_train_aligned.csv"
Y_SUPP_PATH    = PROC / "y_train_supp_aligned.csv"


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    max_cardinality: int = 50  

    # grille
    param_grid: dict = None

    def __post_init__(self):
        if self.param_grid is None:
            self.param_grid = {
                "n_estimators": [200, 400],
                "max_depth": [None, 12, 20],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 3],
                "max_features": ["sqrt", "log2"]
            }


def info(msg: str) -> None:
    print(f"\n[info] {msg}")

def ok(msg: str) -> None:
    print(f"[ok] {msg}")



# Data loaders
def load_X() -> pd.DataFrame:
    if not TRAIN_X_PATH.exists():
        raise FileNotFoundError(f"Fichier X introuvable: {TRAIN_X_PATH}")
    X = pd.read_csv(TRAIN_X_PATH, low_memory=False)
    ok(f"X: {X.shape[0]} lignes × {X.shape[1]} colonnes")
    return X

def load_y_and_supp() -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if not Y_ALIGNED_PATH.exists():
        raise FileNotFoundError(f"Fichier Y introuvable: {Y_ALIGNED_PATH}")
    y = pd.read_csv(Y_ALIGNED_PATH, low_memory=False)
    ok(f"y_aligned: {y.shape[0]} lignes × {y.shape[1]} cols")

    if Y_SUPP_PATH.exists():
        y_s = pd.read_csv(Y_SUPP_PATH, low_memory=False)
        ok(f"y_supp_aligned: {y_s.shape[0]} lignes × {y_s.shape[1]} cols")
    else:
        y_s = None
        info("y_train_supp_aligned.csv manquant → pas de marge de buts (poids=1.0)")
    return y, y_s



# Helpers: cible + encodage
def one_hot_to_class(y_df: pd.DataFrame) -> np.ndarray:
    need = ["HOME_WINS","DRAW","AWAY_WINS"]
    if not set(need).issubset(y_df.columns):
        raise ValueError(f"y_df doit contenir {need}")
    y_mat = y_df[need].to_numpy()
    return y_mat.argmax(axis=1)

def _encode_categoricals(df: pd.DataFrame, max_cardinality: int = 50) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    keep, drop = [], []
    for c in cat_cols:
        nun = int(df[c].nunique(dropna=True))
        (keep if nun <= max_cardinality else drop).append(c)
    if drop:
        info(f"Dropping high-cardinality categoricals: {drop[:10]}{' ...' if len(drop)>10 else ''}")
    df_ = df.drop(columns=drop, errors="ignore")
    if keep:
        enc = pd.get_dummies(df_[keep], drop_first=True, dummy_na=False)
        rest = df_.drop(columns=keep, errors="ignore")
        out = pd.concat([rest, enc], axis=1)
    else:
        out = df_
    return out

def build_Xy_for_training(X: pd.DataFrame, y_like: pd.DataFrame, max_cardinality: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if "ID" not in X.columns or "ID" not in y_like.columns:
        raise ValueError("Les deux tables doivent contenir la colonne ID.")
    merged = y_like[["ID","HOME_WINS","DRAW","AWAY_WINS"]].merge(X, on="ID", how="left")

    feature_cols = [c for c in merged.columns if c not in ["ID","HOME_WINS","DRAW","AWAY_WINS"]]
    feats = merged[feature_cols]

    num = feats.select_dtypes(include=[np.number])
    non_num = feats.drop(columns=num.columns, errors="ignore")
    if not non_num.empty:
        non_num_enc = _encode_categoricals(non_num, max_cardinality=max_cardinality)
        feats_final = pd.concat([num, non_num_enc], axis=1)
    else:
        feats_final = num

    feature_names = feats_final.columns.tolist()
    X_feat = feats_final.fillna(0.0).to_numpy(dtype=np.float32)
    y_class = one_hot_to_class(merged)

    print(f"[ok] Features after encoding: {X_feat.shape[1]} columns "
          f"(num={num.shape[1]}, encoded={X_feat.shape[1]-num.shape[1]})")
    return X_feat, y_class, feature_names



# Poids: 3 scénarios (sans poids, lineaires, exp)
def compute_win_margin(y_df: pd.DataFrame, y_supp_df: Optional[pd.DataFrame]) -> np.ndarray:
    if y_supp_df is None:
        return np.zeros(len(y_df), dtype=float)

    need_y = {"ID","HOME_WINS","DRAW","AWAY_WINS"}
    need_s = {"ID","GOAL_DIFF_HOME_AWAY"}
    if not need_y.issubset(y_df.columns):
        missing = need_y - set(y_df.columns)
        raise ValueError(f"y_df missing columns: {missing}")
    if not need_s.issubset(y_supp_df.columns):
        missing = need_s - set(y_supp_df.columns)
        raise ValueError(f"y_supp_df missing columns: {missing}")

    merged = y_df.merge(y_supp_df, on="ID", how="left")
    gd = merged["GOAL_DIFF_HOME_AWAY"].fillna(0.0).to_numpy()

    home_margin = np.clip(gd, a_min=0, a_max=None)
    away_margin = np.clip(-gd, a_min=0, a_max=None)
    win_margin = np.where(merged["HOME_WINS"].to_numpy() == 1, home_margin,
                   np.where(merged["AWAY_WINS"].to_numpy() == 1, away_margin, 0.0))
    return win_margin

def weights_from_margin(margin: np.ndarray, scheme: str, beta: float = 0.25,
                        alpha: float = 0.2, base: float = 1.0,
                        cap: Optional[float] = None) -> np.ndarray:
    """
    - scheme="none" : poids = 1
    - scheme="linear" : base + beta * margin
    - scheme="exp"   : base + (exp(alpha*margin)-1) * beta_exp (ici on réutilise beta comme facteur d'échelle)
    """
    if scheme == "none":
        w = np.ones_like(margin, dtype=float)
    elif scheme == "linear":
        w = base + beta * margin
    elif scheme == "exp":
        w = base + (np.expm1(alpha * margin)) * 1.0 
    else:
        raise ValueError("scheme must be in {'none','linear','exp'}")
    if cap is not None:
        w = np.minimum(w, cap)
    return w.astype(float)


# Entraînement (une stratégie)
def train_one_scenario(X_feat: np.ndarray, y_cls: np.ndarray, feature_names: List[str],
                       weights: np.ndarray, cfg: TrainConfig, run_name: str) -> Dict:
    info(f"Split train/val ({run_name}) …")
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_feat, y_cls, weights,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y_cls
    )
    ok(f"{run_name} | Train: {X_tr.shape}, Val: {X_val.shape}")

    info(f"GridSearchCV ({run_name}) …")
    base_model = RandomForestClassifier(
        n_estimators=200,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight=None,
    )
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=cfg.param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_tr, y_tr, **{"sample_weight": w_tr})
    ok(f"{run_name} | Best params: {grid.best_params_} | best CV acc: {grid.best_score_:.4f}")
    model: RandomForestClassifier = grid.best_estimator_

    info(f"Évaluation val ({run_name}) …")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    ok(f"{run_name} | Validation accuracy: {acc:.4f}")

    cls_rep = classification_report(
        y_val, y_pred,
        target_names=["HOME_WINS","DRAW","AWAY_WINS"],
        digits=4
    )
    cm = confusion_matrix(y_val, y_pred).tolist()

    importances = getattr(model, "feature_importances_", None)
    imp_path = MODELS_DIR / f"rf_feature_importances_{run_name}.csv"
    top_feats = None
    if importances is not None:
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False)
        top_feats = imp_df.head(30).to_dict(orient="records")
        imp_df.to_csv(imp_path, index=False)
        ok(f"{run_name} | Importances: {imp_path}")

    # Sauvegardes
    model_path = MODELS_DIR / f"random_forest_{run_name}.pkl"
    joblib.dump(model, model_path)
    ok(f"{run_name} | Modèle: {model_path}")

    metrics = {
        "run_name": run_name,
        "val_accuracy": acc,
        "best_params": grid.best_params_,
        "confusion_matrix": cm,
        "class_report": cls_rep,
        "n_features": len(feature_names),
    }
    if top_feats is not None:
        metrics["top_features"] = top_feats

    with open(MODELS_DIR / f"metrics_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    ok(f"{run_name} | Métriques: models/metrics_{run_name}.json")

    return {
        "run_name": run_name,
        "val_accuracy": acc,
        "best_params": grid.best_params_,
        "model_path": str(model_path),
        "importances_path": str(imp_path),
    }


# Main: lance les 3 scénarios
def main(cfg: TrainConfig = TrainConfig()):
    info("Chargement des données …")
    X = load_X()
    y_all, y_supp = load_y_and_supp()

    info("Construction X, y …")
    X_feat, y_cls, feature_names = build_Xy_for_training(X, y_all, max_cardinality=cfg.max_cardinality)

    info("Calcul de la marge de victoire …")
    margin = compute_win_margin(y_all, y_supp)

    scenarios = [
        # 1) pas de poids
        {"run_name": "no_weight", "scheme": "none",   "beta": 0.0,  "alpha": 0.0, "cap": None},
        # 2) linéaire léger
        {"run_name": "linear_small", "scheme": "linear","beta": 0.25, "alpha": 0.0, "cap": None},
        # 3) exponentiel doux 
        {"run_name": "exp_soft", "scheme": "exp",    "beta": 1.0,  "alpha": 0.2, "cap": 5.0},
    ]

    results = []
    for sc in scenarios:
        info(f"=== Scenario: {sc['run_name']} ===")
        w = weights_from_margin(
            margin=margin,
            scheme=sc["scheme"],
            beta=sc["beta"],
            alpha=sc["alpha"],
            base=1.0,
            cap=sc["cap"]
        )
        # stats de poids
        print(f"[ok] {sc['run_name']} weights → min={w.min():.3f} | median={np.median(w):.3f} | max={w.max():.3f}")

        res = train_one_scenario(X_feat, y_cls, feature_names, w, cfg, run_name=sc["run_name"])
        results.append(res)

    # comparatif
    comp = pd.DataFrame(results).sort_values("val_accuracy", ascending=False)
    comp.to_csv(MODELS_DIR / "weighting_compare.csv", index=False)
    print("\n=== Résumé des scénarios ===")
    print(comp)

    # Déployer le meilleur pour la prédiction par défaut
    best = comp.iloc[0]
    best_model = Path(best["model_path"])
    best_imps  = Path(best["importances_path"])

    # copie sous les noms attendus par le script de prédiction
    shutil.copyfile(best_model, MODELS_DIR / "random_forest.pkl")
    shutil.copyfile(best_imps,  MODELS_DIR / "rf_feature_importances.csv")
    ok(f"Meilleur modèle déployé: {best_model.name} → models/random_forest.pkl")
    ok(f"Importances déployées: {best_imps.name} → models/rf_feature_importances.csv")

    info("Terminé ✅")


if __name__ == "__main__":
    main()
