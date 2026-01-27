import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.features_diff import build_features_with_diff, add_interaction_features

# Prépare le jeu de test en reconstruisant les features exactement comme à l'entraînement
def build_X_test_with_diff(
    test_csv: str,
    feature_columns: list[str],
    id_col: str = "ID",
) -> tuple[pd.Series, pd.DataFrame]:
    df_raw = pd.read_csv(test_csv)
    if id_col not in df_raw.columns:
        raise ValueError(f"Le fichier test_csv doit contenir une colonne '{id_col}'.")

    ids = df_raw[id_col]

    X = build_features_with_diff(df_raw, drop_id_cols=True)

    if id_col in X.columns:
        X = X.drop(columns=[id_col])

    X = X.select_dtypes(include=["number"]).copy()

    X = add_interaction_features(X)

    X = X.reindex(columns=feature_columns, fill_value=0.0)

    print(f"[debug] X_test shape après alignement: {X.shape}")
    return ids, X

 # Charge le modèle, prédit les probabilités et génère le fichier de soumission
def predict_lgbm_multiclass_diff(
    test_csv: str,
    model_path: str,
    out_csv: str,
    id_col: str = "ID",
    submit_onehot: bool = False,
    alpha_draw: float = 1.0,
):
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact.get("feature_columns")
    best_iteration = artifact.get("best_iteration", None)

    if feature_columns is None:
        raise ValueError("feature_columns manquant dans l'artifact du modèle.")

    print(f"[load] Modèle chargé depuis: {model_path}")
    print(f"[load] nb features attendues: {len(feature_columns)}")

    ids, X_test = build_X_test_with_diff(
        test_csv=test_csv,
        feature_columns=feature_columns,
        id_col=id_col,
    )

    predict_kwargs = {}
    if best_iteration is not None:
        predict_kwargs["num_iteration"] = best_iteration

    proba = model.predict_proba(X_test, **predict_kwargs)
    proba = np.asarray(proba)
    if proba.shape[1] != 3:
        raise ValueError(
            f"Le modèle doit retourner des proba sur 3 classes, obtenu shape={proba.shape}"
        )

    if alpha_draw != 1.0:
        print(f"[info] Application d'un alpha_draw={alpha_draw} sur la classe Draw (1)")
        proba[:, 1] *= alpha_draw
        row_sums = proba.sum(axis=1, keepdims=True)
        proba = proba / row_sums

    if submit_onehot:
        print("[info] Génération d'une soumission one-hot (argmax)")
        preds = proba.argmax(axis=1)
        onehot = np.zeros_like(proba)
        onehot[np.arange(len(preds)), preds] = 1.0
        out_values = onehot
    else:
        print("[info] Génération d'une soumission probabiliste (proba brutes)")
        out_values = proba

    df_sub = pd.DataFrame(
        out_values,
        columns=["HOME_WINS", "DRAW", "AWAY_WINS"],
    )
    df_sub.insert(0, id_col, ids.values)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(out_path, index=False)
    print(f"[ok] Fichier de soumission écrit dans: {out_path.resolve()}")

 # Gère les arguments de la ligne de commande pour lancer la prédiction
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prédit les issues Home/Draw/Away avec un modèle LightGBM multiclass "
            "entraîné avec features diff et interactions."
        )
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="CSV test fusionné (ex: data/processed/test_merged.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Chemin vers le fichier modèle (joblib/pkl) entraîné.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Chemin de sortie du CSV de soumission.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="ID",
        help="Nom de la colonne ID (défaut: ID).",
    )
    parser.add_argument(
        "--submit-onehot",
        action="store_true",
        help="Si présent, soumet un one-hot (argmax) plutôt que les probas.",
    )
    parser.add_argument(
        "--alpha-draw",
        type=float,
        default=1.0,
        help="Facteur de repondération de la proba de la classe Draw (1.0 = sans effet).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_lgbm_multiclass_diff(
        test_csv=args.test_csv,
        model_path=args.model,
        out_csv=args.out_csv,
        id_col=args.id_col,
        submit_onehot=args.submit_onehot,
        alpha_draw=args.alpha_draw,
    )