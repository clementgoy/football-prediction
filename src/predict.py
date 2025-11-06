# src/predict_hgbc.py
# -*- coding: utf-8 -*-
"""
Prédiction pour l'artifact entraîné par train_hgbc.py

- Charge outputs/models/model.joblib (dict: model, selector, drop_cols, features)
- Construit X_test comme dans le train:
    * numeric only -> float32 -> fillna(0.0)
    * drop des colonnes corrélées (drop_cols)
    * aligne sur 'features' (colonnes sélectionnées par VarianceThreshold)
- Prédit avec le modèle HGBC et génère le CSV de soumission:
    ID, HOME, DRAW, AWAY (one-hot par défaut, ou proba avec --submit-proba)

Usage:
  python -m src.predict_hgbc \
    --test-csv data/processed/test_merged.csv \
    --artifact outputs/models/model.joblib \
    --out-csv outputs/submissions/submission_hgbc.csv
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from joblib import load

VALID_CLASS_NAMES = ["HOME_WINS", "DRAW", "AWAY_WINS"]

def parse_args():
    p = argparse.ArgumentParser(description="Predict HGBC (artifact de train_hgbc.py).")
    p.add_argument("--test-csv", required=True, help="Chemin du CSV test fusionné (avec la colonne ID).")
    p.add_argument("--artifact", default="outputs/models/model.joblib", help="Artifact joblib sauvegardé au train.")
    p.add_argument("--out-csv", required=True, help="Chemin du CSV de soumission à écrire.")
    p.add_argument("--id-col", default="ID", help="Nom de la colonne identifiant (défaut: ID).")
    p.add_argument("--submit-proba", action="store_true",
                   help="Écrit les probabilités au lieu du one-hot.")
    p.add_argument("--class-order", default="HOME,DRAW,AWAY",
                   help="Ordre des classes dans la soumission (défaut: HOME,DRAW,AWAY).")
    return p.parse_args()

def coerce_numeric(df, id_col):
    out = df.copy()
    for c in out.columns:
        if c == id_col:
            continue
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.fillna(0.0)
    return out

def map_proba_to_order(model_classes, proba, wanted_order):
    cls = list(model_classes)
    # Cas fréquent: classes = [0,1,2] -> HOME, DRAW, AWAY
    if all(isinstance(x, (int, np.integer)) for x in cls) and set(cls) == {0,1,2}:
        model_order = ["HOME_WINS", "DRAW", "AWAY_WINS"]
    else:
        model_order = [str(x).upper() for x in cls]

    if set(model_order) != set(VALID_CLASS_NAMES):
        raise ValueError(f"Classes du modèle {model_order} inattendues (attendues: {VALID_CLASS_NAMES}).")

    idx = [model_order.index(name) for name in wanted_order]
    return proba[:, idx]

def main():
    args = parse_args()
    class_order = [x.strip().upper() for x in args.class_order.split(",")]
    if set(class_order) != set(VALID_CLASS_NAMES):
        raise ValueError(f"--class-order doit être une permutation de {VALID_CLASS_NAMES}")

    # 1) Charger artifact
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"[err] Artifact introuvable: {artifact_path}", file=sys.stderr)
        sys.exit(2)
    artifact = load(artifact_path)
    model = artifact["model"]
    drop_cols = artifact.get("drop_cols", [])
    selected_features = artifact["features"]  
    
    # 2) Charger test
    test_df = pd.read_csv(args.test_csv)
    if args.id_col not in test_df.columns:
        raise ValueError(f"Colonne ID '{args.id_col}' introuvable dans {args.test_csv}")
    ids = test_df[args.id_col].copy()

    # 3) Pipeline minimal identique au train
    test_df = coerce_numeric(test_df, args.id_col)
    feats = test_df.drop(columns=[args.id_col], errors="ignore")

    # Drop des colonnes fortement corrélées (même liste qu'au train)
    feats = feats.drop(columns=[c for c in drop_cols if c in feats.columns], errors="ignore")

    # Aligner sur les features sélectionnées par VarianceThreshold au train
    # -> ajoute 0.0 pour les manquantes, supprime l'extra, impose l'ordre
    for c in selected_features:
        if c not in feats.columns:
            feats[c] = 0.0
    X = feats[selected_features].astype(np.float32)

    # 4) Prédire
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = map_proba_to_order(getattr(model, "classes_", [0,1,2]), proba, class_order)
    else:
        # Sécurité (peu probable avec HGBC)
        y_pred = model.predict(X)
        inv = {name:i for i,name in enumerate(class_order)}
        proba = np.zeros((len(y_pred), 3), dtype=float)
        for r, y in enumerate(y_pred):
            name = ("HOME_WINS", "DRAW", "AWAY_WINS")[int(y)] if isinstance(y, (int,np.integer)) else str(y).upper()
            proba[r, inv.get(name, 0)] = 1.0

    # 5) Construire soumission
    if args.submit_proba:
        submit = pd.DataFrame(proba, columns=class_order)
    else:
        argmax = np.argmax(proba, axis=1)
        onehot = np.zeros_like(proba, dtype=int)
        onehot[np.arange(len(argmax)), argmax] = 1
        submit = pd.DataFrame(onehot, columns=class_order)

    submit.insert(0, args.id_col, ids.values)

    # 6) Sauvegarde
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submit.to_csv(out_path, index=False)
    print(f"[ok] Soumission écrite -> {out_path.resolve()}")

if __name__ == "__main__":
    main()
