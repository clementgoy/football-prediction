import argparse
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import joblib

def main():
    # Définition des chemins par défaut
    # Devine où sont les fichiers si l'utilisateur ne précise rien
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    DATA = ROOT / "data"
    PROCESSED = DATA / "processed"
    MODELS = ROOT / "models"
    OUTPUTS = ROOT / "outputs" / "submissions"
    
    # Arguments du script
    parser = argparse.ArgumentParser(description="Prédiction avec le modèle Random Forest PCA (étudiant style)")
    parser.add_argument("--test-csv", default=PROCESSED / "test_merged_pca.csv", help="Le fichier test AVEC les colonnes PCA")
    parser.add_argument("--model", default=MODELS / "rf_pca.joblib", help="Le fichier du modèle entraîné")
    parser.add_argument("--features", default=MODELS / "rf_pca_features.json", help="Le fichier JSON qui contient la liste des colonnes")
    parser.add_argument("--out-csv", default=OUTPUTS / "submission_rf_pca.csv", help="Où on sauvegarde le fichier à soumettre")

    args = parser.parse_args()
    
    print("--- Démarrage de la prédiction RF PCA ---")

    # 1. Chargement du modèle
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Erreur : Impossible de trouver le modèle ici : {model_path}")
        sys.exit(1)
    
    print(f"Chargement du modèle depuis {model_path}...")
    model = joblib.load(model_path)

    # 2. Chargement de la liste des features
    # (important pour être sûr qu'on donne les colonnes dans le bon ordre au modèle)
    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"Erreur : Pas de fichier de features ({feat_path}). Le modèle a besoin de savoir quelles colonnes utiliser.")
        sys.exit(1)
        
    print(f"Lecture des features attendues depuis {feat_path}...")
    with open(feat_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)
        if "feature_names" not in data_json:
             print("Erreur : Le JSON est bizarre, il n'a pas la clé 'feature_names'.")
             sys.exit(1)
        expected_cols = data_json["feature_names"]
    
    print(f"Le modèle attend {len(expected_cols)} colonnes.")

    # 3. Chargement des données de test
    test_path = Path(args.test_csv)
    if not test_path.exists():
        print(f"Erreur : Fichier de test introuvable ({test_path}). As-tu bien lancé add_pca_to_csv ?")
        sys.exit(1)
        
    print(f"Chargement des données de test : {test_path} ...")
    test_df = pd.read_csv(test_path, low_memory=False)
    
    # Vérification qu'on a bien l'ID
    if "ID" not in test_df.columns:
        print("Erreur critique : Pas de colonne 'ID' dans le test. On ne pourra pas faire la soumission.")
        sys.exit(1)
        
    ids = test_df["ID"]

    # 4. Alignement des colonnes
    print("Alignement des colonnes (features)...")
    # On crée une matrice vide avec les bonnes colonnes
    # Si manque des colonnes dans le test on met des 0
    # Si y a des colonnes en trop dans le test on les ignore
    X_test = pd.DataFrame(index=test_df.index)
    
    for col in expected_cols:
        if col in test_df.columns:
            X_test[col] = test_df[col]
        else:
            X_test[col] = 0.0
            
    # On s'assure que l'ordre est le même
    X_test = X_test[expected_cols]
    
    # Nettoyage final comme pour l'entraînement
    print("Remplissage des NaN par 0.0 ...")
    X_test = X_test.fillna(0.0)
    
    # 5. Prédiction
    print("Calcul des prédictions...")
    
    if hasattr(model, "predict_proba"):
        # On récupère les probabilités : [P(Home), P(Draw), P(Away)]
        proba = model.predict_proba(X_test)
        
        # Petit check de sécurité
        if proba.shape[1] != 3:
            print(f"Le modèle a renvoyé {proba.shape[1]} classes au lieu de 3.")
    else:
        print("Le modèle ne sait pas faire de probabilités.")
        sys.exit(1)

    # 6. Création du fichier de soumission
    print("Création du fichier de résultat...")
    submission = pd.DataFrame({
        "ID": ids,
        "HOME_WINS": proba[:, 0],
        "DRAW": proba[:, 1],
        "AWAY_WINS": proba[:, 2]
    })
    
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    submission.to_csv(out_path, index=False)
    print(f"C'est gagné ! Fichier sauvegardé ici : {out_path}")
    print("Tu peux maintenant l'envoyer sur la plateforme.")

if __name__ == "__main__":
    main()
