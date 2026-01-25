import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

def main():
    # Définition des chemins vers les dossiers de données
    DATA_ROOT = Path("data")
    PROCESSED_DIR = DATA_ROOT / "processed"
    
    TRAIN_CSV = PROCESSED_DIR / "train_merged.csv"
    TEST_CSV = PROCESSED_DIR / "test_merged.csv"
    
    OUT_TRAIN_CSV = PROCESSED_DIR / "train_merged_pca.csv"
    OUT_TEST_CSV = PROCESSED_DIR / "test_merged_pca.csv"

    print(f"Chargement des données en cours...")
    if not TRAIN_CSV.exists():
        print(f"Erreur : le fichier {TRAIN_CSV} est introuvable (c'est bizarre).")
        sys.exit(1)
        
    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    print(f"Données d'entraînement chargées ! Taille : {train_df.shape}")
    
    test_df = None
    if TEST_CSV.exists():
        test_df = pd.read_csv(TEST_CSV, low_memory=False)
        print(f"Données de test chargées aussi : {test_df.shape}")
    else:
        print(f"Attention : on n'a pas trouvé le fichier de test {TEST_CSV}. On continue sans.")

    # On sélectionne seulement les colonnes numériques pour faire le PCA donc on enlève les ID 
    # et tout ce qui n'est pas des chiffres et onn enlève aussi les cibles (targets)
    exclude_cols = ["ID", "HOME_WINS", "DRAW", "AWAY_WINS", "match_id", "MatchID"]
    
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    print(f"On a trouvé {len(feature_cols)} colonnes numériques pour faire notre ACP.")
    
    # Préparation des données
    print("Mise à l'échelle (Scaling) et remplissage des trous (NaN)...")
    
    # On remplace les trous par des 0 (peut-être à ameliorer plus tard)
    X_train = train_df[feature_cols].fillna(0.0)
    
    # On utilise StandardScaler pour que toutes les variables aient la même importance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Lancement de l'ACP
    n_components = 10
    print(f"Calcul de l'ACP avec {n_components} composantes...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # On affiche combien d'infos on a gardé 
    print(f"Variance expliquée cumulée : {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Ajout des nouvelles colonnes PCA dans le DataFrame d'entraînement
    pca_col_names = [f"PCA_{i+1}" for i in range(n_components)]
    train_pca_df = pd.DataFrame(X_train_pca, columns=pca_col_names, index=train_df.index)
    train_out = pd.concat([train_df, train_pca_df], axis=1)
    
    print(f"Sauvegarde du nouveau fichier train dans {OUT_TRAIN_CSV}...")
    train_out.to_csv(OUT_TRAIN_CSV, index=False)
    
    # On fait pareil pour le test si on l'a
    if test_df is not None:
        print("Traitement des données de test...")
        # Il faut qu'on ait exactement les mêmes colonnes que pour l'entraînement
        missing_cols = set(feature_cols) - set(test_df.columns)
        if missing_cols:
            print(f"Attention : Il manque des colonnes dans le test : {missing_cols}")
            # On rajoute les colonnes manquantes remplies de 0
            for c in missing_cols:
                test_df[c] = 0.0
                
        X_test = test_df[feature_cols].fillna(0.0)
        X_test_scaled = scaler.transform(X_test) 
        X_test_pca = pca.transform(X_test_scaled)
        
        test_pca_df = pd.DataFrame(X_test_pca, columns=pca_col_names, index=test_df.index)
        test_out = pd.concat([test_df, test_pca_df], axis=1)
        
        print(f"Sauvegarde du fichier test modifié dans {OUT_TEST_CSV}...")
        test_out.to_csv(OUT_TEST_CSV, index=False)

    print("C'est fini !")

if __name__ == "__main__":
    main()
