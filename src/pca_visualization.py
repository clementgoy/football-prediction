import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))

from src.build_dataset import build_Xy

def main():
    # Définition des chemins
    DATA_ROOT = Path("data")
    PROCESSED_DIR = DATA_ROOT / "processed"
    TRAIN_CSV = PROCESSED_DIR / "train_merged.csv"
    Y_CSV = PROCESSED_DIR / "y_train_aligned.csv"
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True) # On crée le dossier s'il existe pas
    PLOT_PATH = OUTPUT_DIR / "pca_plot.png"

    print(f"Chargement des fichiers {TRAIN_CSV} et {Y_CSV}...")
    if not TRAIN_CSV.exists() or not Y_CSV.exists():
        print(f"Oups, fichiers introuvables. Vérifie tes paths !")
        sys.exit(1)

    X_raw = pd.read_csv(TRAIN_CSV)
    y_raw = pd.read_csv(Y_CSV)

    # Utilisation de la logique existante pour nettoyer X et y
    print("Préparation des données...")
    # build_Xy vire les ID et convertit Y en index de classe (0, 1, 2)
    X, y = build_Xy(X_raw, y_raw)

    # Nettoyage supplémentaire pour l'ACP (faut pas de NaN sinon ça plante)
    print("Nettoyage des valeurs manquantes...")
    X = X.select_dtypes(include=["number"]).astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # On standardise les features (centrer-réduire)
    print("Standardisation en cours...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # On applique l'ACP sur 2 dimensions pour pouvoir dessiner
    print("Application de l'ACP (n_components=2)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Calcul de la variance expliquée
    explained_var = pca.explained_variance_ratio_
    print(f"Variance expliquée : {explained_var} (Total: {sum(explained_var):.2f})")

    # Création du graphique
    print(f"Génération du graphique dans {PLOT_PATH}...")
    plt.figure(figsize=(10, 8))
    class_names = ["Home Win", "Draw", "Away Win"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

    for i, target_name in enumerate(class_names):
        plt.scatter(
            X_pca[y == i, 0], 
            X_pca[y == i, 1], 
            color=colors[i], 
            alpha=0.5, 
            label=target_name,
            s=10
        )

    plt.title("Visualisation ACP des features Football (2 axes)")
    plt.xlabel(f"Axe Principal 1 ({explained_var[0]:.2%})")
    plt.ylabel(f"Axe Principal 2 ({explained_var[1]:.2%})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(PLOT_PATH)
    print("C'est bon, graphique enregistré !")

if __name__ == "__main__":
    main()
