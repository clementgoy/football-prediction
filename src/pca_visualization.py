import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
import os

# Add src to path to import build_features if needed, but we can also just implement simple logic
# The original code uses src.build_dataset.build_Xy, let's try to reuse it if possible.
# Adding current dir to path
sys.path.append(str(Path(__file__).parent.parent))

from src.build_dataset import build_Xy

def main():
    # Paths
    DATA_ROOT = Path("data")
    PROCESSED_DIR = DATA_ROOT / "processed"
    TRAIN_CSV = PROCESSED_DIR / "train_merged.csv"
    Y_CSV = PROCESSED_DIR / "y_train_aligned.csv"
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_PATH = OUTPUT_DIR / "pca_plot.png"

    print(f"Loading data from {TRAIN_CSV} and {Y_CSV}...")
    if not TRAIN_CSV.exists() or not Y_CSV.exists():
        print(f"Error: Data files not found.")
        sys.exit(1)

    X_raw = pd.read_csv(TRAIN_CSV)
    y_raw = pd.read_csv(Y_CSV)

    # Use existing logic to clean/prep X and y
    print("Preprocessing data...")
    # build_Xy drops ID columns and converts Y to class index (0, 1, 2)
    # y: 0=HOME_WINS, 1=DRAW, 2=AWAY_WINS (argmax of respective cols)
    X, y = build_Xy(X_raw, y_raw)

    # Further cleanup for PCA (fill NaNs)
    # The existing train_lgbm.py handles NaNs by replace inf and fillna(0)
    print("Cleaning missing values...")
    X = X.select_dtypes(include=["number"]).astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    print("Applying PCA (n_components=2)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Explain variance
    explained_var = pca.explained_variance_ratio_
    print(f"Explained variance ratio: {explained_var} (Total: {sum(explained_var):.2f})")

    # Plot
    print(f"Generating plot to {PLOT_PATH}...")
    plt.figure(figsize=(10, 8))
    class_names = ["Home Win", "Draw", "Away Win"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

    for i, target_name in enumerate(class_names):
        plt.scatter(
            X_pca[y == i, 0], 
            X_pca[y == i, 1], 
            color=colors[i], 
            alpha=0.5, 
            label=target_name,
            s=10
        )

    plt.title("PCA of Football Features (2 Components)")
    plt.xlabel(f"PC1 ({explained_var[0]:.2%})")
    plt.ylabel(f"PC2 ({explained_var[1]:.2%})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(PLOT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
