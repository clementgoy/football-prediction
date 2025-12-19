import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

def main():
    # Paths
    DATA_ROOT = Path("data")
    PROCESSED_DIR = DATA_ROOT / "processed"
    
    TRAIN_CSV = PROCESSED_DIR / "train_merged.csv"
    TEST_CSV = PROCESSED_DIR / "test_merged.csv"
    
    OUT_TRAIN_CSV = PROCESSED_DIR / "train_merged_pca.csv"
    OUT_TEST_CSV = PROCESSED_DIR / "test_merged_pca.csv"

    print(f"Loading data...")
    if not TRAIN_CSV.exists():
        print(f"Error: {TRAIN_CSV} not found.")
        sys.exit(1)
        
    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    print(f"Train data loaded: {train_df.shape}")
    
    test_df = None
    if TEST_CSV.exists():
        test_df = pd.read_csv(TEST_CSV, low_memory=False)
        print(f"Test data loaded: {test_df.shape}")
    else:
        print(f"Warning: {TEST_CSV} not found. Skipping test data.")

    # Select numeric features for PCA
    # Exclude ID and non-numeric columns
    # We also exclude target columns if they are present in train_df (though train_merged usually doesn't have targets merged in it, but let's be safe)
    exclude_cols = ["ID", "HOME_WINS", "DRAW", "AWAY_WINS", "match_id", "MatchID"]
    
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    print(f"Selected {len(feature_cols)} numeric features for PCA.")
    
    # Prepare data for PCA
    print("Preprocessing for PCA (FillNA 0, Scale)...")
    
    # Fill NaNs with 0
    X_train = train_df[feature_cols].fillna(0.0)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # PCA
    n_components = 10
    print(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Add to Train DataFrame
    pca_col_names = [f"PCA_{i+1}" for i in range(n_components)]
    train_pca_df = pd.DataFrame(X_train_pca, columns=pca_col_names, index=train_df.index)
    train_out = pd.concat([train_df, train_pca_df], axis=1)
    
    print(f"Saving {OUT_TRAIN_CSV}...")
    train_out.to_csv(OUT_TRAIN_CSV, index=False)
    
    # Process Test Data if available
    if test_df is not None:
        print("Processing Test data...")
        # Ensure same columns
        missing_cols = set(feature_cols) - set(test_df.columns)
        if missing_cols:
            print(f"Warning: Test data missing columns: {missing_cols}")
            # Add missing columns as 0
            for c in missing_cols:
                test_df[c] = 0.0
                
        X_test = test_df[feature_cols].fillna(0.0)
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)
        
        test_pca_df = pd.DataFrame(X_test_pca, columns=pca_col_names, index=test_df.index)
        test_out = pd.concat([test_df, test_pca_df], axis=1)
        
        print(f"Saving {OUT_TEST_CSV}...")
        test_out.to_csv(OUT_TEST_CSV, index=False)

    print("Done.")

if __name__ == "__main__":
    main()
