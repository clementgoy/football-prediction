
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Chemins
DATA_ROOT = Path("data")
PROCESSED_DIR = DATA_ROOT / "processed"
TRAIN_PCA_CSV = PROCESSED_DIR / "train_merged_pca.csv"
Y_CSV = PROCESSED_DIR / "y_train_aligned.csv"

def main():
    results = {}
    print("Chargement des données avec PCA...")
    if not TRAIN_PCA_CSV.exists():
        print(f"Erreur: {TRAIN_PCA_CSV} n'existe pas.")
        return

    df = pd.read_csv(TRAIN_PCA_CSV, low_memory=False)
    
    # Charger les targets si possible
    if Y_CSV.exists():
        y = pd.read_csv(Y_CSV, low_memory=False)
        if "ID" in df.columns and "ID" in y.columns:
            df = df.merge(y[["ID", "HOME_WINS", "DRAW", "AWAY_WINS"]], on="ID", how="left")
            df['TARGET'] = df[["HOME_WINS", "DRAW", "AWAY_WINS"]].idxmax(axis=1)
    
    # 1. Stats PCA_1
    desc = df['PCA_1'].describe().to_dict()
    results["pca1_stats"] = desc
    
    # Seuil
    threshold = -50 # Un peu plus large pour être sûr d'attraper le mur
    outliers = df[df['PCA_1'] < threshold]
    normal = df[df['PCA_1'] >= threshold]
    
    results["count_outliers"] = len(outliers)
    results["count_normal"] = len(normal)
    
    if len(outliers) == 0:
        results["status"] = "No outliers found"
    else:
        results["status"] = "Found outliers"
        
        # 2. Colonnes suspectes
        cols_to_check = [c for c in df.columns if "PCA_" not in c and c not in ["ID", "HOME_WINS", "DRAW", "AWAY_WINS", "TARGET", "match_id"]]
        suspicious = []
        for col in cols_to_check:
            # On check le type
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            mean_out = float(outliers[col].mean())
            mean_norm = float(normal[col].mean())
            
            # Si c'est 0 partout dans les outliers
            if mean_out == 0 and mean_norm != 0:
                suspicious.append({"col": col, "reason": "Zeros", "mean_out": mean_out, "mean_norm": mean_norm})
            elif abs(mean_out - mean_norm) > abs(mean_norm) * 2 and abs(mean_norm) > 0.01:
                 suspicious.append({"col": col, "reason": "Diff", "mean_out": mean_out, "mean_norm": mean_norm})
        
        # On trie par 'Zeros' d'abord
        suspicious.sort(key=lambda x: x["reason"], reverse=True)
        results["suspicious_cols_top20"] = suspicious[:20]
        results["suspicious_count"] = len(suspicious)

        # 3. Targets
        if 'TARGET' in df.columns:
            results["target_dist_outliers"] = outliers['TARGET'].value_counts(normalize=True).to_dict()
            results["target_dist_normal"] = normal['TARGET'].value_counts(normalize=True).to_dict()

        # 4. IDs
        results["id_min_out"] = int(outliers['ID'].min())
        results["id_max_out"] = int(outliers['ID'].max())
        
        # 5. Objets (Strings)
        obj_cols = df.select_dtypes(include=['object']).columns
        obj_stats = {}
        for c in obj_cols:
            if c in ["ID", "TARGET"]: continue
            # Top 3 values
            top3 = outliers[c].value_counts().head(3).to_dict()
            if top3:
                obj_stats[c] = top3
        results["string_cols_stats"] = obj_stats

    out_file = PROCESSED_DIR.parent.parent / "outputs" / "report_outliers.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Rapport écrit dans {out_file}")

if __name__ == "__main__":
    main()
