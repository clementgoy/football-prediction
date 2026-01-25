from __future__ import annotations

import argparse 
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# ATTENTION : sur UBUNTU mettre "DATA_ROOT = Path("data")" et sur windows : "DATA_ROOT = Path("../data")"
DATA_ROOT = Path("data")
TRAIN_DIR = DATA_ROOT / "Train_Data"
TEST_DIR  = DATA_ROOT / "Test_Data"

Y_TRAIN_PATH = DATA_ROOT / "Y_train_1rknArQ.csv"         
Y_SUPP_PATH  = DATA_ROOT / "benchmark_and_extras" / "Y_train_supp.csv"

OUT_DIR = DATA_ROOT / "processed"


def info(msg: str) -> None:
    print(f"\n infooo : {msg}")

def ok(msg: str) -> None:
    print(f"\n Beau gossseeee : {msg}")

#Return the first CSV file in *directory* whose filename contains a token. 
#C'est utilisé pour trouver les fichiers tout seul si on donne juste le dossier
#C'est récursif pour aller chercher dans les sous-dossiers (pratique pour garder la structure du challenge)
def discover_file(directory: Path, contains: str) -> Optional[Path]:
    candidates = sorted(p for p in directory.rglob('*.csv') if contains in p.name)
    return candidates[0] if candidates else None

#Ajoute un préfixe à toutes les colonnes sauf celles qu'on veut exclure (genre l'ID)
def enforce_prefix(df: pd.DataFrame, prefix: str, exclude: Tuple[str, ...] = ("ID",)) -> pd.DataFrame:
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in exclude}
    return df.rename(columns=rename)

#Lit un CSV avec des paramètres par défaut un peu optimisés et affiche un petit message
def read_csv(path: Path, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    info(f"Chargement de {path.name} ...")
    df = pd.read_csv(path, low_memory=False, usecols=usecols)
    ok(f"{path.name} chargé : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    return df

def aggregate_players_from_csv(path: Path, side_prefix: str, chunksize: int = 100_000) -> pd.DataFrame:
    """
    Alors là c'est la fonction un peu spéciale pour mon PC qui n'a pas beaucoup de RAM.
    Au lieu de tout charger d'un coup (ce qui fait planter mon ordi), on lit le fichier petit bout par petit bout (streaming/chunks).
    Pour chaque joueur :
      - On calcule la somme, la moyenne et l'écart-type (std) des stats.
      - On compte le nombre d'entrées.
    
    Ca m'a permit de travailler pendant les vacances notament quand je n'avais pas accès à l'ordinateur de la salle ia.
    """
    info(f"Traitement en mode streaming (pour économiser la RAM) de {path.name} ...")

    # etit échantillon pour détecter les colonnes numériques
    sample = pd.read_csv(path, nrows=2000, low_memory=False)
    if 'ID' not in sample.columns:
        raise ValueError(f"{path.name} must contain an 'ID' column")
    numeric_cols = sample.select_dtypes(include=['number']).columns.tolist()

    # On ne charge que ID + colonnes numériques
    usecols = ['ID'] + numeric_cols

    sum_acc   = None  
    sumsq_acc = None  
    count_acc = None

    # Lecture par morceaux
    for chunk in pd.read_csv(path, low_memory=False, usecols=usecols, chunksize=chunksize):
    
        sums = chunk.groupby('ID')[numeric_cols].sum()

        sumsq = (chunk[numeric_cols] ** 2).groupby(chunk['ID']).sum()

        counts = chunk.groupby('ID').size()

        sum_acc   = sums   if sum_acc   is None else sum_acc.add(sums,   fill_value=0)
        sumsq_acc = sumsq  if sumsq_acc is None else sumsq_acc.add(sumsq, fill_value=0)
        count_acc = counts if count_acc is None else count_acc.add(counts, fill_value=0)

    # Finalisation : mean, std
    mean = sum_acc.div(count_acc, axis=0)
    var  = sumsq_acc.div(count_acc, axis=0) - (mean ** 2)
    std  = var.clip(lower=0).pow(0.5)

    # Construction du DataFrame final
    out = pd.DataFrame(index=sum_acc.index)
    for col in numeric_cols:
        out[f"{side_prefix}{col}_sum"]  = sum_acc[col]
        out[f"{side_prefix}{col}_mean"] = mean[col]
        out[f"{side_prefix}{col}_std"]  = std[col]
    out[f"{side_prefix}player_count"] = count_acc
    out = out.reset_index()
    ok(f"Aggregated players (stream) → {out.shape[0]} rows × {out.shape[1]} cols")
    return out

#Fonction pour fusionner (merge) deux tables sur l'ID sans se prendre la tête
def safe_merge(left: pd.DataFrame, right: pd.DataFrame, how: str = 'inner') -> pd.DataFrame:
    before = left.shape
    merged = left.merge(right, on='ID', how=how)
    ok(f"Fusion de {before} avec {right.shape} --> Résultat : {merged.shape}")
    return merged

def clean_unique_by_id(df: pd.DataFrame, id_col: str = 'ID') -> pd.DataFrame:
    """
    Nettoyage pour être sûr d'avoir une seule ligne par ID.
    Étapes :
    1) Vire les doublons exacts
    2) Vire les doublons d'ID (garde le premier)
    3) Trie par ID pour que ce soit propre
    """
    assert id_col in df.columns, f"Il manque la colonne ID : {id_col}"

    # Vire les lignes complètements identiques
    n_before = len(df)
    dup_all = int(df.duplicated(keep='first').sum())
    if dup_all:
        info(f"Y'avait {dup_all} doublons exacts -> poubelle !")
        df = df.drop_duplicates(keep='first')

    # Vire les IDs en double
    dups_mask = df.duplicated(subset=[id_col], keep='first')
    n_dup_ids = int(dups_mask.sum())
    if n_dup_ids:
        info(f"Y'avait {n_dup_ids} IDs en double : on garde le premier et on jette les autres.")
        df = df[~dups_mask].copy()

    # Petit tri pour faire propre
    if id_col in df.columns:
        df = df.sort_values(id_col).reset_index(drop=True)

    n_after = len(df)
    removed = n_before - n_after
    ok(f"Nettoyage par ID fini : {removed} lignes supprimées. IDs uniques : {df[id_col].is_unique}")
    return df


#Le cœur du fichier : ici on construit tout !

"""
Construit la table finale pour un split (train ou test).
Étapes :
1) Charge les équipes (home et away)
2) Les fusionne
3) Charge et agrège les joueurs
4) Remet tout dans l'ordre
"""
def build_split(*,
    home_team_path: Path,
    away_team_path: Path,
    home_player_path: Optional[Path],
    away_player_path: Optional[Path],
    lenient: bool = False,
) -> pd.DataFrame:
    #Equipes
    home_team = read_csv(home_team_path)
    away_team = read_csv(away_team_path)

    #Prefixes pour pas confondre domicile et extérieur
    home_team = enforce_prefix(home_team, 'home_team')
    away_team = enforce_prefix(away_team, 'away_team')

    #Fusion des équipes
    how = 'left' if lenient else 'inner'
    teams = safe_merge(home_team, away_team, how=how)

    #Joueurs
    if home_player_path and home_player_path.exists():
        home_player = aggregate_players_from_csv(home_player_path, 'home_player_')
        teams = safe_merge(teams, home_player, how=how)
    else: 
        info("no home player file provided, skipping.")
    
    if away_player_path and away_player_path.exists():
        # Version streaming 
        away_player = aggregate_players_from_csv(away_player_path, 'away_player_')
        teams = safe_merge(teams, away_player, how=how)
    else: 
        info("no away player file provided, skipping.")

    #Reorder: ID first
    cols = ['ID'] + [c for c in teams.columns if c != 'ID']
    teams = teams[cols]

    #sort by ID for reproductinility
    teams = teams.sort_values('ID').reset_index(drop=True)

    teams = clean_unique_by_id(teams, id_col='ID')

    ok("Split build completeeee")
    return teams


# Partie CLI pour lancer le script avec des arguments différents 

def parse_args() -> argparse.Namespace:
    # Fonction qui définit tous les paramètres qu'on peut passer en ligne de commande
    p = argparse.ArgumentParser(description="Fusionner les CSV de foot bruts pour en faire une belle table")

    p.add_argument('--train-home-team', type=Path, default=None)
    p.add_argument('--train-away-team', type=Path, default=None)
    p.add_argument('--train-home-player', type=Path, default=None)
    p.add_argument('--train-away-player', type=Path, default=None)

    p.add_argument('--test-home-team', type=Path, default=None)
    p.add_argument('--test-away-team', type=Path, default=None)
    p.add_argument('--test-home-player', type=Path, default=None)
    p.add_argument('--test-away-player', type=Path, default=None)

    p.add_argument('--train-dir', type=Path, default=None, help='Dossier contenant les CSV de train')
    p.add_argument('--test-dir',  type=Path, default=None, help='Dossier contenant les CSV de test')

    # Targets
    p.add_argument('--y-train', type=Path, default=None, help='Fichier Y_train optionnel (ID + HOME_WINS/DRAW/AWAY_WINS)')
    p.add_argument('--y-train-supp', type=Path, default=None, help='Fichier Y_train_supp optionnel (ID + GOAL_DIFF_HOME_AWAY)')

    # Comportement
    p.add_argument('--lenient', action='store_true', help="Utiliser LEFT join au lieu de INNER (garde plus de lignes même si incomplet)")

    # Sortie
    p.add_argument('--out-dir', type=Path, default=Path('../data/processed'))

    return p.parse_args()

"""
Découverte automatique des fichiers standards dans train_dir/test_dir.
On cherche des CSV qui contiennent des mots-clés comme "home_team", "away_player", etc.
On renvoie un dico des chemins trouvés (ou None si introuvable).
"""
def discover_inputs(train_dir: Optional[Path], test_dir: Optional[Path]) -> Dict[str, Optional[Path]]:

    def discover_pair(root_dir: Optional[Path], split: str) -> Dict[str, Optional[Path]]:
        if root_dir is None:
            return {
                f'{split}_home_team': None,
                f'{split}_away_team': None,
                f'{split}_home_player': None,
                f'{split}_away_player': None,
            }
        return {
            f'{split}_home_team':  discover_file(root_dir, 'home_team'),
            f'{split}_away_team':  discover_file(root_dir, 'away_team'),
            f'{split}_home_player': discover_file(root_dir, 'home_player'),
            f'{split}_away_player': discover_file(root_dir, 'away_player'),
        }

    inputs: Dict[str, Optional[Path]] = {}
    inputs.update(discover_pair(train_dir, 'train'))
    inputs.update(discover_pair(test_dir,  'test'))
    return inputs

"""
Charge le fichier des cibles (Y_train) s'il existe, et vérifie les colonnes attendues.
Colonnes requises : ID + HOME_WINS + DRAW + AWAY_WINS.
Renvoie uniquement les colonnes utiles (dans le bon ordre).
"""
def load_y_train(y_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not y_path:
        info("Pas de Y_train fourni -> on saute l'étape des cibles.")
        return None
    y = read_csv(y_path)
    need = {'ID', 'HOME_WINS', 'DRAW', 'AWAY_WINS'}
    missing = need - set(y.columns)
    if missing:
        raise ValueError(f"Il manque des colonnes dans Y_train : {missing}")
    return y[['ID', 'HOME_WINS', 'DRAW', 'AWAY_WINS']]

def load_y_train_supp(y_supp_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not y_supp_path:
        info("Pas de Y_train_supp fourni -> on saute l'étape des stats supplémentaires.")
        return None
    y = read_csv(y_supp_path)
    need = {'ID', 'GOAL_DIFF_HOME_AWAY'}
    missing = need - set(y.columns)
    if missing:
        raise ValueError(f"Il manque des colonnes dans Y_train_supp : {missing}")
    return y[['ID', 'GOAL_DIFF_HOME_AWAY']]


# Écrit les CSV finaux + un petit schema.json pour garder une trace de ce qu'on a généré.
def save_artifacts(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / 'train_merged.csv'
    test_path = out_dir / 'test_merged.csv'

    info("Sauvegarde des CSVs ...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    ok(f"Fichier écrit : {train_path} ({train_df.shape[0]}×{train_df.shape[1]})")
    ok(f"Fichier écrit : {test_path} ({test_df.shape[0]}×{test_df.shape[1]})")

    schema = {
        'train': {
            'rows': int(train_df.shape[0]),
            'cols': int(train_df.shape[1]),
            'columns_sample': train_df.columns[:10].tolist(),
        },
        'test': {
            'rows': int(test_df.shape[0]),
            'cols': int(test_df.shape[1]),
            'columns_sample': test_df.columns[:10].tolist(),
        },
    }
    with open(out_dir / 'schema.json', 'w') as f:
        json.dump(schema, f, indent=2)
    ok("schema.json sauvegardé")


def main() -> None:
    info("Recherche des fichiers d'entrée ...")
    discovered = discover_inputs(TRAIN_DIR, TEST_DIR)

    for k, v in discovered.items():
        info(f"{k}: {v}")

    train_home_team   = discovered['train_home_team']
    train_away_team   = discovered['train_away_team']
    train_home_player = discovered['train_home_player']
    train_away_player = discovered['train_away_player']

    test_home_team    = discovered['test_home_team']
    test_away_team    = discovered['test_away_team']
    test_home_player  = discovered['test_home_player']
    test_away_player  = discovered['test_away_player']

    # On a besoin au moins des fichiers équipes pour faire quelque chose
    need_train = [train_home_team, train_away_team]
    need_test  = [test_home_team,  test_away_team]
    if any(p is None for p in need_train + need_test):
        raise SystemExit("Il manque les CSV des équipes (home/away) dans Data/Train_Data ou Data/Test_Data.")

    info("Construction du split TRAIN ...")
    train = build_split(
        home_team_path=train_home_team,
        away_team_path=train_away_team,
        home_player_path=train_home_player,
        away_player_path=train_away_player,
        lenient=False,
    )

    # Sauvegarde précoce : si le test plante derrière, au moins le train est déjà écrit
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / 'train_merged.csv', index=False)
    ok("train_merged.csv écrit (sauvegarde précoce)")

    # Alignement des cibles sur les IDs du train final (pour éviter tout décalage)
    if Y_TRAIN_PATH.exists():
        y = load_y_train(Y_TRAIN_PATH)
        if y is not None:
            train_ids = train[['ID']].sort_values('ID').reset_index(drop=True)
            y_aligned = train_ids.merge(y, on='ID', how='left')
            n_missing = int(y_aligned[['HOME_WINS','DRAW','AWAY_WINS']].isna().any(axis=1).sum())
            info(f"Cibles alignées sur les IDs de train. Manquants : {n_missing}")
            y_aligned.to_csv(OUT_DIR / 'y_train_aligned.csv', index=False)
            ok("y_train_aligned.csv sauvegardé")

    if Y_SUPP_PATH.exists():
        y_supp = load_y_train_supp(Y_SUPP_PATH)
        if y_supp is not None:
            train_ids = train[['ID']].sort_values('ID').reset_index(drop=True)
            y_supp_aligned = train_ids.merge(y_supp, on='ID', how='left')
            n_missing_supp = int(y_supp_aligned[['GOAL_DIFF_HOME_AWAY']].isna().any(axis=1).sum())
            info(f"Y_train_supp aligné. Manquants : {n_missing_supp}")
            y_supp_aligned.to_csv(OUT_DIR / 'y_train_supp_aligned.csv', index=False)
            ok("y_train_supp_aligned.csv sauvegardé")

    info("Construction du split TEST …")
    test = build_split(
        home_team_path=test_home_team,
        away_team_path=test_away_team,
        home_player_path=test_home_player,
        away_player_path=test_away_player,
        lenient=False,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test.to_csv(OUT_DIR / 'test_merged.csv', index=False)
    ok("test_merged.csv écrit (sauvegarde précoce)")

    save_artifacts(train, test, OUT_DIR)
    ok("Tout est parfaittttttt")


if __name__ == '__main__':
    main()