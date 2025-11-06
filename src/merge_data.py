from __future__ import annotations

import argparse 
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def info(msg: str) -> None:
    print(f"\n infooo : {msg}")

def ok(msg: str) -> None:
    print(f"\n Beau gossseeee : {msg}")

#Return the first CSV file in *directory* whose filename contains a token. 
#Used for lightweight auto-discovery when users pass only a folder.
def discover_file(directory: Path, contains: str) -> Optional[Path]:
    candidates = sorted(p for p in directory.glob('*.csv') if contains in p.name )
    return candidates[0] if candidates else None

#Add a prefix to all columns except those in *exclude*.
#This avoids name collisions after merges (e.g. shots_on_target exists in both home and away tables).
def enforce_prefix(df: pd.DataFrame, prefix: str, exclude: Tuple[str, ...] = ("ID",)) -> pd.DataFrame:
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in exclude}
    return df.rename(columns=rename)

#Read a CSV with sane defaults and short progress logs.
def read_csv(path: Path, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    info("Loading {path.name} ...")
    df = pd.read_csv(path, low_memory=False, usecols=usecols)
    ok(f"{path.name}: {df.shape[0]} rows x {df.shape[1]} cols")
    return df 

"""Aggregate player-level rows to one row per match (ID).
Numeric columns are aggregated with ``sum``, ``mean``, and ``std``. Column
names are flattened using the pattern ``{side_prefix}{feature}_{agg}``.
Also adds ``{side_prefix}player_count`` as the number of player rows per match.
"""
def aggregate_players(df: pd.DataFrame, side_prefix: str) -> pd.DataFrame:
    assert 'ID' in df.columns, "Player table must contain ID"

    # 1) Ne garder que les colonnes numériques + ID pour grouper
    numeric = df.select_dtypes(include=['number']).copy()
    numeric['ID'] = df['ID']

    # 2) Colonnes à agréger (toutes sauf ID)
    cols = [c for c in numeric.columns if c != 'ID']

    # 3) Agrégations multi-fonctions → MultiIndex sur les colonnes
    out = numeric.groupby('ID')[cols].agg(['sum', 'mean', 'std'])

    # 4) Aplatir le MultiIndex proprement
    #    to_flat_index() renvoie des tuples (col, agg)
    out.columns = [f"{side_prefix}{col}_{agg}" for col, agg in out.columns.to_flat_index()]

    # 5) Ajouter le nombre de lignes joueur par match
    counts = df.groupby('ID').size().rename(f"{side_prefix}player_count")
    out = out.join(counts)

    # 6) Reset index + petit log
    out = out.reset_index()
    ok(f"Aggregated players → {out.shape[0]} rows × {out.shape[1]} cols")
    return out


#Merge two tables on ``ID`` with a short size log.
def safe_merge(left: pd.DataFrame, right: pd.DataFrame, how: str = 'inner') -> pd.DataFrame:
    before = left.shape
    merged = left.merge(right, on='ID', how=how)
    ok(f"Merged {before} and {right.shape} --> {merged.shape}")
    return merged

def clean_unique_by_id(df: pd.DataFrame, id_col: str = 'ID') -> pd.DataFrame:
    """Ensure one and only one row per ``id_col`` and drop exact duplicates.

    Steps (with short logs):
    1) Drop exact duplicate rows (all columns identical)
    2) Drop duplicate IDs (keep first occurrence)
    3) Sort by ID and reset index for reproducibility

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe after merges/aggregations.
    id_col : str
        Name of the identifier column. Defaults to ``'ID'``.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with unique IDs.
    """
    assert id_col in df.columns, f"Missing id column: {id_col}"

    # 1) Drop exact duplicate rows
    n_before = len(df)
    dup_all = int(df.duplicated(keep='first').sum())
    if dup_all:
        info(f"Found {dup_all} exact duplicate rows → dropping…")
        df = df.drop_duplicates(keep='first')

    # 2) Drop duplicate IDs
    dups_mask = df.duplicated(subset=[id_col], keep='first')
    n_dup_ids = int(dups_mask.sum())
    if n_dup_ids:
        info(f"Found {n_dup_ids} duplicate {id_col}s → keeping first, dropping others…")
        df = df[~dups_mask].copy()

    # 3) Sort and reset index
    if id_col in df.columns:
        df = df.sort_values(id_col).reset_index(drop=True)

    n_after = len(df)
    removed = n_before - n_after
    ok(f"Cleaned by {id_col}: removed {removed} rows; IDs unique: {df[id_col].is_unique}")
    return df


#the heart of this file : 

"""Build a modeling table for one split (train or test).
Steps:
1) Load home/away **team** tables and apply prefixes
2) Merge team tables on ``ID`` (``inner`` by default or ``left`` if *lenient*)
3) Optionally load and aggregate home/away **player** tables, then merge
4) Reorder columns (``ID`` first), sort by ``ID`` for reproducibility
"""
def build_split(*,
    home_team_path: Path,
    away_team_path: Path,
    home_player_path: Optional[Path],
    away_player_path: Optional[Path],
    lenient: bool = False,
) -> pd.DataFrame:
    # 1) Teams 
    home_team = read_csv(home_team_path)
    away_team = read_csv(away_team_path)

    #Prefixes
    home_team = enforce_prefix(home_team, 'home_team')
    away_team = enforce_prefix(away_team, 'away_team')

    # 2) Merge team tables on ID
    how = 'left' if lenient else 'inner'
    teams = safe_merge(home_team, away_team, how=how)

    # 3) Players 
    if home_player_path and home_player_path.exists():
        home_player_raw = read_csv(home_player_path)
        home_player = aggregate_players(home_player_raw, 'home_player_')
        teams = safe_merge(teams, home_player, how=how)
    else: 
        info("no home player file provided, skipping.")
    
    if away_player_path and away_player_path.exists():
        away_player_raw = read_csv(away_player_path)
        away_player = aggregate_players(away_player_raw, 'away_player_')
        teams = safe_merge(teams, away_player, how=how)
    else: 
        info("no away player file provided, skipping.")

    #4) Final tidy ups 
    #Reorder: ID first
    cols = ['ID'] + [c for c in teams.columns if c != 'ID']
    teams = teams[cols]

    #sort by ID for reproductinility
    teams = teams.sort_values('ID').reset_index(drop=True)

    teams = clean_unique_by_id(teams, id_col='ID')

    ok("Split build completeeee")
    return teams


# Partie CLI pour lancer le script avec des arguments différents 

#function which define all the params
def parse_args() -> argparse.Namespace: 
    p = argparse.ArgumentParser(description="Merge raw football CSVs into modeling table")

    #Discovery roots
    p.add_argument('--train-home-team', type=Path, default=None)
    p.add_argument('--train-away-team', type=Path, default=None)
    p.add_argument('--train-home-player', type=Path, default=None)
    p.add_argument('--train-away-player', type=Path, default=None)

    p.add_argument('--test-home-team', type=Path, default=None)
    p.add_argument('--test-away-team', type=Path, default=None)
    p.add_argument('--test-home-player', type=Path, default=None)
    p.add_argument('--test-away-player', type=Path, default=None)

    p.add_argument('--train-dir', type=Path, default=None, help='Folder containing train CSVs')
    p.add_argument('--test-dir',  type=Path, default=None, help='Folder containing test CSVs')

    #Targets
    p.add_argument('--y-train', type=Path, default=None, help='Optional Y_train CSV (with ID + y_home_win,y_draw,y_away_win)')

    #Behavior
    p.add_argument('--lenient', action='store_true', help='Use LEFT joins instead of INNER (keep more rows)')

    #ouput
    p.add_argument('--out-dir', type=Path, default=Path('data/processed'))

    return p.parse_args()

"""Auto-discover standard file names inside *train_dir*/*test_dir*.
We look for CSVs containing tokens like "home_team", "away_player", etc.
Returns a dictionary whose keys match the argument names used later
(e.g. "train_home_team"). Missing entries are set to ``None``.
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

"""Load the one-hot targets file if provided and validate required columns.

Required columns: ``ID``, ``y_home_win``, ``y_draw``, ``y_away_win``.
Returns the reduced frame with exactly these columns.
"""
def load_y_train(y_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not y_path:
        info("No Y_train provided – skipping targets merge.")
        return None
    y = read_csv(y_path)
    need = {'ID', 'HOME_WINS', 'DRAW', 'AWAY_WINS'}
    missing = need - set(y.columns)
    if missing:
        raise ValueError(f"Y_train missing columns: {missing}")
    return y[['ID', 'HOME_WINS', 'DRAW', 'AWAY_WINS']]


#Write merged CSVs and a tiny schema.json for traceability.
def save_artifacts(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> None: 
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / 'train_merged.csv'
    test_path = out_dir / 'test_merged.csv'

    info("Saving CSVs ...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    ok(f"Wrote {train_path} ({train_df.shape[0]}×{train_df.shape[1]})")
    ok(f"Wrote {test_path} ({test_df.shape[0]}×{test_df.shape[1]})")

    #Schéma
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
    ok("Saved schema.json")

#Entry point: parse args → build train/test → optional targets → save.
def main() -> None:
    args = parse_args()
    
    info("Discovery inputs ...")
    discovered = discover_inputs(args.train_dir, args.test_dir)

    #If explicit files are passed, override discovery
    train_home_team = args.train_home_team or discovered['train_home_team']
    train_away_team = args.train_away_team or discovered['train_away_team']
    train_home_player = args.train_home_player or discovered['train_home_player']
    train_away_player = args.train_away_player or discovered['train_away_player']

    test_home_team = args.test_home_team or discovered['test_home_team']
    test_away_team = args.test_away_team or discovered['test_away_team']
    test_home_player = args.test_home_player or discovered['test_home_player']
    test_away_player = args.test_away_player or discovered['test_away_player']

    #Sanity check
    need_train = [train_home_team, train_away_team]
    need_test = [test_home_team, test_away_team]
    if any(p is None for p in need_train + need_test):
        raise SystemExit("Missing required team CSVs. pass --*-team or set --*-dir to a folder with those files.")

    info("Building TRAIN split ...")
    train = build_split(
        home_team_path=train_home_team,
        away_team_path=train_away_team, 
        home_player_path=train_home_player,
        away_player_path= train_away_player,
        lenient=args.lenient,
    )

    #Targets 
    y = load_y_train(args.y_train)
    if y is not None:
        train = safe_merge(train, y, how='inner')
        ok("Targets merged into train")
        train = clean_unique_by_id(train, id_col='ID')

    info("Building TEST split …")
    test = build_split(
        home_team_path=test_home_team,
        away_team_path=test_away_team,
        home_player_path=test_home_player,
        away_player_path=test_away_player,
        lenient=args.lenient,
    )

    save_artifacts(train, test, args.out_dir)
    ok("Tout est parfaittttttt")


if __name__ == '__main__':
    main()