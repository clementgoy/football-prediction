# Projet de parcours IA – Prédiction de résultats de matchs de football - Challenge Data

Ce projet s’inscrit dans le cadre d’un challenge de data science portant sur la **prédiction de l’issue de matchs de football** (victoire à domicile, match nul ou victoire à l’extérieur) à partir de statistiques agrégées au niveau des équipes et des joueurs.

L’objectif est de construire un pipeline complet de machine learning incluant :
- la préparation et la fusion des données,
- l’ingénierie de variables,
- l’entraînement et l’évaluation de différents modèles,
- la génération de fichiers de prédiction compatibles avec la plateforme du challenge.

## Architecture du projet

Fichiers principaux : 

├── Data/
│   ├── benchmark_and_extras  # Données brutes (train / test)
│   ├── processed/            # Données après fusion et préparation
│   ├── Test_Data             # Données brutes (train / test)
│   ├── Train_Data            # Données brutes (train / test)
|
├── models/
│   ├── processed/                # Fichiers csv des prédictions à soumettre au Challenge Data
│   ├── best.json                 # Référence du meilleur modèle/paramétrage retenu
│   │
│   ├── features.json           # Liste des features utilisées (par expérience)
│   ├── metrics.json            # Résultats/métriques associées (accuracy, etc.)
│   │
│   ├── random_forest.pkl       # Modèles RandomForest sauvegardés
│   ├── lgbm.pkl                # Modèles LightGBM sauvegardés
│   └── *.csv                   # Ex: importances de features / exports d’analyse
|
├── notebook/
│   ├── stats_infos_sur_data.ipynb  # Fichier d'analyse des données
│
├── src/
│   ├── merge_data.py       # Fusion des différentes tables
│   ├── build_dataset.py    # Construction du dataset final
│   ├── features.py         # Feature engineering
│   ├── train_*.py          # Entraînements des modèles
│   ├── predict_*.py        # Prédictions des matchs
│
├── requirements.txt
├── README.md
├── .gitignore
└── Makefile

## Prérequis

- Python
- pip
- numpy
- pandas
- scikit-learn
- lightgbm
- xgboost
- catboost
- mlflow
- optuna
- pyyaml
- joblib

## Installation

### 1️. Cloner le dépôt

```bash
git clone https://github.com/clementgoy/football-prediction.git
cd <nom-du-depot>
```

### 2️. Créer un environnement virtuel

Avec venv

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```


### 3️. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Données

Les données du challenge doivent être placées dans le dossier Data/.

Elles comprennent notamment :
	•	les statistiques des équipes à domicile et à l’extérieur,
	•	les statistiques des joueurs,
	•	les fichiers Y_train et Y_train_supp,
	•	les données de test.

## Préparation des données

Fusion et nettoyage des données

```bash
make merge
```

Cette étape :
	•	fusionne les tables équipes et joueurs,
	•	gère les valeurs manquantes,
	•	crée des variables dérivées (différences domicile / extérieur).


## Entraînement des modèles

Entraînements 

```bash
make train ##_<nom_de_la_methode>
```
ou 

```bash
python chemin/vers/nom_de_la_methode.py
```

## Tests et évaluation

Les performances sont évaluées à l’aide :
	•	d’un jeu de validation interne,
	•	d’une validation croisée stratifiée,
	•	de comparaisons entre différents hyperparamètres et modèles (Random Forest, Gradient Boosting, etc.).

Les résultats permettent de sélectionner le modèle le plus robuste avant soumission.


## Génération des prédictions

Prédictions 

```bash
make predict ##_<nom_de_la_methode>
```
ou 

```bash
python chemin/vers/nom_de_la_methode.py
```

Cette commande génère un fichier .csv conforme au format attendu par la plateforme du challenge


## Precision supplementaire pour les PCA à 2 dimensions

Pour visualiser la structure des données en 2D, exécuter pca_visualization.py. 
```bash
python src/pca_visualization.py
```
Le script charge data/processed/train_merged.csv, conserve les colonnes numériques, standardise les features, calcule une PCA à 2 composantes puis sauvegarde la figure dans outputs/pca_plot.png.

Sur ce graphe, on peut observer un “mur” de points (PCA_1 environ à -65 et alignement vertical). Cela peut correspondre à un sous-ensemble d’exemples très similaires sur certaines features dominantes (peut-être des patterns de valeurs manquantes).

Pour investiguer ce groupement, exécuter investigate_pca_wall.py. 
```bash
python src/investigate_pca_wall.py
```
Il produit un rapport dans outputs/report_outliers.json (statistiques de PCA_1, seuil d’outliers, comparaison outliers vs normal sur les features, distribution des classes, et colonnes suspectes).


## Precision supplementaire pour les PCA à 10 dimensions pour enrichir les données

Pour générer des features PCA_1 … PCA_10, exécuter add_pca_to_csv.py. 
```bash
python src/add_pca_to_csv.py
```
Cela crée data/processed/train_merged_pca.csv (et test_merged_pca.csv si le test existe) avec les 10 colones supplémentaires.

Ensuite, on peut effectuer l’entraînement normalement, RandomForest (fichier train_rf_pca.py) va utiliser ces CSV enrichis.


## Precisions sur les methodes de random forest mises en place dans les fichiers d'entrainement (nous sommes conscients que les noms des fichiers ne sont pas toujours très parlants)

- train_rdm_forest.py 
Dans ce fichier on execute cherche à identifier les hyperparametres et les poids associés à la différence de buts (y_supp) qui permettent d'avoir les modèles avec les meilleurs performances. On identifie donc 3 scénarios de poids : 
								- Chaque match a la même valeur dans l'entrainement
								- Poids linéaire = 1 + beta * | diff buts| (avec beta = 0,25 par défaut)
								- Poids exponentiel = max(exp(alpha * |diff buts|), cap) (avec alpha = 0,2 et cap =5 par défaut)
Ensuite pour chacun de ces scénarios on fait une GridSearch pour essayer de trouver les meilleurs hyperparamètres. 

- train_rdmf_upgrade_test.py
Dans ce fichier on entraine 3 modèles (RandomForestClassifier, ExtratreesClassifier et HistGradientBoostingClassifier) afin de comparer leurs performances (les deux premiers sont assez similaires mais le dernier est vraiment différent dans le fonctionnement donc c'est intéressant). Les hyperparamètres sont fixés dans le code.

- train_rf_pca.py
Dans ce fichier on entraine un modèle de rf sur un fichier de données enrichies avec 10 colones supplémentaires créées en appliquant des PCA sur les données d'entrainement.

- train_rf_noneWeights_best.py
C'est notre entrainement qui avait donné le modèle avec la meilleur accuracy dans challenge data mais depuis il a été battu. Tous les matchs ont le même poids dans l'entrainement. Cet entrainement crée un modèle de random forest.

- train_rf_optimized.py
On entraine dans ce fichier un premier modèle de random forest sur toutes les features des données d'entrainement. Cela va servir à identifier les features les plus pertinentes, on ne va garder que les fs_top_k premières (800 par défaut) et on entraine un nouveau modèle de random forest dessus. Cela vise à essayer de réduire le bruit et voir si une selection comme celle là peut augmenter les performances du modèle.


## Auteurs

Projet réalisé dans le cadre d’un challenge académique en intelligence artificielle par Clément GOY et Emma TREMLET