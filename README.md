- Données brutes dans `data/` (non versionnées dans git).
- EDA & protos dans `notebooks/`.
- Code dans `src/`.
- Suivi d'expé: MLflow (par défaut local ./mlruns) + CSV `logs/experiments.csv`.

<strong>🧭 Vision d’ensemble du pipeline complet : </strong>
Étape	Script utilisé	Rôle
1. Fusion / Nettoyage brut:	merge_data.py	--> Construit train_merged.csv et test_merged.csv à partir des 4×2 CSV bruts
2. Construction X/y:	build_dataset.py + features.py	--> Nettoie, sélectionne les colonnes utiles, et prépare les matrices d’entraînement
3. Entraînement modèle:	train.py + models.py + utils.py	--> Entraîne un Gradient Boosting (ou autre)
4. Prédiction / Soumission:	predict.py	--> Utilise le modèle entraîné pour prédire sur test_merged.csv


<strong>Rôle central de merge_data.py</strong>

C’est le cœur du pipeline de préparation des données.
Il remplace et englobe plusieurs fonctions que tu aurais pu coder à la main avant.

🔁 En résumé, il fait :

Lecture automatique de tous les fichiers CSV bruts du challenge (home/away × team/player).

Nettoyage → suppression des doublons, harmonisation, tri par ID.

Agrégation → combine les données des joueurs (sum/mean/std par match).

Fusion complète → jointure des 4 tables sur ID.

Ajout des labels Y_train (si fournis).

Sauvegarde → écrit deux fichiers finaux :

    - train_merged.csv

    - test_merged.csv

    - un petit schema.json récapitulatif (noms, dimensions…).

