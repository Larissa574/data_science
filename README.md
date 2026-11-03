# FoodPrice AI

## Projet intégrateur - Science des Données

**FoodPrice AI** est un projet de science des données consacré à l'analyse et à l'estimation des prix des produits alimentaires sur les marchés du Bénin.

L'objectif est de construire une chaîne complète : compréhension du problème, exploration des données, préparation, modélisation de régression, évaluation, application web et automatisation du réentraînement.

## Données

Le projet utilise le jeu de données **Benin - Food Prices | Africa (original)**. Les fichiers Parquet sont conservés localement et ne doivent pas être publiés dans ce dépôt.

Les champs du jeu de données comprennent notamment la date, le département, la commune, le marché, le produit, l'unité et le prix. La variable cible du projet est `price`.

**Précautions méthodologiques :**
- Un prix doit toujours être interprété avec son produit et son unité.
- Les prix de produits ou d'unités différents ne doivent pas être comparés sans précaution.
- `usdprice` ne doit pas être utilisé comme variable explicative si la cible est `price`, car il est dérivé du prix et créerait une fuite de cible.
- Les valeurs extrêmes doivent être examinées, pas supprimées automatiquement.
- L'évaluation doit privilégier une séparation temporelle : les observations anciennes pour l'entraînement et les plus récentes pour le test/validation.

Place les fichiers Parquet dans `data/raw/`. Si le dataset est réparti sur plusieurs fichiers, le script MLOps les lit et les concatène. Pour réentraîner avec un nouvel arrivage, fournis les fichiers historiques et le nouveau lot ensemble afin de conserver l'historique.

## Structure du dépôt

```text
data_science/
├── data/
│   └── raw/                  # Parquet locaux, non versionnés
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py              # Pipeline ML de la personne 2
│   ├── predict.py
│   └── retrain.py            # Orchestration MLOps
├── app/
│   └── app.py
├── models/                   # Modèles locaux, non versionnés
├── logs/                     # Historique local des réentraînements
├── scripts/
│   └── retrain_windows.bat
├── tests/
│   └── test_retrain.py
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 ou supérieur est recommandé.

```bash
git clone https://github.com/Larissa574/data_science.git
cd data_science
python -m venv .venv
```

Sous Windows :

```powershell
.venv\Scripts\activate
```

Sous Linux/macOS :

```bash
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Partie MLOps : réentraînement contrôlé

Le script `src/retrain.py` ne remplace pas aveuglément le modèle actif.

Il réalise les étapes suivantes :

1. recherche et lecture des fichiers Parquet ;
2. vérification des colonnes obligatoires (`date`, `commodity`, `unit`, `price`) ;
3. conversion des dates et des prix, retrait des lignes dont les champs essentiels sont invalides et suppression des doublons exacts ;
4. appel au pipeline d'entraînement de la personne 2 ;
5. évaluation du candidat et du modèle actif sur **le même jeu de validation** ;
6. acceptation du candidat seulement si MAE et RMSE ne se dégradent pas et qu'au moins l'une des deux s'améliore ;
7. archivage local de l'ancien modèle avant promotion du candidat ;
8. ajout du résultat à `logs/retraining_history.csv`.

Les lignes avec prix non positif ou date/produit/unité invalide sont écartées du processus d'entraînement. Les valeurs extrêmes positives ne sont pas supprimées automatiquement.

### Contrat d'intégration avec la personne 2

Pour que le réentraînement fonctionne, le fichier `src/train.py` doit exposer cette fonction :

```python
def train_candidate(data: pd.DataFrame) -> dict:
    # Effectuer le split temporel et entraîner le modèle candidat.
    # X_valid et y_valid doivent représenter le même jeu de validation
    # pour le candidat et pour le modèle déjà en production.
    return {
        "model": fitted_pipeline,
        "X_valid": X_valid,
        "y_valid": y_valid,
    }
```

Le modèle retourné doit être un pipeline sklearn sérialisable avec Joblib et fournir `predict(X)`. Le prétraitement doit être inclus dans le pipeline afin que le modèle puisse traiter les mêmes variables lors de l'évaluation et de la prédiction.

**Important :** le script MLOps est prêt côté orchestration, mais il dépend de ce contrat. Le code d'entraînement initial du dépôt doit encore être remplacé/intégré par le pipeline de régression FoodPrice AI de la personne 2. Le script refusera de continuer si la fonction attendue n'existe pas.

### Lancer un réentraînement

Avec tous les fichiers Parquet dans `data/raw/` :

```bash
python src/retrain.py --data data/raw
```

Ou en indiquant explicitement plusieurs fichiers (historique + nouveau lot) :

```bash
python src/retrain.py --data data/historique.parquet data/nouveau_lot.parquet
```

Le modèle actif est enregistré localement dans `models/model.joblib`. Les anciennes versions acceptées sont archivées dans `models/archive/`. L'historique est conservé dans `logs/retraining_history.csv`.

Ces fichiers générés sont exclus de GitHub.

### Automatisation sous Windows

Lancer manuellement :

```powershell
scripts\retrain_windows.bat
```

Pour planifier l'exécution avec le Planificateur de tâches Windows :

1. créer une tâche et choisir la fréquence souhaitée ;
2. dans **Action**, sélectionner le programme `cmd.exe` ;
3. ajouter comme arguments : `/c "C:\chemin\vers\data_science\scripts\retrain_windows.bat"` ;
4. définir le dossier du projet comme répertoire de démarrage si nécessaire ;
5. vérifier que les fichiers Parquet actualisés sont présents avant l'heure d'exécution.

Un ordonnanceur ne crée pas de nouvelles données : il lance le script. Il faut donc qu'un nouveau lot soit disponible dans le dossier de données.

## Tests

```bash
pytest -q
```

Les tests vérifient notamment le contrôle des données et la règle de comparaison entre le modèle candidat et le modèle actif.

## Application et entraînement

Les modules `train.py`, `predict.py` et `app/app.py` doivent être alignés sur la régression FoodPrice AI lors de l'intégration des travaux des personnes 2 et 3. Ne pas utiliser les anciens éléments FinAccess AI pour entraîner ou présenter le projet FoodPrice AI.

## Équipe

- Personne 1 : analyse métier et exploration des données (EDA)
- Personne 2 : préparation des variables, entraînement et évaluation ML
- Personne 3 : application et intégration des prédictions
- Personne 4 : automatisation, validation des données, suivi des modèles et intégration MLOps
