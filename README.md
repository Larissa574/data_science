# FoodPrice AI

## Projet intégrateur - Science des Données

**FoodPrice AI** est un projet de science des données consacré à l'analyse et à l'estimation des prix des produits alimentaires sur les marchés du Bénin.

L'objectif est de construire une chaîne complète : compréhension du problème, exploration des données, préparation, modélisation de régression, évaluation, application web et automatisation du réentraînement.

## Données

Le projet utilise le jeu de données **Benin - Food Prices | Africa (original)** (Programme alimentaire mondial). Les fichiers Parquet sont conservés localement et ne doivent pas être publiés dans ce dépôt.

Les champs du jeu de données comprennent notamment la date, le département, la commune, le marché, le produit, l'unité et le prix. La variable cible du projet est `price`.

**Précautions méthodologiques :**
- Un prix doit toujours être interprété avec son produit et son unité.
- Les prix de produits ou d'unités différents ne doivent pas être comparés sans précaution.
- `usdprice` ne doit pas être utilisé comme variable explicative si la cible est `price`, car il est dérivé du prix et créerait une fuite de cible.
- Les valeurs extrêmes doivent être examinées, pas supprimées automatiquement.
- L'évaluation doit privilégier une séparation temporelle : les observations anciennes pour l'entraînement et les plus récentes pour le test/validation.

Place les fichiers Parquet dans `data/raw/`. Si le dataset est réparti sur plusieurs fichiers, le script MLOps et l'application les lisent tous et les concatènent. Pour réentraîner avec un nouvel arrivage, fournis les fichiers historiques et le nouveau lot ensemble afin de conserver l'historique.

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
│   ├── Home.py               # Page d'accueil de l'application Streamlit
│   ├── pages/
│   │   ├── 1_Dashboard.py
│   │   ├── 2_Estimation.py
│   │   └── 3_Batch_CSV.py
│   └── utils/
│       ├── data.py           # Chargement des données
│       └── model.py          # Chargement du modèle et prédiction
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

## Application web (Streamlit)

Interface de consultation des prix alimentaires au Bénin et d'estimation de prix à partir du modèle de machine learning.

### Lancer l'application

Depuis la racine du dépôt :

```bash
streamlit run app/Home.py
```

L'application s'ouvre sur `http://localhost:8501`.

Prérequis :
- les fichiers Parquet dans `data/raw/` (partagés séparément, non versionnés) ;
- le modèle entraîné dans `models/model.joblib` (livré par le pipeline d'entraînement puis mis à jour par `retrain.py`). Sans lui, les pages Estimation et Batch CSV affichent un message d'erreur, mais le Dashboard reste utilisable.

### Description des fichiers

#### `app/utils/data.py`
- `load_data()` : lit tous les fichiers `.parquet` de `data/raw/`, les fusionne, supprime les doublons et les lignes sans prix ni date, puis ajoute les colonnes `year` et `month`. Résultat mis en cache : après l'ajout d'un nouveau fichier, vider le cache depuis le menu de Streamlit ("Clear cache").
- `get_options(df)` : listes triées (produits, marchés, départements...) et bornes de dates, utilisées pour remplir les listes déroulantes.

#### `app/utils/model.py`
Seul fichier de l'application qui dépend du modèle. Si le modèle change, c'est ici qu'on ajuste :
- `MODEL_PATH` : emplacement du modèle (`models/model.joblib`) ;
- `FEATURES` : colonnes d'entrée, dans l'ordre attendu par le modèle ;
- `LOG_TARGET` : `True` si le modèle prédit `log1p(prix)` (la prédiction est alors reconvertie en FCFA).

Fonctions : `load_model()`, `add_time_features()`, `predict()`.

#### `app/pages/1_Dashboard.py`
Filtres en cascade (produit, unité, type de prix, départements, période), puis indicateurs (moyenne, médiane, nombre de marchés) et trois onglets : évolution mensuelle, prix par marché, répartition par département. Les prix ne sont comparables que pour une même unité et un même type de prix, d'où la cascade.

#### `app/pages/2_Estimation.py`
L'utilisateur choisit produit, unité, type de prix, marché et date. Le département, la commune, les coordonnées et la catégorie sont déduits du marché. L'application affiche le prix estimé, le dernier prix observé, la moyenne historique et un graphique plaçant l'estimation sur l'historique.

#### `app/pages/3_Batch_CSV.py`
Import d'un CSV avec les colonnes `date`, `market`, `commodity`, `unit`, `pricetype` (un modèle de fichier est téléchargeable dans la page). Les lignes invalides (date illisible, marché ou produit inconnu) sont rejetées avec leur motif, les autres sont prédites. La colonne `known_combination` signale les combinaisons marché/produit jamais observées. Le séparateur `,` ou `;` est détecté automatiquement. Les résultats sont téléchargeables.

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

### Contrat d'intégration avec la personne 2 (entraînement)

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

### Contrat d'intégration avec la personne 3 (application)

- **Emplacement** : l'application charge un seul fichier, `models/model.joblib`. Il ne doit jamais être absent, même brièvement : le nouveau modèle doit être écrit dans un fichier temporaire du même dossier, puis placé avec `os.replace`, qui est atomique.
- **Contenu** : pipeline sklearn complet (prétraitement inclus), sérialisé avec Joblib.
- **Entrées** : un DataFrame pandas avec les colonnes brutes `admin1`, `admin2`, `market`, `latitude`, `longitude`, `category`, `commodity`, `unit`, `pricetype`, `year`, `month` (la liste exacte attendue est `FEATURES` dans `app/utils/model.py`). La cible est `price` en XOF. Les colonnes `usdprice`, `currency`, `priceflag`, `esa_source` et `esa_processed` ne sont pas des entrées.
- **Sortie** : un prix en XOF par ligne. Si le modèle prédit `log1p(price)`, mettre `LOG_TARGET = True` dans `app/utils/model.py`.
- **Changement de variables** : si la liste des colonnes d'entrée change, la personne 3 doit être prévenue pour mettre à jour `FEATURES` et les pages Estimation et Batch CSV.
- **Données** : l'application et `retrain.py` lisent tous deux `data/raw/`. Un nouveau lot doit conserver les mêmes colonnes que les fichiers d'origine.
- **Versions** : un modèle Joblib dépend de la version de scikit-learn utilisée pour l'entraîner. Les versions de `scikit-learn`, `pandas`, `joblib` et `streamlit` sont figées dans `requirements.txt`.

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

## Limites connues

- Pour une date postérieure aux dernières observations, les modèles à base d'arbres ne prolongent pas les tendances : l'estimation reflète plutôt le niveau récent des prix.
- Une combinaison marché/produit absente des données donne une estimation moins fiable (signalée dans la page Batch CSV).
- Ne pas utiliser les anciens éléments FinAccess AI pour entraîner ou présenter le projet FoodPrice AI.

## Équipe

- Personne 1 : analyse métier et exploration des données (EDA)
- Personne 2 : préparation des variables, entraînement et évaluation ML
- Personne 3 : application web et intégration des prédictions
- Personne 4 : automatisation, validation des données, suivi des modèles et intégration MLOps