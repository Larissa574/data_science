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

Le fichier brut initial est conservé dans `data/raw/`. Le découpage préparé par le projet produit `data/train_fit.parquet` et `data/validation.parquet`. Le fichier de test reste réservé à l'évaluation finale et ne doit jamais être utilisé pour le réentraînement.

### Surveillance et réentraînement automatique

1. Déposer chaque nouveau lot Parquet dans `data/incoming/`.
2. Lancer une vérification ponctuelle :
   ```bash
   python src/retrain.py
   ```
3. Ou activer la surveillance continue (vérification toutes les 30 secondes) :
   ```bash
   python src/retrain.py --watch
   ```
4. Pour arrêter la surveillance : `Ctrl+C`.

Le script :
- combine `train_fit.parquet` avec les lots de `data/incoming/` ;
- nettoie les données, retire les doublons et calcule une empreinte du contenu ;
- ignore l'exécution si aucune observation nouvelle ou modifiée n'est détectée ;
- calcule le PSI (Population Stability Index) sur les variables explicatives et le prix, en comparant le nouveau lot à une référence fixe ;
- entraîne un candidat uniquement lorsque le jeu d'entraînement change ;
- compare le candidat et le modèle actif sur le même jeu de validation figé ;
- conserve le modèle actif si le candidat ne respecte pas la règle d'acceptation.

Les seuils PSI utilisés sont des repères de surveillance : inférieur à 0,10 = dérive faible ; de 0,10 à 0,20 = dérive modérée ; à partir de 0,20 = dérive importante. Le PSI signale un changement de distribution, mais ne prouve pas à lui seul une baisse de performance ni un concept drift.

Les états sont conservés dans `logs/retraining_state.json`, le profil initial dans `logs/data_drift_reference.json` et les exécutions dans `logs/retraining_history.csv`. Ces fichiers sont locaux et ne doivent pas être versionnés.

**Important :** le dossier `data/raw/` ne doit pas être passé directement au script. Le script refuse explicitement ce dossier afin d'éviter d'aspirer le shard de test. Les nouveaux lots doivent contenir les colonnes attendues par le schéma du modèle, notamment `price`, et représenter des observations étiquetées disponibles pour l'entraînement.

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
