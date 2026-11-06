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
