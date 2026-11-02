# FinAccess AI

## Projet intégrateur - Science des Données

**FinAccess AI** est un projet de Machine Learning consacré à la prédiction de l'accès aux services bancaires à partir de caractéristiques socio-économiques et démographiques.

### Objectif

Construire une chaîne complète de Science des Données :

1. compréhension du problème métier ;
2. exploration et nettoyage des données ;
3. préparation des variables ;
4. entraînement et comparaison de modèles ;
5. déploiement sous forme d'application web ;
6. automatisation du réentraînement.

> La prédiction concerne l'appartenance à une classe de possession de compte bancaire. Elle ne constitue pas une décision d'éligibilité bancaire ou de crédit.

## Structure

```text
FinAccess-AI/
├── data/                  # Dataset local, non versionné
├── notebooks/             # Analyse exploratoire
├── src/
│   ├── preprocessing.py   # Nettoyage et prétraitement
│   ├── train.py           # Entraînement et comparaison des modèles
│   ├── predict.py         # Prédiction
│   └── retrain.py         # Réentraînement automatisé
├── app/
│   └── app.py             # Application Streamlit
├── models/                # Modèle entraîné, non versionné
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

Le projet utilise le dataset **Financial Inclusion in Africa** de Zindi.

Le dataset doit être téléchargé depuis la compétition officielle puis placé localement dans :

```text
data/financial_inclusion.csv
```

Les données de compétition ne sont pas poussées sur GitHub.

## Installation

```bash
git clone https://github.com/Larissa574/data_science.git
cd data_science
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Puis :

```bash
pip install -r requirements.txt
```

## Entraînement

Après avoir placé le dataset dans `data/financial_inclusion.csv` :

```bash
python src/train.py
```

Le meilleur modèle selon le F1-score est sauvegardé dans :

```text
models/model.joblib
```

## Application

```bash
streamlit run app/app.py
```

L'application permet une prédiction individuelle et une prédiction par fichier CSV.

## Réentraînement automatique

Le script :

```bash
python src/retrain.py
```

réutilise le pipeline d'entraînement. Il peut ensuite être programmé avec Windows Task Scheduler, cron ou un autre ordonnanceur.

## Modèles

Le pipeline compare actuellement :

- Régression Logistique : modèle de référence interprétable ;
- Random Forest : modèle non linéaire capable de capturer des interactions entre variables.

La sélection finale est basée principalement sur le F1-score, avec précision, rappel et ROC-AUC comme métriques complémentaires lorsque disponibles.

## Équipe

Projet réalisé dans le cadre du TP intégrateur de Science des Données.

- Personne 1 : Business / EDA
- Personne 2 : Machine Learning
- Personne 3 : Application
- Personne 4 : Intégration / MLOps / automatisation
