# FoodPrice AI — Partie Machine Learning (Personne 2)

Prédire le prix (FCFA) d'un produit alimentaire sur un marché du Bénin à une date donnée.
Problème de **régression supervisée** ; cible = `price`.

## Modèle retenu

**LightGBM** (gradient boosting), dans un pipeline scikit-learn unique (prétraitement + modèle).

| Modèle | MAE (FCFA) | RMSE (FCFA) | R² | Taille |
|---|---|---|---|---|
| Baseline (médiane) | 410,7 | 1040,4 | -0,05 | n/a |
| Ridge | 205,7 | 561,2 | 0,695 | < 1 Mo |
| Random Forest | 75,4 | 219,5 | 0,953 | 50,8 Mo |
| **LightGBM** | **76,1** | **200,3** | **0,961** | **6,1 Mo** |

Comparaison sur la **validation figée** (18 306 lignes). Évaluation finale du seul modèle retenu sur le
**test réservé** (22 883 lignes, utilisé une seule fois, après le choix) : **MAE 78,5 FCFA, RMSE 217,9 FCFA,
R² 0,960**. Détails, validation croisée et contrôle temporel : `reports/model_comparison.md`.

## Jeux de données

| Fichier | Lignes | Rôle |
|---|---|---|
| `data/train_fit.parquet` | 73 223 | entraînement et validation croisée (80 % du train d'origine) |
| `data/validation.parquet` | 18 306 | **validation figée** : comparaison des modèles et acceptation d'un réentraînement |
| `data/test.parquet` | 22 883 | test final, jamais utilisé pour décider |

`train_fit` et `validation` sont créés une seule fois à partir de `train.parquet` (découpage aléatoire,
seed 42, `src.preprocessing.ensure_splits()`). **Ces deux fichiers doivent être partagés tels quels au groupe**
pour que tout le monde utilise exactement la même validation. Elle ne doit jamais changer.

**Pourquoi LightGBM ?** Meilleur RMSE et R², MAE équivalente au Random Forest, mais un fichier 10 fois
plus léger (6,1 Mo contre 50,8 Mo ; un Random Forest sans contrainte de taille atteint environ 140 Mo,
au-delà de la limite de 100 Mo de GitHub, et se charge lentement dans Streamlit), et un entraînement 2 fois
plus rapide (6 s contre 12 s), utile pour `retrain.py`.

## Limites à connaître (à reprendre dans le rapport)

- **Les découpages sont aléatoires** : des mois voisins d'une même série marché×produit
  sont présents des deux côtés, donc le score de 0,96 est optimiste.
- **Contrôle temporel** (entraînement avant 2025, test à partir de 2025) : R² ≈ 0,56 et MAE ≈ 210 FCFA
  pour LightGBM. Le modèle reste bien meilleur que la baseline (MAE 426) mais prédire l'avenir est
  nettement plus difficile. Environ 50 % de l'erreur absolue vient de 5 produits chers et volatils
  (piment rouge séché, crevettes, escargots, riz paddy, igname sèche). C'est une raison de plus pour
  réentraîner régulièrement avec les nouvelles données.
- Le modèle ne connaît que 51 marchés et 53 produits. Un marché ou produit inconnu ne provoque pas
  d'erreur, mais la prédiction est alors peu fiable.
- Les données se concentrent sur 2019-2026 (97 % des lignes).

## Variables

Entrées (10 colonnes brutes) : `date, admin1, admin2, market, category, commodity, unit, pricetype,
latitude, longitude`. Dans le pipeline, `date` devient année, mois, saisonnalité (sin/cos) et année décimale.

Exclues : `usdprice` (corrélation 0,997 avec la cible : fuite de données), `currency, priceflag,
esa_source, esa_processed` (constantes), `market_id, commodity_id` (redondantes).

Importance des variables (gain LightGBM) : produit 63 %, date 9 %, commune 7 %, catégorie 6 %.

## Installation et utilisation

```bash
pip install -r requirements.txt            # mêmes versions pour tout le groupe
# placer train.parquet et test.parquet dans data/ (train_fit et validation sont créés automatiquement)
python -m src.compare_models               # comparaison complète (~15 min) -> reports/, src/model_config.json
python -m src.compare_models --figures-only
python -m src.train                        # entraîne le modèle final -> models/ + reference.json
python -m pytest tests                     # tests
```

## Contrat d'intégration

### Personne 4 — `retrain.py`

```python
from src.preprocessing import TRAIN_PATH, load_data
from src.train import train_candidate, evaluate_model, save_model, read_metadata, MODEL_PATH
from src.schema import REQUIRED_COLUMNS, validate_dataframe   # validation des nouveaux lots

base = load_data(TRAIN_PATH)                      # data/train_fit.parquet (SANS la validation)
data = pd.concat([base, nouveaux_lots])           # lots d'abord validés avec validate_dataframe
bundle = train_candidate(data)                    # dict {"model", "X_valid", "y_valid"} ; validation = data/validation.parquet, figée
model, X_valid, y_valid = bundle["model"], bundle["X_valid"], bundle["y_valid"]
metrics = evaluate_model(model, X_valid, y_valid) # {"MAE", "RMSE", "R2"}
```

- `train_candidate(data, valid=None)` retourne un dictionnaire `{"model", "X_valid", "y_valid"}` (`fit_candidate` retourne le tuple `(pipeline, X_valid, y_valid)`). Données d'entraînement et de
  validation sont contrôlées avec le même schéma (`REQUIRED_COLUMNS`, qui inclut la cible `price`).
- **Ne jamais ajouter les lignes de `validation.parquet` ni de `test.parquet` aux données d'entraînement.**
  `test.parquet` ne sert à aucune décision de réentraînement.
- Comparer l'ancien et le nouveau modèle sur les mêmes lignes : `joblib.load(MODEL_PATH).predict(X_valid)` contre
  `model.predict(X_valid)`. Les métriques de l'ancien modèle sont aussi dans `models/model_current.json`.
- **Règle de remplacement convenue (proposition P4)** : le nouveau modèle est accepté si ni la MAE ni le RMSE
  ne se dégradent et si au moins l'un des deux s'améliore ; le R² est enregistré pour le suivi.
- `save_model(model, metrics, model_name, params, n_train)` archive l'ancienne version dans
  `models/archive/`, écrit le nouveau modèle de façon atomique et incrémente la version. À n'appeler
  que si la règle est satisfaite. Pour réutiliser le modèle et les paramètres actuels :
  `meta = read_metadata(); meta["model_name"], meta["params"]`.
- Un lot est valide si `validate_dataframe(df) == []`.

### Personne 3 — application Streamlit

```python
import joblib, os, streamlit as st
from src.reference import load_reference, make_input_row

@st.cache_resource
def get_model(mtime):                       # mtime change quand retrain.py remplace le fichier
    return joblib.load("models/model_current.joblib")
model = get_model(os.path.getmtime("models/model_current.joblib"))

ref = load_reference()                      # listes de marchés, produits, unités, dates
row = make_input_row(ref, market="Parakou", commodity="Rice (imported)", date="2026-03-15")
prix = model.predict(row)[0]                # prix en FCFA
```

- Entrée : DataFrame avec les 10 colonnes de `src.schema.INPUT_COLUMNS`. `make_input_row` complète
  département, commune, coordonnées, catégorie et unité à partir de `reference.json`.
- Prédiction par lot (CSV) : construire un DataFrame avec ces 10 colonnes puis `model.predict(df)`.
  Un CSV avec seulement marché + produit + date doit passer par `make_input_row` ligne par ligne.
- `models/model_current.json` : version, date d'entraînement, métriques, librairies (pour la page de
  suivi du modèle).
- `reports/` : `model_comparison.csv`, `temporal_check.csv`, `feature_importance.csv`, `figures/*.png`
  (pour la page de comparaison des modèles).

## Arborescence

```
src/schema.py          colonnes attendues + validation des données
src/preprocessing.py   nettoyage + feature engineering + prétraitement
src/models.py          construction des modèles (Ridge, Random Forest, LightGBM, baseline)
src/compare_models.py  protocole de comparaison -> reports/
src/train.py           train_candidate, evaluate_model, save_model, entraînement final
src/reference.py       table de référence pour l'application
src/model_config.json  modèle et hyperparamètres retenus (écrit par compare_models.py)
models/                model_current.joblib + .json, archive/
reference.json         marchés, produits, unités, plage de dates
reports/               tableaux et figures de la comparaison
tests/                 tests pytest
```
