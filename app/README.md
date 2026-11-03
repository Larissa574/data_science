# FoodPrice AI : application web (Streamlit)

Interface de consultation des prix alimentaires au Bénin et d'estimation de prix
à partir du modèle de machine learning.

## Lancer l'application

Depuis la racine du dépôt :

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

Prérequis :
- les fichiers `train.parquet` et `test.parquet` dans `data/` (partagés
  séparément, non versionnés) ;
- le modèle entraîné dans `artifacts/model.joblib` (livré par le pipeline
  d'entraînement). Sans lui, les pages Estimation et Batch CSV affichent un
  message d'erreur, mais le Dashboard reste utilisable.

## Structure

```
app/
├── Home.py              Page d'accueil : indicateurs clés et aperçu des données
├── pages/
│   ├── 1_Dashboard.py   Tableau de bord : prix, évolution, marchés, départements
│   ├── 2_Estimation.py  Estimation d'un prix pour un produit, un marché, une date
│   └── 3_Batch_CSV.py   Prédictions en lot à partir d'un fichier CSV
└── utils/
    ├── data.py          Chargement et préparation des données
    └── model.py         Chargement du modèle et fonction de prédiction
```

## Description des fichiers

### `utils/data.py`
- `load_data()` : lit tous les fichiers `.parquet` de `data/`, les fusionne,
  supprime les doublons et les lignes sans prix ni date, puis ajoute les
  colonnes `year` et `month`. Résultat mis en cache.
- `get_options(df)` : listes triées (produits, marchés, départements...) et
  bornes de dates, utilisées pour remplir les listes déroulantes.

### `utils/model.py`
Seul fichier qui dépend du modèle. Si le modèle change, c'est ici qu'on ajuste :
- `MODEL_PATH` : emplacement du modèle ;
- `FEATURES` : colonnes d'entrée, dans l'ordre attendu par le modèle ;
- `LOG_TARGET` : `True` si le modèle prédit `log1p(prix)` (la prédiction est
  alors reconvertie en FCFA).

Fonctions : `load_model()`, `add_time_features()`, `predict()`.

### `pages/1_Dashboard.py`
Filtres en cascade (produit, unité, type de prix, départements, période), puis
indicateurs (moyenne, médiane, nombre de marchés) et trois onglets : évolution
mensuelle, prix par marché, répartition par département.
Les prix ne sont comparables que pour une même unité et un même type de prix,
d'où la cascade.

### `pages/2_Estimation.py`
L'utilisateur choisit produit, unité, type de prix, marché et date. Le
département, les coordonnées et la catégorie sont déduits du marché. L'app
affiche le prix estimé, le dernier prix observé, la moyenne historique et un
graphique plaçant l'estimation sur l'historique.

### `pages/3_Batch_CSV.py`
Import d'un CSV avec les colonnes `date`, `market`, `commodity`, `unit`,
`pricetype` (un modèle est téléchargeable dans la page). Les lignes invalides
(date illisible, marché ou produit inconnu) sont rejetées avec leur motif, les
autres sont prédites. La colonne `known_combination` signale les combinaisons
marché/produit jamais observées. Le séparateur `,` ou `;` est détecté
automatiquement. Les résultats sont téléchargeables.

## Limites connues
- Pour une date postérieure aux dernières observations, les modèles à base
  d'arbres ne prolongent pas les tendances : l'estimation reflète plutôt le
  niveau récent des prix.
- Une combinaison marché/produit absente des données donne une estimation moins
  fiable (signalée dans la page Batch CSV).