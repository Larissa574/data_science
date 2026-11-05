# Comparaison des modèles — FoodPrice AI

Cible : `price` (FCFA). Entraînement : 73223 lignes. Validation figée : 18306 lignes.
Test final (réservé) : 22883 lignes.

## Comparaison sur la validation figée

| Modèle | MAE | RMSE | R2 | MAPE_% | CV MAE (moy ± écart-type) | fit_seconds | size_MB |
|---|---|---|---|---|---|---|---|
| Baseline (médiane) | 410.7 | 1040.4 | -0.0489 | 68.3 | — | 0 | 0.0 |
| Ridge | 205.7 | 561.2 | 0.6948 | 32.5 | 210.7 ± 4.1 | 0 | 0.0 |
| Random Forest | 75.4 | 219.5 | 0.9533 | 14.1 | 93.2 ± 1.9 | 12 | 50.8 |
| LightGBM (gradient boosting) | 76.1 | 200.3 | 0.9611 | 14.3 | 91.2 ± 3.0 | 6 | 6.1 |

MAE et RMSE en FCFA. MAPE = erreur relative moyenne (%). `fit_seconds` = temps d'entraînement
sur tout le jeu d'entraînement. `size_MB` = taille du modèle sauvegardé (joblib, compress=3).

## Modèle retenu

**LightGBM (gradient boosting)** — hyperparamètres : `{"subsample_freq": 1, "subsample": 0.8, "num_leaves": 127, "n_estimators": 1500, "min_child_samples": 5, "learning_rate": 0.1, "colsample_bytree": 0.8}`

Règle de choix : plus faible MAE en validation croisée (3 plis, sur l'entraînement) parmi les
modèles de moins de 100 Mo ; si plusieurs sont à moins de 5 % du meilleur MAE, le plus léger est retenu.

## Évaluation finale sur le test (utilisé une seule fois, après le choix)

| Modèle | MAE (FCFA) | RMSE (FCFA) | R² | MAPE (%) |
|---|---|---|---|---|
| LightGBM (gradient boosting) | 78.5 | 217.9 | 0.9600 | 14.2 |

## Contrôle temporel (entraînement avant 2025, test à partir de 2025)

| Modèle | MAE | RMSE | R2 | MAPE_% |
|---|---|---|---|---|
| Baseline (médiane) | 425.6 | 1037.0 | -0.0713 | 63.8 |
| Ridge | 247.6 | 629.2 | 0.6056 | 38.2 |
| Random Forest | 204.7 | 650.5 | 0.5784 | 29.1 |
| LightGBM (gradient boosting) | 209.7 | 665.9 | 0.5582 | 30.5 |

Ce test est plus proche d'un usage réel (prédire le futur). Les scores y sont plus bas que
sur le découpage aléatoire, qui est plus optimiste (des mois voisins d'une même série de
prix se retrouvent à la fois dans l'entraînement et la validation/test).
