Fichiers de données (non versionnés sur GitHub, voir .gitignore) :
- train.parquet et test.parquet : fichiers PAM / Hugging Face partagés dans le groupe.
- train_fit.parquet et validation.parquet : créés par src.preprocessing.ensure_splits() à partir de
  train.parquet. Ces deux fichiers doivent être partagés tels quels au groupe (validation FIGÉE).
- test.parquet : test final, jamais utilisé pour décider (ni pour choisir un modèle, ni pour accepter un réentraînement).
