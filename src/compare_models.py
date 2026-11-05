"""Comparaison des modèles de régression FoodPrice AI (Personne 2).

Protocole d'évaluation (identique pour tous les modèles) :
  1. Trois jeux distincts, créés une fois (voir preprocessing.ensure_splits) :
       train_fit.parquet (80 % du train d'origine) -> entraînement et validation croisée
       validation.parquet (20 %, figé)             -> comparaison des modèles
       test.parquet                                -> évaluation finale du seul modèle retenu
  2. Réglage des hyperparamètres : validation croisée 3 plis (KFold mélangé) sur
     train_fit uniquement, critère principal = MAE (en FCFA).
  3. Chaque modèle réglé est réentraîné sur train_fit puis évalué sur la validation :
     MAE, RMSE, R² (+ MedAE, MAPE à titre informatif).
  4. Choix final : plus faible MAE en validation croisée parmi les modèles dont le
     fichier fait moins de 100 Mo ; si plusieurs sont à moins de 5 % du meilleur MAE,
     on retient le plus léger (déploiement et réentraînement plus simples).
  5. Le modèle retenu est évalué UNE fois sur test.parquet (aucune décision n'en dépend).
  6. Contrôle de robustesse temporel : entraînement avant 2025, test à partir de 2025
     (sur train_fit + validation ; le découpage fourni est aléatoire, donc optimiste).

Usage :  python -m src.compare_models [--quick | --figures-only]
"""
import argparse
import json
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error, mean_absolute_percentage_error,
    median_absolute_error, r2_score, root_mean_squared_error,
)
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler, cross_validate  # noqa: E402

from src.models import CONFIG_PATH, MODEL_LABELS, RANDOM_STATE, build_model  # noqa: E402
from src.preprocessing import (  # noqa: E402
    ROOT, TEST_PATH, TRAIN_PATH, VALID_PATH, clean_data, ensure_splits, load_data, split_xy,
)
from src.schema import TARGET  # noqa: E402

warnings.filterwarnings("ignore")
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MAX_SIZE_MB = 100          # limite GitHub pour un fichier
TOLERANCE = 0.05           # écart de MAE toléré pour préférer le modèle le plus léger
SCORING = {"mae": "neg_mean_absolute_error",
           "rmse": "neg_root_mean_squared_error", "r2": "r2"}

# Espaces de recherche d'hyperparamètres
SEARCH = {
    "ridge": ("grid", {"alpha": [0.1, 1, 10, 100], "log_target": [False, True]}),
    "random_forest": ("random", 5, {
        "n_estimators": [100], "max_features": [0.33, 0.5, 0.7],
        "min_samples_leaf": [2, 3, 5]}),
    "lightgbm": ("random", 8, {
        "n_estimators": [500, 1000, 1500], "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [63, 127, 255], "min_child_samples": [3, 5, 10, 20],
        "colsample_bytree": [0.6, 0.8, 1.0], "subsample": [0.8], "subsample_freq": [1]}),
}


def log(msg):
    print(msg, flush=True)


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "MAPE_%": 100 * mean_absolute_percentage_error(y_true, y_pred),
    }


def model_size_mb(model) -> float:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.joblib"
        joblib.dump(model, path, compress=3)
        return path.stat().st_size / 1e6


def tune(name, X, y, quick=False) -> tuple:
    """Recherche par validation croisée. Retourne (meilleurs params, tableau des essais)."""
    spec = SEARCH[name]
    if spec[0] == "grid":
        candidates = list(ParameterGrid(spec[1]))
    else:
        n_iter = 2 if quick else spec[1]
        candidates = list(ParameterSampler(spec[2], n_iter=n_iter, random_state=RANDOM_STATE))
    if quick:
        candidates = candidates[:2]
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for i, params in enumerate(candidates, 1):
        t = time.time()
        res = cross_validate(build_model(name, params), X, y, cv=cv, scoring=SCORING,
                             n_jobs=1, error_score="raise")
        row = {"model": name, "params": json.dumps(params),
               "cv_MAE": -res["test_mae"].mean(), "cv_MAE_std": res["test_mae"].std(),
               "cv_RMSE": -res["test_rmse"].mean(), "cv_R2": res["test_r2"].mean(),
               "seconds": time.time() - t}
        rows.append(row)
        log(f"  [{name} {i}/{len(candidates)}] CV MAE={row['cv_MAE']:.2f} "
            f"RMSE={row['cv_RMSE']:.1f} R2={row['cv_R2']:.4f} ({row['seconds']:.0f}s) {params}")
    table = pd.DataFrame(rows).sort_values("cv_MAE").reset_index(drop=True)
    return json.loads(table.loc[0, "params"]), table


def group_importances(model) -> pd.Series:
    """Importance des variables regroupées par colonne d'origine."""
    est = model.named_steps["model"]
    names = model.named_steps["prep"].get_feature_names_out()
    if hasattr(est, "booster_"):          # LightGBM : importance en « gain » (moins biaisée)
        imp = est.booster_.feature_importance(importance_type="gain")
    else:
        imp = getattr(est, "feature_importances_", None)
    if imp is None:
        return pd.Series(dtype=float)
    groups = []
    for n in names:
        if n.startswith("cat__"):
            groups.append(n[len("cat__"):].split("_")[0])
        elif n.startswith("date__"):
            groups.append("date (année, mois)")
        else:
            groups.append(n.split("__")[1])
    s = pd.Series(imp, index=groups).groupby(level=0).sum()
    return (s / s.sum()).sort_values(ascending=False)


def make_figures(results: pd.DataFrame, final_name, final_model, X_te, y_te, te_df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [MODEL_LABELS[m].split(" (")[0] for m in results["model"]]

    # 1. Comparaison des modèles
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(axes, ["MAE", "RMSE", "R2"],
                              ["MAE (FCFA) — plus bas = mieux", "RMSE (FCFA) — plus bas = mieux",
                               "R² — plus haut = mieux"]):
        bars = ax.bar(labels, results[col], color=["#9aa5b1", "#5b8def", "#f0a04b", "#3aa675"][:len(labels)])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, results[col]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}" if col == "R2" else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Comparaison des modèles sur la validation figée", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=130)
    plt.close(fig)

    pred = final_model.predict(X_te)

    # 2. Prédit vs réel
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_te, pred, s=3, alpha=0.25)
    lim = [max(1, min(y_te.min(), pred.min())), max(y_te.max(), pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Prix réel (FCFA, échelle log)"); ax.set_ylabel("Prix prédit (FCFA, échelle log)")
    ax.set_title(f"Prédit vs réel — {MODEL_LABELS[final_name].split(' (')[0]}")
    fig.tight_layout()
    fig.savefig(FIGURES / "predicted_vs_actual.png", dpi=130)
    plt.close(fig)

    # 3. Erreur par sous-groupe (catégorie de produit, département)
    err = te_df.assign(abs_err=np.abs(y_te.values - pred))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col in zip(axes, ["category", "admin1"]):
        g = err.groupby(col)["abs_err"].mean().sort_values()
        ax.barh(g.index, g.values, color="#5b8def")
        ax.set_title(f"MAE par {'catégorie de produit' if col == 'category' else 'département'} (FCFA)", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "error_by_group.png", dpi=130)
    plt.close(fig)

    # 4. Importance des variables
    imp = group_importances(final_model)
    if len(imp):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        imp.sort_values().plot.barh(ax=ax, color="#3aa675")
        ax.set_title("Importance des variables (regroupées par colonne d'origine)", fontsize=10)
        ax.set_xlabel("Part de l'importance totale")
        fig.tight_layout()
        fig.savefig(FIGURES / "feature_importance.png", dpi=130)
        plt.close(fig)
        imp.rename("importance").to_csv(REPORTS / "feature_importance.csv")


def to_markdown(df: pd.DataFrame, fmt: dict) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [fmt.get(c, str)(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figures_only():
    """Régénère les figures à partir de reports/model_comparison.csv et model_config.json."""
    ensure_splits()
    tr = clean_data(load_data(TRAIN_PATH))
    va = clean_data(load_data(VALID_PATH))
    X_tr, y_tr = split_xy(tr)
    X_va, y_va = split_xy(va)
    results = pd.read_csv(REPORTS / "model_comparison.csv")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = build_model(cfg["model_name"], cfg["params"]).fit(X_tr, y_tr)
    make_figures(results, cfg["model_name"], model, X_va, y_va, va)
    log("Figures régénérées dans reports/figures/")


def main(quick: bool = False):
    REPORTS.mkdir(exist_ok=True)
    ensure_splits()
    tr = clean_data(load_data(TRAIN_PATH))
    va = clean_data(load_data(VALID_PATH))
    te = clean_data(load_data(TEST_PATH))
    X_tr, y_tr = split_xy(tr)
    X_va, y_va = split_xy(va)
    log(f"Entraînement : {len(tr)} | Validation : {len(va)} | Test (réservé) : {len(te)}")

    # ---- 1. Réglage par validation croisée (sur train_fit uniquement) --------
    best_params, cv_tables = {}, []
    for name in ["ridge", "random_forest", "lightgbm"]:
        log(f"\n== Réglage : {MODEL_LABELS[name]} ==")
        best_params[name], table = tune(name, X_tr, y_tr, quick)
        cv_tables.append(table)
    cv_all = pd.concat(cv_tables, ignore_index=True)
    cv_all.to_csv(REPORTS / "cv_search.csv", index=False)

    # ---- 2. Comparaison sur la validation figée -------------------------------
    log("\n== Évaluation sur la validation figée ==")
    rows, fitted = [], {}
    order = ["baseline", "ridge", "random_forest", "lightgbm"]
    for name in order:
        params = best_params.get(name, {})
        model = build_model(name, params)
        t = time.time()
        model.fit(X_tr, y_tr)
        fit_s = time.time() - t
        m = metrics(y_va, model.predict(X_va))
        cv_row = cv_all[cv_all.model == name]
        row = {"model": name, **m,
               "cv_MAE": cv_row.cv_MAE.iloc[0] if len(cv_row) else np.nan,
               "cv_MAE_std": cv_row.cv_MAE_std.iloc[0] if len(cv_row) else np.nan,
               "fit_seconds": fit_s,
               "size_MB": model_size_mb(model) if name != "baseline" else 0.0,
               "params": json.dumps(params)}
        rows.append(row)
        fitted[name] = model
        log(f"  {MODEL_LABELS[name]:32s} MAE={m['MAE']:7.2f} RMSE={m['RMSE']:8.2f} "
            f"R2={m['R2']:.4f} MAPE={m['MAPE_%']:.1f}% fit={fit_s:.0f}s size={row['size_MB']:.1f}MB")
    results = pd.DataFrame(rows)
    results.to_csv(REPORTS / "model_comparison.csv", index=False)

    # ---- 3. Choix du modèle final (sans le test) -------------------------------
    cand = results[(results.model != "baseline") & (results.size_MB <= MAX_SIZE_MB)]
    best_cv = cand.cv_MAE.min()
    close = cand[cand.cv_MAE <= best_cv * (1 + TOLERANCE)]
    final_name = close.sort_values("size_MB").iloc[0]["model"]
    final_params = best_params[final_name]
    log(f"\nModèle retenu : {MODEL_LABELS[final_name]} | params = {final_params}")
    CONFIG_PATH.write_text(json.dumps(
        {"model_name": final_name, "params": final_params}, indent=2), encoding="utf-8")

    make_figures(results, final_name, fitted[final_name], X_va, y_va, va)

    # ---- 4. Évaluation finale UNIQUE sur le test --------------------------------
    X_te, y_te = split_xy(te)
    final_test = metrics(y_te, fitted[final_name].predict(X_te))
    pd.DataFrame([{"model": final_name, **final_test}]).to_csv(REPORTS / "final_test.csv", index=False)
    log(f"Test final ({MODEL_LABELS[final_name]}) : MAE={final_test['MAE']:.2f} "
        f"RMSE={final_test['RMSE']:.2f} R2={final_test['R2']:.4f}")

    # ---- 5. Contrôle temporel ------------------------------------------------
    log("\n== Contrôle de robustesse temporel (train < 2025 | test >= 2025) ==")
    full = pd.concat([tr, va], ignore_index=True)     # le test reste intact
    cutoff = pd.Timestamp("2025-01-01")
    past, future = full[full.date < cutoff], full[full.date >= cutoff]
    Xp, yp = split_xy(past)
    Xf, yf = split_xy(future)
    trows = []
    for name in order:
        model = build_model(name, best_params.get(name, {})).fit(Xp, yp)
        m = metrics(yf, model.predict(Xf))
        trows.append({"model": name, "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                      "MAPE_%": m["MAPE_%"], "n_train": len(past), "n_test": len(future)})
        log(f"  {MODEL_LABELS[name]:32s} MAE={m['MAE']:7.2f} RMSE={m['RMSE']:8.2f} R2={m['R2']:.4f}")
    temporal = pd.DataFrame(trows)
    temporal.to_csv(REPORTS / "temporal_check.csv", index=False)

    # ---- 6. Rapport markdown --------------------------------------------------
    f0 = lambda v: f"{v:.0f}"      # noqa: E731
    f1 = lambda v: f"{v:.1f}"      # noqa: E731
    f4 = lambda v: f"{v:.4f}"      # noqa: E731
    show = results.copy()
    show["Modèle"] = show["model"].map(MODEL_LABELS)
    show["CV MAE (moy ± écart-type)"] = [
        "—" if np.isnan(a) else f"{a:.1f} ± {b:.1f}" for a, b in zip(show.cv_MAE, show.cv_MAE_std)]
    show = show[["Modèle", "MAE", "RMSE", "R2", "MAPE_%", "CV MAE (moy ± écart-type)",
                 "fit_seconds", "size_MB"]]
    tshow = temporal.copy()
    tshow["Modèle"] = tshow["model"].map(MODEL_LABELS)
    tshow = tshow[["Modèle", "MAE", "RMSE", "R2", "MAPE_%"]]
    md = f"""# Comparaison des modèles — FoodPrice AI

Cible : `price` (FCFA). Entraînement : {len(tr)} lignes. Validation figée : {len(va)} lignes.
Test final (réservé) : {len(te)} lignes.

## Comparaison sur la validation figée

{to_markdown(show, {"MAE": f1, "RMSE": f1, "R2": f4, "MAPE_%": f1, "fit_seconds": f0, "size_MB": f1})}

MAE et RMSE en FCFA. MAPE = erreur relative moyenne (%). `fit_seconds` = temps d'entraînement
sur tout le jeu d'entraînement. `size_MB` = taille du modèle sauvegardé (joblib, compress=3).

## Modèle retenu

**{MODEL_LABELS[final_name]}** — hyperparamètres : `{json.dumps(final_params)}`

Règle de choix : plus faible MAE en validation croisée (3 plis, sur l'entraînement) parmi les
modèles de moins de {MAX_SIZE_MB} Mo ; si plusieurs sont à moins de {int(TOLERANCE * 100)} % du meilleur MAE, le plus léger est retenu.

## Évaluation finale sur le test (utilisé une seule fois, après le choix)

| Modèle | MAE (FCFA) | RMSE (FCFA) | R² | MAPE (%) |
|---|---|---|---|---|
| {MODEL_LABELS[final_name]} | {final_test['MAE']:.1f} | {final_test['RMSE']:.1f} | {final_test['R2']:.4f} | {final_test['MAPE_%']:.1f} |

## Contrôle temporel (entraînement avant 2025, test à partir de 2025)

{to_markdown(tshow, {"MAE": f1, "RMSE": f1, "R2": f4, "MAPE_%": f1})}

Ce test est plus proche d'un usage réel (prédire le futur). Les scores y sont plus bas que
sur le découpage aléatoire, qui est plus optimiste (des mois voisins d'une même série de
prix se retrouvent à la fois dans l'entraînement et la validation/test).
"""
    (REPORTS / "model_comparison.md").write_text(md, encoding="utf-8")
    log("\nTerminé. Rapports dans reports/ ; configuration finale dans src/model_config.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="test rapide (2 essais par modèle)")
    ap.add_argument("--figures-only", action="store_true", help="régénère seulement les figures")
    args = ap.parse_args()
    figures_only() if args.figures_only else main(args.quick)
