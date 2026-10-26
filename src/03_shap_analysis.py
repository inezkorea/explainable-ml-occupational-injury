"""Explain the XGBoost model with TreeSHAP, overall and stratified by gender.

Follows Sections 2.5 and 3.2 of the paper:
  * TreeSHAP (shap.TreeExplainer) on the final XGBoost model
  * summary (beeswarm) plots of the top 20 features for all, male and
    female workers; the gender-specific plots come from separate models
  * mean |SHAP| ranking per stratum, to compare feature importance by gender
  * SHAP interaction values to find feature pairs that act together
    (the paper used the DALEX / EIX R packages for this step)

Outputs: results/figures/shap_summary_*.png, results/tables/shap_*.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from config import CATEGORICAL, DATA, FIG, OUTCOME, SEED, TAB, XGB_PARAMS

FEATURES = {"all": CATEGORICAL,
            "male": [c for c in CATEGORICAL if c != "gender"],
            "female": [c for c in CATEGORICAL if c != "gender"]}


def fit_and_explain(df: pd.DataFrame, cols):
    X = pd.get_dummies(df[cols + ["claim_year"]], columns=cols, dtype=int)
    model = XGBClassifier(**XGB_PARAMS, random_state=SEED, n_jobs=-1)
    model.fit(X, df[OUTCOME])
    explainer = shap.TreeExplainer(model)
    return model, explainer, X, explainer.shap_values(X)


LABELS = {"gender": "gender", "age_group": "age", "nationality": "nationality",
          "industry": "industry", "occupation": "occupation",
          "body_part": "body part", "injury_type": "injury type"}


def pretty(col: str) -> str:
    """'body_part_Head' -> 'Body part: Head'."""
    for var, label in LABELS.items():
        if col.startswith(var + "_"):
            return f"{label.capitalize()}: {col[len(var) + 1:]}"
    return "Claim year" if col == "claim_year" else col


def main():
    df = pd.read_csv(DATA)
    strata = {"all": df,
              "male": df[df.gender == "Male"],
              "female": df[df.gender == "Female"]}

    rankings = {}
    for name, sub in strata.items():
        print(f"TreeSHAP: {name} workers (n={len(sub):,}, deaths={sub[OUTCOME].sum():,})")
        model, explainer, X, sv = fit_and_explain(sub, FEATURES[name])
        Xp = X.rename(columns=pretty)

        plt.figure()
        shap.summary_plot(sv, Xp, max_display=20, show=False, plot_size=(8, 8))
        plt.title(f"SHAP summary: {name} migrant workers (synthetic)", fontsize=11)
        plt.tight_layout(); plt.savefig(FIG / f"shap_summary_{name}.png", dpi=200)
        plt.close()

        # Mean |SHAP| and direction (mean SHAP among rows where the dummy = 1)
        rank = pd.DataFrame({
            "feature": Xp.columns,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
            "mean_shap_when_present": [sv[X.iloc[:, j] == 1, j].mean()
                                       if X.iloc[:, j].isin([0, 1]).all() and X.iloc[:, j].sum()
                                       else np.nan for j in range(X.shape[1])],
        }).sort_values("mean_abs_shap", ascending=False)
        rank["direction"] = np.select(
            [rank.mean_shap_when_present.isna(), rank.mean_shap_when_present > 0],
            ["continuous", "risk up"], "risk down")
        rank.to_csv(TAB / f"shap_importance_{name}.csv", index=False)
        rankings[name] = rank

        if name == "all":
            iv = explainer.shap_interaction_values(X)
            strength = np.abs(iv).mean(axis=0)
            np.fill_diagonal(strength, 0)
            i, j = np.triu_indices_from(strength, k=1)
            pairs = (pd.DataFrame({"feature_1": Xp.columns[i], "feature_2": Xp.columns[j],
                                   "mean_abs_interaction": strength[i, j]})
                       .sort_values("mean_abs_interaction", ascending=False).head(15))
            pairs.to_csv(TAB / "shap_top_interactions.csv", index=False)
            print("\nStrongest pairwise SHAP interactions (all workers):")
            print(pairs.round(4).to_string(index=False))

    # Side-by-side top 10 by gender, like Table 2 of the paper
    top = pd.concat({g: rankings[g].head(10)[["feature", "direction"]].reset_index(drop=True)
                     for g in ("female", "male")}, axis=1)
    top.index = top.index + 1
    top.to_csv(TAB / "shap_top10_by_gender.csv")
    print("\nTop 10 features by gender:")
    print(top.to_string())

    # Sanity check: does SHAP recover the gender interactions planted in
    # 01_generate_synthetic_data.py? Compare mean SHAP (log-odds) when present.
    planted = {"Occupation: Care/personal service": "female only",
               "Industry: Pharmaceutical manufacturing": "female only",
               "Body part: Trunk": "stronger in female",
               "Injury type: Amputation": "male only"}
    check = pd.DataFrame({
        "planted_effect": planted.values(),
        **{f"mean_shap_{g}": rankings[g].set_index("feature")
                                .mean_shap_when_present.reindex(list(planted))
           for g in ("female", "male")},
    })
    check.index.name = "feature"
    check.round(3).to_csv(TAB / "planted_effects_check.csv")
    print("\nPlanted gender interactions vs. SHAP (log-odds, mean when present):")
    print(check.round(3).to_string())


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    main()
