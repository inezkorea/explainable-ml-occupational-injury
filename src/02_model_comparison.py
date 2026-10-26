"""Compare logistic regression, random forest and XGBoost on a holdout set.

Follows Section 2.4 of the paper:
  * models: LR, RF, XGB
  * selection by AUROC; sensitivity breaks ties
  * LR built by backward stepwise selection on AIC, with claim year and
    industry forced out of the stepwise search (kept in the model)
  * XGB uses the final hyperparameters reported in the paper, with 10-fold CV
    on the training set

Outputs: results/tables/model_performance.csv, results/figures/roc_curves.png
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from config import (CATEGORICAL, CV_FOLDS, DATA, FIG, NUMERIC, OUTCOME, SEED,
                    TAB, TEST_SIZE, XGB_PARAMS)

warnings.filterwarnings("ignore")


def encode(df: pd.DataFrame, cols=CATEGORICAL) -> pd.DataFrame:
    """One-hot encode categorical features (all levels kept, for tree models)."""
    return pd.get_dummies(df[cols + NUMERIC], columns=cols, dtype=int)


def holdout_metrics(name, y, prob):
    """AUROC plus sensitivity/specificity at the Youden-optimal threshold."""
    fpr, tpr, thr = roc_curve(y, prob)
    k = np.argmax(tpr - fpr)
    return dict(model=name, auroc=roc_auc_score(y, prob),
                sensitivity=tpr[k], specificity=1 - fpr[k], threshold=thr[k])


def stepwise_logit(train: pd.DataFrame, candidates, forced):
    """Backward elimination on AIC over whole variables (all dummies of a factor
    leave together). Reference level = most frequent category."""
    def design(vars_):
        X = pd.get_dummies(train[vars_], columns=[v for v in vars_ if v in CATEGORICAL],
                           dtype=float)
        for v in vars_:                       # drop modal level as reference
            if v in CATEGORICAL:
                X = X.drop(columns=f"{v}_{train[v].mode()[0]}")
        return sm.add_constant(X)

    def fit(vars_):
        return sm.Logit(train[OUTCOME], design(vars_)).fit(disp=0, maxiter=200)

    current = list(candidates)
    best = fit(current + forced)
    improved = True
    while improved and current:
        improved = False
        trials = {v: fit([c for c in current if c != v] + forced) for v in current}
        v, m = min(trials.items(), key=lambda kv: kv[1].aic)
        if m.aic < best.aic:
            current.remove(v)
            best, improved = m, True
            print(f"  stepwise: dropped {v} (AIC {m.aic:,.1f})")
    return best, current + forced, design


def main():
    df = pd.read_csv(DATA)
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[OUTCOME],
                                   random_state=SEED)
    X_tr, X_te = encode(train), encode(test)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    y_tr, y_te = train[OUTCOME], test[OUTCOME]

    results, curves = [], {}

    # Logistic regression (stepwise AIC)
    print("Fitting logistic regression (backward stepwise AIC)...")
    candidates = [c for c in CATEGORICAL if c != "industry"]
    lr, kept, design = stepwise_logit(train, candidates, forced=["industry"] + NUMERIC)
    X_lr_te = design_for_test(test, train, kept)
    p = lr.predict(X_lr_te.reindex(columns=lr.params.index, fill_value=0))
    results.append(holdout_metrics("Logistic regression", y_te, p)); curves["LR"] = p
    lr.summary2().tables[1].to_csv(TAB / "logistic_regression_coefficients.csv")

    # Random forest
    print("Fitting random forest...")
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=-1,
                                random_state=SEED).fit(X_tr, y_tr)
    p = rf.predict_proba(X_te)[:, 1]
    results.append(holdout_metrics("Random forest", y_te, p)); curves["RF"] = p

    # XGBoost (paper's final hyperparameters)
    print("Fitting XGBoost with 10-fold CV on the training set...")
    xgb = XGBClassifier(**XGB_PARAMS, random_state=SEED, n_jobs=-1)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    cv_auc = cross_val_score(xgb, X_tr, y_tr, cv=cv, scoring="roc_auc")
    xgb.fit(X_tr, y_tr)
    p = xgb.predict_proba(X_te)[:, 1]
    row = holdout_metrics("XGBoost", y_te, p)
    row["cv_auroc_mean"], row["cv_auroc_sd"] = cv_auc.mean(), cv_auc.std()
    results.append(row); curves["XGB"] = p

    perf = pd.DataFrame(results).round(3)
    perf.to_csv(TAB / "model_performance.csv", index=False)
    print("\nHoldout performance (synthetic data):")
    print(perf.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, prob in curves.items():
        fpr, tpr, _ = roc_curve(y_te, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y_te, prob):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    ax.set(xlabel="1 - specificity", ylabel="Sensitivity",
           title="Holdout ROC curves (synthetic data)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "roc_curves.png", dpi=200)


def design_for_test(test, train, vars_):
    X = pd.get_dummies(test[vars_], columns=[v for v in vars_ if v in CATEGORICAL],
                       dtype=float)
    return sm.add_constant(X, has_constant="add")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    main()
