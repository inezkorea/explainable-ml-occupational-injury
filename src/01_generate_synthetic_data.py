"""Generate a synthetic workers' compensation dataset.

The real data (KCOMWEL claims, 2007-2018) cannot be shared. This script creates
an artificial dataset with the same structure: one row per approved injury claim,
eight categorical/numeric features and a binary outcome (fatal vs. non-fatal).

Marginal distributions are loosely based on the published descriptive table
(Table 1 of the paper). The outcome is simulated from a known logistic model with
planted gender interactions, so we can check whether XGBoost + SHAP recovers them:

  * head / whole body / multiple-region injuries are strongly fatal (both genders)
  * care and personal-service jobs raise fatality risk for women only
  * amputation raises fatality risk for men only
  * trunk injuries raise fatality risk for women only
  * age and nationality have little or no effect

No row corresponds to a real person. Numbers produced from this file are
not the study's results.
"""
import numpy as np
import pandas as pd

from config import DATA, SEED

N = 67_576
rng = np.random.default_rng(SEED)


def draw(categories: dict, size: int) -> np.ndarray:
    keys = list(categories)
    p = np.array(list(categories.values()), dtype=float)
    return rng.choice(keys, size=size, p=p / p.sum())


# Gender-specific marginals (proportions, approximated from Table 1)
SPEC = {
    "Female": dict(
        age_group={"14-28": 10, "29-36": 13, "37-45": 19, "46-53": 30, "54+": 28},
        nationality={"Korean-Chinese": 62, "Chinese": 18, "Vietnamese": 7,
                     "Uzbekistani": 2, "Sri Lankan": 0.5, "Other": 10.5},
        industry={"Service": 52, "Manufacturing": 42, "Construction": 4,
                  "Pharmaceutical manufacturing": 0.8, "Other": 1.2},
        occupation={"Elementary": 46, "Care/personal service": 14,
                    "Cooking/food service": 16, "Office": 12,
                    "Machine operation": 7, "Technical": 5},
        body_part={"Hand/finger": 47, "Upper limb": 13, "Lower limb": 11,
                   "Foot/toe": 9, "Trunk": 6, "Head": 3, "Multiple regions": 1.3,
                   "Whole body": 0.3, "Other": 9.4},
        injury_type={"Fracture": 36, "Amputation": 12, "Burn": 12, "Crush": 9,
                     "Open wound": 9, "Soft tissue": 6, "Other": 16},
    ),
    "Male": dict(
        age_group={"14-28": 19, "29-36": 23, "37-45": 21, "46-53": 18, "54+": 19},
        nationality={"Korean-Chinese": 47, "Chinese": 13, "Vietnamese": 7,
                     "Uzbekistani": 4, "Sri Lankan": 4, "Other": 25},
        industry={"Manufacturing": 62, "Construction": 26, "Service": 8,
                  "Pharmaceutical manufacturing": 0.5, "Other": 3.5},
        occupation={"Elementary": 53, "Technical": 19, "Office": 14,
                    "Machine operation": 11, "Care/personal service": 1,
                    "Cooking/food service": 2},
        body_part={"Hand/finger": 48, "Lower limb": 11, "Foot/toe": 10,
                   "Upper limb": 8, "Trunk": 7, "Head": 4, "Multiple regions": 1.2,
                   "Whole body": 0.6, "Other": 10.2},
        injury_type={"Fracture": 42, "Amputation": 14, "Crush": 10,
                     "Open wound": 8, "Burn": 4, "Soft tissue": 6, "Other": 16},
    ),
}

n_male = rng.binomial(N, 0.849)
parts = []
for gender, n in (("Male", n_male), ("Female", N - n_male)):
    block = {"gender": np.repeat(gender, n)}
    for var, cats in SPEC[gender].items():
        block[var] = draw(cats, n)
    block["claim_year"] = rng.integers(2007, 2019, n)
    parts.append(pd.DataFrame(block))
df = pd.concat(parts, ignore_index=True)

# ---- Outcome model (true data-generating process) -----------------------
BODY = {"Head": 3.6, "Whole body": 4.6, "Multiple regions": 3.3, "Trunk": 0.4,
        "Other": 1.0, "Hand/finger": -2.0, "Upper limb": -1.6,
        "Lower limb": -1.3, "Foot/toe": -1.5}
INJ = {"Other": 1.2, "Soft tissue": 0.6, "Crush": 0.1, "Fracture": -0.4,
       "Open wound": -0.6, "Burn": 0.3, "Amputation": -0.8}
OCC = {"Technical": 0.4, "Machine operation": 0.2, "Elementary": 0.0,
       "Office": 0.0, "Cooking/food service": -0.3, "Care/personal service": 0.0}
IND = {"Construction": 0.5, "Other": 0.3, "Manufacturing": 0.0,
       "Service": -0.2, "Pharmaceutical manufacturing": 0.0}

male = (df.gender == "Male").to_numpy()
lp = (-6.7
      + df.body_part.map(BODY) + df.injury_type.map(INJ)
      + df.occupation.map(OCC) + df.industry.map(IND)
      + 1.35 * male                                           # men: higher baseline risk
      + 1.6 * (~male & (df.occupation == "Care/personal service"))
      + 1.2 * (~male & (df.industry == "Pharmaceutical manufacturing"))
      + 1.3 * (~male & (df.body_part == "Trunk"))
      + 1.5 * (male & (df.injury_type == "Amputation"))
      + 0.3 * (male & (df.nationality == "Vietnamese"))
      + 0.01 * (df.claim_year - 2012))
p = 1 / (1 + np.exp(-lp.to_numpy()))
df["fatal"] = rng.binomial(1, p)

DATA.parent.mkdir(parents=True, exist_ok=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df.insert(0, "id", np.arange(1, len(df) + 1))
df.to_csv(DATA, index=False)

print(f"Wrote {len(df):,} synthetic claims to {DATA.relative_to(DATA.parents[1])}")
print(df.groupby("gender").fatal.agg(n="size", deaths="sum", froi_pct="mean")
        .assign(froi_pct=lambda t: (100 * t.froi_pct).round(2)))
