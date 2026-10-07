# -*- coding: utf-8 -*-
"""Step 0 for the comparator arms: clean the cohort, impute, and emit the shared inputs.

Reads the cohort workbook and produces the files that all downstream comparator
arms consume:

    data/clinical_full.csv    winsorised + MICE-imputed analysis table
    data/llm_input_full.csv   same variables, original missingness kept
                              (used to build the LLM case cards)
    data/meta.json            variable groupings (COMP / LAB / TREAT / ...)
    data/baseline_table.csv   Table 1 style summary
    data/hy4_sample_300.csv   stratified 300-case subsample (LLM input side)
    data/outcome_key_300.csv  outcomes for that subsample, held separately so the
                              LLM arms stay blinded until scoring

Endpoint: in-hospital mortality (`outcome`).

Note on leakage: every identifier and every post-admission time-to-event column
is dropped before modelling (see LEAK below).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# --- Path resolution: overridable via env vars, defaults to repo data/ ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.environ.get("ALF_DATA_DIR", os.path.join(_ROOT, "data"))
COHORT_XLSX = os.environ.get(
    "ALF_COHORT_XLSX",
    os.path.join(_ROOT, "cohort", "alf_icu_final_first_stay.xlsx"),
)
os.makedirs(DATA, exist_ok=True)

from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

OUT = DATA
XLSX = COHORT_XLSX
SEED = 2024
np.random.seed(SEED)

if not os.path.exists(XLSX):
    sys.exit(f"Missing cohort workbook: {XLSX}\n"
             "  Set ALF_COHORT_XLSX or place the file under cohort/ "
             "(see cohort/README.md).")

df = pd.read_excel(XLSX, sheet_name="data")

# ------------------------------------------------ 1. drop leakage / identifiers
LEAK = [
    "subject_id", "hadm_id", "stay_id", "hadm_seq", "icustay_seq", "first_icu_stay",
    "days_to_death_icu", "days_to_death_hosp", "los_icu", "los_hospital",
    "icu_intime", "icu_outtime", "hospital_expire_flag",
]
df = df.drop(columns=[c for c in LEAK if c in df.columns])
df["case_id"] = np.arange(1, len(df) + 1)
if "gender" in df.columns:
    df["gender_male"] = (df["gender"] == "M").astype(int)


# ------------------------------------------------ 2. mutually exclusive aetiology
def cohort_of(r):
    if r["Alcoholic_plus_viral"] == 1:
        return "Alcoholic_plus_viral"
    if r["Alcoholic_only"] == 1:
        return "Alcoholic_only"
    if r["Viral_only"] == 1:
        return "Viral_only"
    return "Other"


df["cohort"] = df.apply(cohort_of, axis=1)

# ------------------------------------------------ 3. variable groups
DEMO = ["Age", "gender_male"]
ETIO = ["Alcoholic_plus_viral", "Alcoholic_only", "Viral_only", "Other"]
COMP = ["Ascites", "Sepsis", "HE", "HRS", "EVB", "SBP", "Shock", "Pneumonia"]
LAB = ["Hemoglobin", "Platelet", "WBC", "Albumin", "BUN", "Chloride", "Creatinine",
       "Sodium", "ALT", "AST", "Total_bilirubin", "PT", "APTT", "INR", "Lactate"]
VITAL = ["SpO2", "MAP"]
SCORE = ["MELD", "SOFA"]
TREAT = ["Liver_transplantation", "Vasopressin", "rrt"]
CONT = LAB + VITAL + SCORE + ["Age"]
ALLPRED = DEMO + ETIO + COMP + LAB + VITAL + SCORE + TREAT
BIN = ETIO + COMP + TREAT + ["gender_male"]

missing = [c for c in ALLPRED + ["outcome"] if c not in df.columns]
if missing:
    sys.exit(f"Cohort workbook is missing required columns: {missing}")

# ------------------------- 4. winsorise (1st/99th pct) then MICE (clinical table)
X = df[ALLPRED].copy()
miss_rate = X.isna().mean()
print("=== Variables with missingness > 0 (%) ===")
print(miss_rate[miss_rate > 0].mul(100).round(1).to_string())
print()

Xw = X.copy()
for c in CONT:
    lo, hi = X[c].quantile([0.01, 0.99])
    Xw[c] = X[c].clip(lo, hi)

imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                       random_state=SEED, sample_posterior=False)
Ximp = pd.DataFrame(imp.fit_transform(Xw), columns=ALLPRED, index=df.index)
for c in BIN:
    Ximp[c] = Ximp[c].round().clip(0, 1)   # keep binary variables at 0/1

clinical = pd.concat(
    [df[["case_id", "cohort", "outcome"]].reset_index(drop=True),
     Ximp[ALLPRED].reset_index(drop=True)],
    axis=1,
)
clinical.to_csv(os.path.join(OUT, "clinical_full.csv"), index=False)

# ------------------------------------------------ 5. LLM input (missingness kept)
llm = df[["case_id", "cohort"] + ALLPRED].copy()
llm.to_csv(os.path.join(OUT, "llm_input_full.csv"), index=False)

# ------------------------- 6. stratified 300-case subsample (aetiology x outcome)
N_TARGET = 300
cohorts = ["Other", "Alcoholic_only", "Viral_only", "Alcoholic_plus_viral"]
n_death_t = int(round(N_TARGET * df["outcome"].mean()))
n_surv_t = N_TARGET - n_death_t


def largest_remainder(pop, total):
    """Apportion `total` seats across `pop` using the largest-remainder method."""
    s = sum(pop)
    raw = [total * p / s for p in pop]
    base = [int(np.floor(x)) for x in raw]
    rem = np.array(raw) - np.array(base)
    for i in np.argsort(rem)[::-1][: total - sum(base)]:
        base[i] += 1
    return base


d_pop = [int(((df["cohort"] == c) & (df["outcome"] == 1)).sum()) for c in cohorts]
s_pop = [int(((df["cohort"] == c) & (df["outcome"] == 0)).sum()) for c in cohorts]
d_alloc = largest_remainder(d_pop, n_death_t)
s_alloc = largest_remainder(s_pop, n_surv_t)

print("=== Subsample allocation (aetiology x outcome) ===")
parts = []
for i, c in enumerate(cohorts):
    sub_d = df[(df["cohort"] == c) & (df["outcome"] == 1)]
    sub_s = df[(df["cohort"] == c) & (df["outcome"] == 0)]
    print(f"{c}: n={int((df['cohort'] == c).sum())} "
          f"(event rate {df.loc[df['cohort'] == c, 'outcome'].mean():.3f}) "
          f"-> sampled {d_alloc[i] + s_alloc[i]} "
          f"(deaths {d_alloc[i]} / survivors {s_alloc[i]})")
    if d_alloc[i]:
        parts.append(sub_d.sample(n=d_alloc[i], random_state=SEED))
    if s_alloc[i]:
        parts.append(sub_s.sample(n=s_alloc[i], random_state=SEED))
print()

samp = pd.concat(parts).sort_values("case_id").reset_index(drop=True)
alloc = {c: d_alloc[i] + s_alloc[i] for i, c in enumerate(cohorts)}

# LLM side: no outcome column, so the LLM arms remain blinded while scoring
samp[["case_id", "cohort"] + ALLPRED].to_csv(
    os.path.join(OUT, "hy4_sample_300.csv"), index=False)
samp[["case_id", "cohort", "outcome"]].to_csv(
    os.path.join(OUT, "outcome_key_300.csv"), index=False)

print(f"Subsample event rate: {samp['outcome'].mean():.3f} "
      f"(full cohort {df['outcome'].mean():.3f}), n={len(samp)}")
print()

# ------------------------------------------------ 7. baseline table
def med_iqr(s):
    s = s.dropna()
    return f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"


rows = []
for c in ALLPRED:
    v0 = df.loc[df["outcome"] == 0, c].dropna()
    v1 = df.loc[df["outcome"] == 1, c].dropna()
    if c in BIN:
        rows.append([c, f"{int(df[c].sum())} ({df[c].mean() * 100:.1f}%)",
                     f"{int(v0.sum())} ({v0.mean() * 100:.1f}%)",
                     f"{int(v1.sum())} ({v1.mean() * 100:.1f}%)"])
    else:
        rows.append([c, med_iqr(df[c]), med_iqr(v0), med_iqr(v1)])

bt = pd.DataFrame(rows, columns=["variable", "Overall", "Survivor", "Non-survivor"])
bt.to_csv(os.path.join(OUT, "baseline_table.csv"), index=False)
print("=== Baseline table ===")
print(bt.to_string(index=False))
print()

meta = {"ALLPRED": ALLPRED, "BIN": BIN, "CONT": CONT, "DEMO": DEMO, "ETIO": ETIO,
        "COMP": COMP, "LAB": LAB, "VITAL": VITAL, "SCORE": SCORE, "TREAT": TREAT,
        "cohorts": cohorts, "alloc": alloc, "seed": SEED}
with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as fh:
    json.dump(meta, fh, ensure_ascii=False, indent=1)

print("Outputs written to", OUT)
print("PREPARE_COHORT_DONE")
