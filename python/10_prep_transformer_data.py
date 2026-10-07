# -*- coding: utf-8 -*-
"""Comparator arm 2a: build the feature table for the FT-Transformer.

Projects clinical_full.csv onto the exact feature list expected by the
alf-transformer-mimic configs (configs/default.yaml) and renames case_id to
stay_id, which is the identifier that package expects.

Input : data/clinical_full.csv
Output: data/transformer_cohort.csv
"""
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

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

OUT = DATA

NUM = ["Age", "MAP", "SpO2", "Hemoglobin", "Platelet", "WBC", "Albumin", "BUN",
       "Chloride", "Creatinine", "Sodium", "ALT", "AST", "Total_bilirubin", "PT",
       "APTT", "INR", "Lactate", "MELD", "SOFA"]
BIN = ["Alcoholic_only", "Alcoholic_plus_viral", "Viral_only", "Other", "Ascites",
       "Sepsis", "HE", "HRS", "EVB", "SBP", "Shock", "Pneumonia",
       "Liver_transplantation", "Vasopressin", "rrt"]

src = os.path.join(OUT, "clinical_full.csv")
if not os.path.exists(src):
    sys.exit(f"Missing {src}. Run 08_prepare_cohort.py first.")

df = pd.read_csv(src)
print(f"Source clinical_full.csv: {df.shape} (winsorised + MICE imputed)")

missing_cols = [c for c in NUM + BIN + ["outcome"] if c not in df.columns]
if missing_cols:
    sys.exit(f"clinical_full.csv is missing required columns: {missing_cols}")

out = df[["case_id"] + NUM + BIN + ["outcome"]].copy()
out = out.rename(columns={"case_id": "stay_id"})

print("\n=== Missingness check ===")
na = out.isna().sum()
print("total missing:", int(na.sum()), "(expected 0 after MICE)")
if na.sum():
    print(na[na > 0])

print("\n=== Outcome ===")
print(f"n = {len(out)} | events = {int(out['outcome'].sum())} | "
      f"event rate = {out['outcome'].mean():.4f}")

print("\n=== Binary variable value check (expect 0/1) ===")
bad = [c for c in BIN if not set(pd.unique(out[c])).issubset({0, 1, 0.0, 1.0})]
print("non 0/1 variables:", bad if bad else "none")

print("\n=== Continuous variable ranges ===")
print(out[NUM].describe().T[["mean", "std", "min", "max"]].round(2).to_string())

p = os.path.join(OUT, "transformer_cohort.csv")
out.to_csv(p, index=False)
print(f"\n-> {p}  shape={out.shape}")

# The Hybrid Transformer needs a daily trajectory panel, which this cohort does
# not have; only the cross-sectional FT-Transformer can be run here.
print("\n=== Daily trajectory panel check ===")
traj_needed = [f"{v}_d{d}" for v in ["bilirubin", "inr", "creatinine", "platelet"]
               for d in range(7)]
have = [c for c in traj_needed if c in df.columns]
print(f"Hybrid Transformer needs {len(traj_needed)} columns "
      f"(bilirubin/inr/creatinine/platelet x d0-d6); available: {len(have)}")
print("=> only the FT-Transformer (cross-sectional) arm is runnable here")
print("PREP_TRANSFORMER_DONE")
