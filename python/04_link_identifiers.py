# -*- coding: utf-8 -*-
"""Rebuild the one-to-one mapping between case_id (row order 1..N in the cohort xlsx)
and id (the MIMIC-derived identifier).

Background
----------
Different arms of this project use two sets of subject identifiers:
  * the cohort xlsx and all LLM / classical model arms: the row index case_id (1..N)
  * longitudinal and survival data extracted from PostgreSQL: the MIMIC-derived id

There is no direct key linking them, so the bijection must be rebuilt by
"composite exact matching". This script uses
(age, sex, in-hospital death, liver transplantation, vasopressin, renal replacement therapy)
as the matching key, performs a strict unique match first, then fills in remaining cases
using the (Age, MELD, SOFA) triple.

Output
------
data/jm_id_to_caseid.csv : id, case_id, plus the fields used for matching

Usage
-----
    python python/04_link_identifiers.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# --- Path resolution: overridable via environment variables, defaults to data/ in the repo ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.environ.get("ALF_DATA_DIR", os.path.join(_ROOT, "data"))
COHORT_XLSX = os.environ.get(
    "ALF_COHORT_XLSX",
    os.path.join(_ROOT, "cohort", "alf_icu_final_first_stay.xlsx"),
)
os.makedirs(DATA, exist_ok=True)

# ---------------------------------------------------------------- Inputs
raw = pd.read_excel(COHORT_XLSX, sheet_name="data")
raw["case_id"] = range(1, len(raw) + 1)
raw["outcome"] = raw["outcome"].astype(int)
raw["gender_male"] = (raw["gender"] == "M").astype(int)

base = pd.read_csv(os.path.join(DATA, "jm_base.csv"))          # contains id and covariates
print("Cohort xlsx:", raw.shape, " MIMIC-side jm_base:", base.shape)

# ---------------------------------------------------------------- Composite exact matching
# Use only fields present on both sides. gender_male exists on the xlsx side but not in
# MIMIC-side jm_base, so it is treated as an optional strengthening key
# (included when present, skipped otherwise).
BASE_KEY = ["Age", "outcome", "Liver_transplantation", "Vasopressin", "rrt"]
OPTIONAL_KEY = ["gender_male"]


def norm(df, cols, optional=()):
    out = df.copy()
    keep = []
    for c in cols:
        if c not in out.columns:
            raise KeyError(f"missing column {c}; available: {list(out.columns)[:20]}")
        keep.append(c)
    for c in optional:
        if c in out.columns:
            keep.append(c)
    return out, keep


raw_n, keys_r = norm(raw, BASE_KEY, OPTIONAL_KEY)
bas_n, keys_b = norm(base, BASE_KEY, OPTIONAL_KEY)
# Build the match string only from keys shared by both sides
KEYS = [k for k in keys_r if k in keys_b]
print("Matching keys:", KEYS)


# Compare Age with a float tolerance (precision may differ between the two sides)
def key_of(df):
    parts = []
    for c in KEYS:
        v = df[c]
        parts.append(v.round(4).astype(str) if v.dtype.kind == "f"
                     else v.astype(int).astype(str))
    return parts[0].str.cat(parts[1:], sep="|") if len(parts) > 1 else parts[0]


raw_n["_k"] = key_of(raw_n)
bas_n["_k"] = key_of(bas_n)

# Keep only keys that are unique on both sides, to avoid many-to-many matches
rc = raw_n["_k"].value_counts()
bc = bas_n["_k"].value_counts()
uniq = set(rc[rc == 1].index) & set(bc[bc == 1].index)

m1 = (raw_n[raw_n["_k"].isin(uniq)][["case_id", "_k"]]
      .merge(bas_n[bas_n["_k"].isin(uniq)][["id", "_k"]], on="_k", how="inner")
      .drop(columns="_k"))
print(f"Stage 1 strict unique matching: {len(m1)} subjects")

# ---------------------------------------------------------------- Remaining cases: Age+MELD+SOFA
done_x = set(m1["case_id"]); done_b = set(m1["id"])
r2 = raw_n[~raw_n["case_id"].isin(done_x)].copy()
b2 = bas_n[~bas_n["id"].isin(done_b)].copy()

m2 = pd.DataFrame(columns=["case_id", "id"])
if len(r2) and len(b2) and {"MELD", "SOFA"}.issubset(r2.columns) and {"MELD", "SOFA"}.issubset(b2.columns):
    def k3(df):
        return (df["Age"].round(3).astype(str) + "|" +
                df["MELD"].round(3).astype(str) + "|" +
                df["SOFA"].round(3).astype(str))
    r2["_k"] = k3(r2); b2["_k"] = k3(b2)
    rc2 = r2["_k"].value_counts(); bc2 = b2["_k"].value_counts()
    u2 = set(rc2[rc2 == 1].index) & set(bc2[bc2 == 1].index)
    m2 = (r2[r2["_k"].isin(u2)][["case_id", "_k"]]
          .merge(b2[b2["_k"].isin(u2)][["id", "_k"]], on="_k", how="inner")
          .drop(columns="_k"))
    print(f"Stage 2 Age+MELD+SOFA fill-in: {len(m2)} subjects")

mp = pd.concat([m1, m2], ignore_index=True).drop_duplicates(subset=["case_id"]).drop_duplicates(subset=["id"])

# ---------------------------------------------------------------- Validation
n_raw, n_base = len(raw), len(base)
ok = (len(mp) == n_raw == n_base) and mp["case_id"].is_unique and mp["id"].is_unique
print(f"\nMapping result: {len(mp)} rows  (xlsx {n_raw} / MIMIC {n_base})  bijection consistent={ok}")
if not ok:
    print("!! Not a perfect bijection; check the key definition or cohort composition "
          "(unmatched cases listed below)")
    print("  unmatched case_id count:", n_raw - mp["case_id"].nunique())
    print("  unmatched id count:", n_base - mp["id"].nunique())

out = mp.merge(base[["id"] + [c for c in ["Age", "MELD", "SOFA", "outcome"] if c in base.columns]],
               on="id", how="left")
out.to_csv(os.path.join(DATA, "jm_id_to_caseid.csv"), index=False, encoding="utf-8")
print("-> jm_id_to_caseid.csv")
