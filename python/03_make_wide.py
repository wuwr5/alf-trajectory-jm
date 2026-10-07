# -*- coding: utf-8 -*-
"""Aggregate the longitudinal long table into a person-day wide table for R / JMbayes2.

Output: data/jm_wide.csv      (id, tday, one column per marker)
        data/jm_wide_core.csv (core three markers only, a version with less missingness)
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import numpy as np
import pandas as pd
# --- Path resolution: overridable via environment variables, defaults to data/ in the repo ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.environ.get("ALF_DATA_DIR", os.path.join(_ROOT, "data"))
COHORT_XLSX = os.environ.get("ALF_COHORT_XLSX",
                             os.path.join(_ROOT, "cohort", "alf_icu_final_first_stay.xlsx"))
os.makedirs(DATA, exist_ok=True)

OUT = DATA
lab = pd.read_csv(os.path.join(OUT, 'jm_long.csv'))
surv = pd.read_csv(os.path.join(OUT, 'jm_surv.csv'))
base = pd.read_csv(os.path.join(OUT, 'jm_base.csv'))

print(f"[1] long table {len(lab):,} rows, {lab['id'].nunique()} subjects")

# Aggregate by day: t_d -> day index (0 = first ICU day)
lab['tday'] = np.floor(lab['time_d']).astype(int)
wide = (lab.groupby(['id', 'tday', 'marker'])['log_value'].mean()
           .unstack('marker').reset_index())
print(f"[2] after daily aggregation: {len(wide):,} rows (person-days)")

# Restrict to days 0-14 (the main joint-model window)
wide = wide[(wide['tday'] >= 0) & (wide['tday'] <= 14)]
print(f"    restricted to tday 0-14: {len(wide):,} rows")

CORE = ['Bilirubin', 'INR', 'Creatinine']
EXT = ['Lactate', 'Platelet']
keep = ['id', 'tday'] + [c for c in CORE + EXT if c in wide.columns]
wide = wide[keep]

# Merge time-invariant subject-level variables
ind = surv[['id', 'event_time_d', 'event', 'Age', 'gender_male',
            'Liver_transplantation', 'Vasopressin', 'rrt']]
b2 = base[['id', 'MELD', 'SOFA', 'Ascites', 'Sepsis', 'HE', 'HRS', 'Shock']]
ind = ind.merge(b2, on='id', how='left')
wide = wide.merge(ind, on='id', how='left')

wide = wide.sort_values(['id', 'tday'])
wide.to_csv(os.path.join(OUT, 'jm_wide.csv'), index=False, encoding='utf-8')
print(f"[3] -> data/jm_wide.csv  {len(wide):,} rows / {wide['id'].nunique()} subjects")
print("    columns:", list(wide.columns))

# Missingness per marker in the wide table
print("\n[4] Missingness per marker in the wide table (person-day level)")
for c in CORE + EXT:
    if c in wide.columns:
        miss = wide[c].isna().mean()
        per_id = wide.groupby('id')[c].apply(lambda s: s.notna().sum())
        print(f"    {c:<12} missing {100*miss:.1f}%  |  median days with a value per subject "
              f"{per_id.median():.0f} (P25-P75 {per_id.quantile(.25):.0f}-{per_id.quantile(.75):.0f})")

# Event indicator per person-day (id-level survival data for JMbayes2 is stored separately)
print("\n[5] Survival data (id level):", surv.shape, "| events", int(surv['event'].sum()))
print("    -> on the R side use jm_wide.csv (longitudinal) + jm_surv.csv (survival, unique id)")
print("\nDONE")
