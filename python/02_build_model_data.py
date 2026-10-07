# -*- coding: utf-8 -*-
"""Build the joint-model analysis dataset (JMbayes2-friendly format) plus landmark wide tables.

Input:  data/long_labs_clean.csv, data/long_surv.csv, the raw cohort xlsx
Output:
    data/jm_long.csv          multivariate long table (id, marker, time_d, value, log_value)
    data/jm_surv.csv          survival data (id, time_d, event, transplant)
    data/jm_base.csv          baseline covariates (id, demographics / complications /
                              treatments / scores)
    data/jm_landmark_L2.csv   landmark Day-2 wide table (for JM vs static model comparison)
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
XLSX = COHORT_XLSX

# Physiologically plausible ranges (values outside are set to missing to guard against
# unit or entry errors)
RANGE = {
    'Bilirubin': (0.05, 80), 'INR': (0.5, 25), 'Creatinine': (0.05, 30),
    'Lactate': (0.1, 40), 'Platelet': (1, 1500), 'ALT': (1, 20000),
    'AST': (1, 20000), 'Albumin': (0.5, 8), 'BUN': (1, 200),
    'Sodium': (100, 180), 'WBC': (0.1, 200), 'Hemoglobin': (1, 25),
}
CORE = ['Bilirubin', 'INR', 'Creatinine']          # core markers of the JM longitudinal submodels
EXT = ['Lactate', 'Platelet']                       # optional extensions

# ---------------------------------------------------------------- Load data
lab = pd.read_csv(os.path.join(OUT, 'long_labs_clean.csv'))
surv = pd.read_csv(os.path.join(OUT, 'long_surv.csv'))
raw = pd.read_excel(XLSX, sheet_name='data')
print(f"[1] longitudinal {len(lab):,} rows | survival {len(surv):,} subjects | raw {len(raw):,} subjects")

# ---------------------------------------------------------------- Cleaning
n0 = len(lab)
lab = lab[lab['t_d'] >= 0]                                  # keep only post ICU admission
for m, (lo, hi) in RANGE.items():
    bad = (lab['marker'] == m) & ((lab['valuenum'] < lo) | (lab['valuenum'] > hi))
    if bad.sum():
        lab = lab[~bad]
        print(f"    {m}: dropped {int(bad.sum())} out-of-range records")
print(f"    {len(lab):,} rows after range cleaning (from {n0:,})")

# Duplicates at the same (id, marker, time) -> take the mean
lab = (lab.groupby(['hadm_id', 'marker', 't_d'], as_index=False)['valuenum'].mean()
          .rename(columns={'hadm_id': 'id', 't_d': 'time_d'}))
print(f"    {len(lab):,} rows after de-duplication")

# Log transform (bilirubin / INR / creatinine / lactate / ALT / AST are right-skewed)
lab['log_value'] = np.where(lab['marker'].isin(['Bilirubin', 'INR', 'Creatinine',
                                                'Lactate', 'ALT', 'AST']),
                            np.log(lab['valuenum']), lab['valuenum'])
lab = lab.sort_values(['id', 'marker', 'time_d'])
lab.to_csv(os.path.join(OUT, 'jm_long.csv'), index=False, encoding='utf-8')
print(f"    -> data/jm_long.csv  {len(lab):,} rows, {lab['id'].nunique()} subjects")

# ---------------------------------------------------------------- Survival data
s = surv[['hadm_id', 'event_time_d', 'event']].rename(columns={'hadm_id': 'id'})
s = s.merge(raw[['hadm_id', 'Liver_transplantation', 'Vasopressin', 'rrt',
                 'Age', 'gender']], left_on='id', right_on='hadm_id', how='left')
s['gender_male'] = (s['gender'] == 'M').astype(int)
s = s.drop(columns=['hadm_id', 'gender'])
s.loc[s['event_time_d'] <= 0, 'event_time_d'] = 0.05
s = s.sort_values('id')
s.to_csv(os.path.join(OUT, 'jm_surv.csv'), index=False, encoding='utf-8')
print(f"[2] -> data/jm_surv.csv  n={len(s)}, events {int(s['event'].sum())} "
      f"({100*s['event'].mean():.1f}%), liver transplantation {int(s['Liver_transplantation'].sum())}")

# ---------------------------------------------------------------- Baseline covariates
base_cols = ['hadm_id', 'Age', 'gender_male', 'Alcoholic_plus_viral', 'Alcoholic_only',
             'Viral_only', 'Other', 'Ascites', 'Sepsis', 'HE', 'HRS', 'EVB', 'SBP',
             'Shock', 'Pneumonia', 'MELD', 'SOFA', 'Liver_transplantation',
             'Vasopressin', 'rrt', 'outcome']
b = raw[[c for c in base_cols if c in raw.columns]].rename(columns={'hadm_id': 'id'})
b.to_csv(os.path.join(OUT, 'jm_base.csv'), index=False, encoding='utf-8')
print(f"[3] -> data/jm_base.csv  n={len(b)}, {len(b.columns)-1} covariates")

# ---------------------------------------------------------------- Landmark wide tables
print("\n[4] Landmark wide tables (for a fair comparison between JM and static models)")
def landmark_table(L, window_lo=0.0):
    """For t_d in [window_lo, L], take per marker: first value, last value, mean, slope, count."""
    sub = lab[(lab['time_d'] >= window_lo) & (lab['time_d'] <= L)]
    rows = []
    for m in CORE + EXT:
        s2 = sub[sub['marker'] == m].sort_values(['id', 'time_d'])
        g = s2.groupby('id')
        agg = pd.DataFrame({
            f'{m}_first': g['valuenum'].first(),
            f'{m}_last': g['valuenum'].last(),
            f'{m}_mean': g['valuenum'].mean(),
            f'{m}_n': g['valuenum'].size(),
            f'{m}_t_first': g['time_d'].first(),
            f'{m}_t_last': g['time_d'].last(),
        })
        # Simple slope ((last - first) / time difference); set to 0 if the gap is < 0.5 d
        dt = agg[f'{m}_t_last'] - agg[f'{m}_t_first']
        agg[f'{m}_slope'] = np.where(dt >= 0.5, (agg[f'{m}_last'] - agg[f'{m}_first']) / dt, 0.0)
        rows.append(agg.drop(columns=[f'{m}_t_first', f'{m}_t_last']))
    w = pd.concat(rows, axis=1)
    return w

for L in [1, 2, 3, 5]:
    w = landmark_table(L)
    at_risk = set(s.loc[s['event_time_d'] > L, 'id'])
    w = w.reindex(sorted(at_risk))
    cov = w[[f'{m}_n' for m in CORE]].notna().all(axis=1).mean()
    print(f"    L={L}d: risk set n={len(w)}, all three core markers available for {100*cov:.1f}%")
    w.to_csv(os.path.join(OUT, f'jm_landmark_L{L}.csv'), encoding='utf-8')

# Primary analysis landmark = Day 2
w2 = landmark_table(2).reindex(sorted(set(s.loc[s['event_time_d'] > 2, 'id'])))
extra = raw.set_index('hadm_id')[['MELD', 'SOFA', 'Ascites', 'Sepsis', 'HE', 'HRS',
                                  'Shock', 'Liver_transplantation', 'Vasopressin', 'rrt']]
w2 = w2.join(s.set_index('id')[['event_time_d', 'event', 'Age', 'gender_male']]).join(extra)
w2.to_csv(os.path.join(OUT, 'jm_landmark_L2.csv'), encoding='utf-8')
print(f"    primary analysis L=2d wide table -> data/jm_landmark_L2.csv  n={len(w2)}")

# ---------------------------------------------------------------- Summary
print("\n" + "=" * 72)
print("[5] Modelling readiness check")
print(f"    long table   : {len(lab):,} rows / {lab['id'].nunique()} subjects / {lab['marker'].nunique()} markers")
for m in CORE:
    sub = lab[lab['marker'] == m]
    cnt = sub.groupby('id').size()
    print(f"      {m:<12} coverage {100*len(cnt)/2508:.1f}%  median {cnt.median():.0f} measurements  "
          f"P25-P75 {cnt.quantile(.25):.0f}-{cnt.quantile(.75):.0f}")
print(f"    survival data: n={len(s)}, events {int(s['event'].sum())}, "
      f"median event time {s.loc[s['event']==1,'event_time_d'].median():.2f}d, "
      f"median censoring time {s.loc[s['event']==0,'event_time_d'].median():.2f}d")
print(f"    liver transplant: {int(s['Liver_transplantation'].sum())} cases "
      f"(competing risk, needs handling)")
print("\nDONE")
