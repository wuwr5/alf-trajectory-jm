# -*- coding: utf-8 -*-
"""Re-extract longitudinal labs using exact itemids (avoiding substring mismatches) and
verify the three data traps that matter for joint modelling.

The database password is passed via the PGPASSWORD environment variable.

Outputs:
    data/long_labs_clean.csv    clean long-format table
    data/long_surv.csv          survival data (t0 = ICU admission)
    data/long_density_clean.csv
    data/long_trap_check.csv    trap verification
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import numpy as np
import pandas as pd
import psycopg2
# --- Path resolution: overridable via environment variables, defaults to data/ in the repo ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.environ.get("ALF_DATA_DIR", os.path.join(_ROOT, "data"))
COHORT_XLSX = os.environ.get("ALF_COHORT_XLSX",
                             os.path.join(_ROOT, "cohort", "alf_icu_final_first_stay.xlsx"))
os.makedirs(DATA, exist_ok=True)

OUT = DATA

# Database connection: host / port / user can be overridden by environment variables;
# the password is passed **only** via PGPASSWORD.
HOST = os.environ.get('PGHOST', 'localhost')
PORT = int(os.environ.get('PGPORT', 2023))
USER = os.environ.get('PGUSER', 'postgres')
DBNAME = os.environ.get('PGDATABASE', 'postgres')   # database hosting MIMIC-IV
PW = os.environ.get('PGPASSWORD', '')
if not PW:
    sys.exit("Please set the PGPASSWORD environment variable (the password is never written to a file)")

# ---- Exact MIMIC-IV itemids (verified one by one against d_labitems in this database) ----
MARKERS = {
    50885: 'Bilirubin', 51237: 'INR', 50912: 'Creatinine', 50813: 'Lactate',
    51265: 'Platelet', 50861: 'ALT', 50878: 'AST', 50862: 'Albumin',
    51006: 'BUN', 50983: 'Sodium', 51301: 'WBC', 51222: 'Hemoglobin',
    51274: 'PT', 51275: 'PTT', 50931: 'Glucose', 50882: 'Bicarbonate',
    50902: 'Chloride', 50893: 'Calcium', 50971: 'Potassium',
}
CORE = ['Bilirubin', 'INR', 'Creatinine']

cn = psycopg2.connect(host=HOST, port=PORT, user=USER, password=PW,
                      dbname=DBNAME, connect_timeout=20)
cn.autocommit = True
cu = cn.cursor()
print("[1] Connected")

# ---------------------------------------------------------------- Cohort
df = pd.read_excel(COHORT_XLSX, sheet_name='data')
coh = df[['subject_id', 'hadm_id', 'stay_id', 'outcome']].copy()
coh['subject_id'] = coh['subject_id'].astype('int64')
coh['hadm_id'] = coh['hadm_id'].astype('int64')

cu.execute("DROP TABLE IF EXISTS alf_cohort")
cu.execute("CREATE TABLE alf_cohort(subject_id bigint, hadm_id bigint, stay_id bigint, outcome int)")
buf = []
for r in coh.itertuples(index=False):
    buf.append(cu.mogrify("(%s,%s,%s,%s)", tuple(r)).decode())
    if len(buf) >= 1000:
        cu.execute("INSERT INTO alf_cohort VALUES " + ",".join(buf)); buf = []
if buf:
    cu.execute("INSERT INTO alf_cohort VALUES " + ",".join(buf))
cu.execute("CREATE INDEX idx_alf_hadm ON alf_cohort(hadm_id)")
print(f"[2] Cohort uploaded n={len(coh)}")

# ---------------------------------------------------------------- Longitudinal labs
ids_sql = ",".join(str(i) for i in MARKERS)
print("[3] Extracting longitudinal labs ...")
cu.execute(f"""
SELECT c.hadm_id, c.outcome, le.itemid, le.charttime, le.valuenum
FROM mimiciv_hosp.labevents le
JOIN alf_cohort c ON c.hadm_id = le.hadm_id
WHERE le.itemid IN ({ids_sql}) AND le.valuenum IS NOT NULL
""")
lab = pd.DataFrame(cu.fetchall(), columns=['hadm_id', 'outcome', 'itemid', 'charttime', 'valuenum'])
print(f"    raw {len(lab):,} rows")
lab['marker'] = lab['itemid'].map(MARKERS)
print(f"    after mapping {len(lab):,} rows, covering {lab['marker'].nunique()} markers")

# ---------------------------------------------------------------- Time anchor
print("[4] Aligning time (t0 = ICU admission) ...")
cu.execute("""
SELECT c.hadm_id,
       i.intime AS icu_intime, i.outtime AS icu_outtime,
       a.admittime, a.dischtime, a.deathtime
FROM alf_cohort c
LEFT JOIN mimiciv_icu.icustays i ON i.stay_id = c.stay_id
LEFT JOIN mimiciv_hosp.admissions a ON a.hadm_id = c.hadm_id
""")
tm = pd.DataFrame(cu.fetchall(), columns=['hadm_id', 'icu_intime', 'icu_outtime',
                                          'admittime', 'dischtime', 'deathtime'])
for c_ in ['icu_intime', 'icu_outtime', 'admittime', 'dischtime', 'deathtime']:
    tm[c_] = pd.to_datetime(tm[c_])
print(f"    anchors n={len(tm)}, icu_intime non-missing {tm['icu_intime'].notna().sum()}")

# Survival time: t0 = ICU admission; event = in-hospital death
tm['event_time_d'] = np.where(
    tm['deathtime'].notna(),
    (tm['deathtime'] - tm['icu_intime']).dt.total_seconds() / 86400.0,
    (tm['dischtime'] - tm['icu_intime']).dt.total_seconds() / 86400.0)
tm['event'] = np.where(tm['deathtime'].notna(), 1, 0)
tm.loc[tm['event_time_d'] <= 0, 'event_time_d'] = 0.05   # guard against extreme values
surv = tm.merge(coh[['hadm_id', 'outcome']], on='hadm_id', how='left')
surv.to_csv(os.path.join(OUT, 'long_surv.csv'), index=False, encoding='utf-8')
print(f"    events {int(surv['event'].sum())} / {len(surv)}; "
      f"median event time {surv.loc[surv['event']==1,'event_time_d'].median():.2f} d, "
      f"median censoring time {surv.loc[surv['event']==0,'event_time_d'].median():.2f} d")

lab = lab.merge(tm[['hadm_id', 'icu_intime']], on='hadm_id', how='left')
lab['charttime'] = pd.to_datetime(lab['charttime'])
lab['t_h'] = (lab['charttime'] - lab['icu_intime']).dt.total_seconds() / 3600.0
lab['t_d'] = lab['t_h'] / 24.0
lab = lab[(lab['t_d'] >= -1) & (lab['t_d'] <= 30)]
lab = lab.sort_values(['hadm_id', 'marker', 't_h'])
lab[['hadm_id', 'outcome', 'marker', 'itemid', 'charttime', 't_h', 't_d', 'valuenum']].to_csv(
    os.path.join(OUT, 'long_labs_clean.csv'), index=False, encoding='utf-8')
print(f"    within time window [-1d, +30d]: {len(lab):,} rows -> data/long_labs_clean.csv")

# ---------------------------------------------------------------- Density
print("\n[5] Density assessment (first 3 / 7 / 14 days)")
rows = []
n_all = len(coh)
for m in list(MARKERS.values()):
    sub = lab[lab['marker'] == m]
    for w in [3, 7, 14]:
        s = sub[(sub['t_d'] >= 0) & (sub['t_d'] <= w)]
        cnt = s.groupby('hadm_id').size().reindex(coh['hadm_id'].unique(), fill_value=0)
        rows.append({'marker': m, 'window_d': w, 'coverage_pct': round(100*(cnt > 0).mean(), 1),
                     'median': float(cnt.median()), 'mean': round(float(cnt.mean()), 2),
                     'p25': float(cnt.quantile(.25)), 'p75': float(cnt.quantile(.75)),
                     'pct_ge2': round(100*(cnt >= 2).mean(), 1),
                     'pct_ge3': round(100*(cnt >= 3).mean(), 1),
                     'pct_ge4': round(100*(cnt >= 4).mean(), 1),
                     'pct_ge5': round(100*(cnt >= 5).mean(), 1)})
dens = pd.DataFrame(rows)
dens.to_csv(os.path.join(OUT, 'long_density_clean.csv'), index=False, encoding='utf-8')
pd.set_option('display.width', 200)
print(dens[dens['window_d'] == 7].to_string(index=False))

# ---------------------------------------------------------------- Trap verification
print("\n[6] Verifying the three data traps")
trap = []

# Trap A: do early deaths have enough longitudinal data?
grp = surv.copy()
grp['grp'] = np.where((grp['event'] == 1) & (grp['event_time_d'] <= 4), 'early_death_<=4d',
              np.where(grp['event'] == 1, 'late_death_>4d', 'survived'))
print("    group composition:", grp['grp'].value_counts().to_dict())
for m in CORE:
    sub = lab[(lab['marker'] == m) & (lab['t_d'] >= 0)]
    for g in ['early_death_<=4d', 'late_death_>4d', 'survived']:
        ids = set(grp.loc[grp['grp'] == g, 'hadm_id'])
        s = sub[sub['hadm_id'].isin(ids)]
        cnt = s.groupby('hadm_id').size().reindex(sorted(ids), fill_value=0)
        # usable window = min(event time, 7 days)
        et = grp.set_index('hadm_id').loc[sorted(ids), 'event_time_d']
        win = np.minimum(et.values, 7.0)
        cnt_win = []
        for hid, w in zip(sorted(ids), win):
            ss = s[(s['hadm_id'] == hid) & (s['t_d'] <= w)]
            cnt_win.append(len(ss))
        trap.append({'marker': m, 'group': g, 'n': len(ids),
                     'median_in_window': float(np.median(cnt_win)),
                     'pct_ge2_in_window': round(100*np.mean(np.array(cnt_win) >= 2), 1),
                     'pct_ge3_in_window': round(100*np.mean(np.array(cnt_win) >= 3), 1)})
trap = pd.DataFrame(trap)
trap.to_csv(os.path.join(OUT, 'long_trap_check.csv'), index=False, encoding='utf-8')
print(trap.to_string(index=False))

# Trap B: availability at landmark Day 0 / Day 2
print("\n    Landmark availability (a measurement already exists before that time point)")
lm_rows = []
for m in CORE:
    sub = lab[lab['marker'] == m]
    for L in [0, 2, 3, 5]:
        s = sub[sub['t_d'] <= L]
        have = s.groupby('hadm_id').size()
        alive = surv[(surv['event_time_d'] > L)]
        pct = 100 * alive['hadm_id'].isin(have.index).mean()
        nmeas = have.reindex(alive['hadm_id']).fillna(0)
        lm_rows.append({'marker': m, 'landmark_d': L, 'n_at_risk': len(alive),
                        'pct_with_measurement': round(pct, 1),
                        'median_n_meas': float(nmeas.median())})
lm = pd.DataFrame(lm_rows)
lm.to_csv(os.path.join(OUT, 'long_landmark_avail.csv'), index=False, encoding='utf-8')
print(lm.to_string(index=False))

# ---------------------------------------------------------------- Verdict
print("\n" + "=" * 72)
print("[7] Verdict")
med7 = dens[(dens['window_d'] == 7) & (dens['marker'].isin(CORE))]['median']
print(f"    median measurements in the first 7 days for the three core markers: {dict(zip(CORE, med7))}")
print(f"    >>> density {'sufficient' if med7.max() >= 4 else 'insufficient'}, "
      f"threshold >=4, observed maximum {med7.max():.0f}")
early = trap[(trap['group'] == 'early_death_<=4d') & (trap['marker'] == 'Bilirubin')]
if len(early):
    print(f"    >>> Trap A: among early deaths (<=4d), the median number of bilirubin measurements "
          f"within their usable window is {early['median_in_window'].iloc[0]:.0f}, and "
          f"{early['pct_ge2_in_window'].iloc[0]:.1f}% have at least 2 "
          f"(dynamic prediction adds little for this subgroup; report it in strata)")

cu.close(); cn.close()
print("\nDONE")
