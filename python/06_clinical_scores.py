# -*- coding: utf-8 -*-
"""Clinical score comparator arms: discrimination and calibration of MELD / SOFA / ALFSG
on the Day-2 risk set.

- MELD / SOFA: use the ready-made scores in jm_base.csv, converted to probabilities by
  in-cohort 5-fold cross-fitted logistic regression.
- ALFSG: adopt the US-ALFSG variable structure (hepatic encephalopathy grade / log bilirubin /
  log INR / vasopressin / etiology; Koch 2016, Clin Gastroenterol Hepatol, C = 0.84), with
  coefficients estimated in this cohort (the publication gives no portable coefficients).

Output: data/scores_pred.csv, data/scores_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
# --- Path resolution: overridable via environment variables, defaults to data/ in the repo ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.environ.get("ALF_DATA_DIR", os.path.join(_ROOT, "data"))
COHORT_XLSX = os.environ.get("ALF_COHORT_XLSX",
                             os.path.join(_ROOT, "cohort", "alf_icu_final_first_stay.xlsx"))
os.makedirs(DATA, exist_ok=True)

OUT = DATA
XLSX = COHORT_XLSX
L = 2

raw = pd.read_excel(XLSX, sheet_name="data")
raw["case_id"] = range(1, len(raw) + 1); raw["outcome"] = raw["outcome"].astype(int)
mp  = pd.read_csv(os.path.join(OUT, "jm_id_to_caseid.csv"))
js  = pd.read_csv(os.path.join(OUT, "jm_surv.csv")).merge(mp, on="id", how="left")
truth = raw[["case_id","outcome"]].merge(js[["case_id","event_time_d"]], on="case_id", how="left")

base = pd.read_csv(os.path.join(OUT, "jm_base.csv"))
# Both jm_base.csv and jm_id_to_caseid.csv carry their own `outcome` column.
# Drop them and take `outcome` from the cohort workbook as the single ground truth,
# otherwise pandas suffixes the colliding columns (outcome_x / outcome_y).
base = base.drop(columns=["outcome"], errors="ignore")
mp_id = mp[["id", "case_id"]].drop_duplicates("id")
b = base.merge(mp_id, on="id", how="inner").merge(truth, on="case_id", how="inner")

# Day-2 longitudinal values (most recent measurement with tday <= 2);
# jm_wide stores log values -> exponentiate back to the original scale
wide = pd.read_csv(os.path.join(OUT, "jm_wide.csv"))
wd = wide[wide["tday"] <= L]
day2 = wd.sort_values("tday").groupby("id").last().reset_index()[["id","Bilirubin","INR","Creatinine"]]
b = b.merge(day2, on="id", how="left")
b["Bili_raw"] = np.exp(b["Bilirubin"]); b["INR_raw"] = np.exp(b["INR"]); b["Cre_raw"] = np.exp(b["Creatinine"])

# Day-2 landmark risk set (unified definition): exclude only those who died within L days.
# Do NOT filter with event_time_d > L -- some patients left the ICU within 2 days with
# event = 0 and are still alive at Day 2, so they are legitimate members of the risk set.
b = b[~((b["outcome"] == 1) & (b["event_time_d"] <= L))].copy()
print(f"Score-arm sample (Day-2 risk set, unified definition): n={len(b)}")

def clean(X):
    X = np.asarray(X, float)
    if X.ndim == 1: X = X.reshape(-1, 1)
    # Impute with the column median so that no NaN remains
    for j in range(X.shape[1]):
        col = X[:, j]; med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        col[~np.isfinite(col)] = med
    return X

def cv_prob(design, y, k=5):
    X = clean(design); y = np.asarray(y, int); n = len(y); prob = np.full(n, np.nan)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    for tr, te in skf.split(X, y):
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(X[tr], y[tr]); prob[te] = lr.predict_proba(X[te])[:, 1]
    return prob

def delong(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); m = np.isfinite(s); y = y[m]; s = s[m]
    pos = s[y==1]; neg = s[y==0]; n1, n0 = len(pos), len(neg)
    auc = (pos[:,None] > neg[None,:]).mean() + 0.5*(pos[:,None]==neg[None,:]).mean()
    V10 = (pos[:,None] >= neg[None,:]).mean(axis=1); V01 = (pos[:,None] >= neg[None,:]).mean(axis=0)
    S10 = V10-auc; S01 = V01-auc
    var = np.var(S10, ddof=1)/n1 + np.var(S01, ddof=1)/n0; se = np.sqrt(var)
    return float(auc), float(auc-1.96*se), float(auc+1.96*se)
def cal(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float); m = np.isfinite(s)&(s>0)&(s<1); y=y[m]; s=s[m]
    lp = np.log(s/(1-s)).clip(-15,15)
    lr = LogisticRegression(C=1e6, max_iter=1000); lr.fit(lp.reshape(-1,1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])
def brier(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float); m = np.isfinite(s)
    return float(np.mean((y[m]-s[m])**2))

# ALFSG variable structure (US-ALFSG: HE grade + log bilirubin + log INR + vasopressin + etiology)
X_alf = pd.DataFrame({
    "HE":      b["HE"].fillna(0),
    "logBili": np.log(b["Bili_raw"].fillna(np.nanmedian(b["Bili_raw"])) + 1),
    "logINR":  np.log(b["INR_raw"].fillna(np.nanmedian(b["INR_raw"]))),
    "Vaso":    (b["Vasopressin"] == 1).astype(int),
})
for c in ["Alcoholic_only", "Viral_only", "Other"]:
    X_alf[c] = b[c].astype(int) if c in b.columns else 0

out = b[["id","case_id","outcome","event_time_d"]].copy()
rows = []
for h in (7, 14):
    yh = ((b["outcome"]==1) & (b["event_time_d"] <= L+h)).astype(int).values
    p_meld = cv_prob(b[["MELD"]].values, yh)
    p_sofa = cv_prob(b[["SOFA"]].values, yh)
    p_alf  = cv_prob(X_alf.values, yh)
    out[f"MELD_h{h}"]  = p_meld
    out[f"SOFA_h{h}"]  = p_sofa
    out[f"ALFSG_h{h}"] = p_alf
    for name, pp in [("MELD", p_meld), ("SOFA", p_sofa), ("ALFSG", p_alf)]:
        a, lo, hi = delong(yh, pp); ci, cs = cal(yh, pp); br = brier(yh, pp)
        rows.append({"arm": name, "horizon": h, "AUC": round(a,3), "AUC_lo": round(lo,3),
                     "AUC_hi": round(hi,3), "cal_int": round(ci,3), "cal_slope": round(cs,3),
                     "Brier": round(br,4), "n": int(np.isfinite(pp).sum()), "ev": int(yh[np.isfinite(pp)].sum())})
        print(f"{name} h={h} AUC={a:.3f}[{lo:.3f},{hi:.3f}] cal(sl={cs:.2f},int={ci:.2f}) "
              f"Brier={br:.4f} ev={int(yh[np.isfinite(pp)].sum())}")

out.to_csv(os.path.join(OUT, "scores_pred.csv"), index=False, encoding="utf-8")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "scores_summary.csv"), index=False, encoding="utf-8")
print("SCORES_DONE")
