# -*- coding: utf-8 -*-
"""Recalibrate LLM probabilities (isotonic / Platt) using 5-fold cross-fitting to avoid
optimistic bias. On the Day-2 risk set, fit calibrators to the raw Hy4 / DeepSeek
probabilities and output calibrated probabilities plus pre/post-calibration metrics.

Output: data/ll_calibrated.csv, data/ll_calibration_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass
from sklearn.isotonic import IsotonicRegression
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

hy4 = pd.read_csv(os.path.join(OUT, "hy4_pred_full.csv")); hy4["hy4_raw"] = hy4[["run1","run2","run3"]].mean(axis=1)/100.0
ds1 = pd.read_csv(os.path.join(OUT, "ds_pred_run1.csv")).rename(columns={"risk":"r1"})
ds2 = pd.read_csv(os.path.join(OUT, "ds_pred_run2.csv")).rename(columns={"risk":"r2"})
ds3 = pd.read_csv(os.path.join(OUT, "ds_pred_run3.csv")).rename(columns={"risk":"r3"})
ds  = ds1.merge(ds2[["case_id","r2"]], on="case_id").merge(ds3[["case_id","r3"]], on="case_id")
ds["ds_raw"] = ds[["r1","r2","r3"]].mean(axis=1)/100.0

ll = hy4[["case_id","hy4_raw"]].merge(ds[["case_id","ds_raw"]], on="case_id").merge(truth, on="case_id")
# Day-2 landmark risk set (unified definition, consistent with the all-arm summary):
#   include those still alive at Day 2. Exclude only those who died within L days of ICU
#   admission (event == 1 & event_time_d <= L).
#   Note: do NOT filter with event_time_d > L -- some patients left the ICU within 2 days,
#   have event = 0, and their in-hospital follow-up ends there; they are still alive at Day 2
#   and are legitimate members of the risk set (an earlier version wrongly dropped 15 of them).
ll = ll[~((ll["outcome"] == 1) & (ll["event_time_d"] <= L))].copy()
print(f"LLM calibration sample (Day-2 risk set, unified definition): n={len(ll)}")

def cv_calib(raw, y, kind):
    raw = np.clip(raw, 1e-4, 1-1e-4); y = np.asarray(y, int); n = len(raw)
    cal = np.full(n, np.nan)
    if kind == "isotonic":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        for tr, te in skf.split(raw, y):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1-1e-4)
            ir.fit(raw[tr], y[tr]); cal[te] = ir.predict(raw[te])
    else:  # Platt: logistic regression on logit(raw)
        lp = np.log(raw/(1-raw))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        for tr, te in skf.split(raw, y):
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(lp[tr].reshape(-1,1), y[tr]); cal[te] = lr.predict_proba(lp[te].reshape(-1,1))[:,1]
    return cal

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

out = ll[["case_id","outcome","event_time_d"]].copy()
rows = []
for arm, rc in [("Hy4","hy4_raw"), ("DeepSeek","ds_raw")]:
    for h in (7, 14):
        yh = ((ll["outcome"]==1) & (ll["event_time_d"] <= L+h)).astype(int).values
        raw_p = ll[rc].clip(1e-4, 1-1e-4).values
        iso = cv_calib(raw_p, yh, "isotonic")
        pla = cv_calib(raw_p, yh, "platt")
        out[f"{arm}_raw_h{h}"] = raw_p
        out[f"{arm}_iso_h{h}"] = iso
        out[f"{arm}_pla_h{h}"] = pla
        for name, pp in [("raw", raw_p), ("isotonic", iso), ("platt", pla)]:
            a, lo, hi = delong(yh, pp); ci, cs = cal(yh, pp); br = brier(yh, pp)
            rows.append({"arm":arm,"horizon":h,"method":name,"AUC":round(a,3),"AUC_lo":round(lo,3),
                         "AUC_hi":round(hi,3),"cal_int":round(ci,3),"cal_slope":round(cs,3),
                         "Brier":round(br,4),"n":int(np.isfinite(pp).sum()),"ev":int(yh[np.isfinite(pp)].sum())})
            print(f"{arm} h={h} {name:8s} AUC={a:.3f}[{lo:.3f},{hi:.3f}] "
                  f"cal(sl={cs:.2f},int={ci:.2f}) Brier={br:.4f}")

out.to_csv(os.path.join(OUT, "ll_calibrated.csv"), index=False, encoding="utf-8")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "ll_calibration_summary.csv"), index=False, encoding="utf-8")
print("LL_CALIB_DONE")
