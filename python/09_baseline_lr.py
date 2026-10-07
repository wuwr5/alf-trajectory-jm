# -*- coding: utf-8 -*-
"""Comparator arm 1: baseline logistic regression.

Multivariable logistic regression with forward stepwise selection by Wald test
(entry P < 0.05). The model is fitted once on the full cohort and, for
completeness, separately within each aetiology subgroup.

The arm's out-of-sample estimate is the bootstrap optimism-corrected AUC
(fixed variable set, coefficients re-estimated in 500 bootstrap replicates).
The per-case predicted probability of the overall model is written to
data/clinical_pred_full.csv, which is what the 11-arm comparison consumes.

Input : data/clinical_full.csv, data/meta.json, data/outcome_key_300.csv
Output: data/clinical_pred_full.csv, data/clinical_coefs.csv,
        data/clinical_auc_summary.csv, data/clinical_selection_trace.csv
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

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

OUT = DATA
SEED = 2024
N_BOOT = 500
np.random.seed(SEED)


def _need(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        sys.exit(f"Missing {p}. Run 08_prepare_cohort.py first.")
    return p


clinical = pd.read_csv(_need("clinical_full.csv"))
with open(_need("meta.json"), encoding="utf-8") as fh:
    import json
    meta = json.load(fh)

ALLPRED = meta["ALLPRED"]
ETIO = meta["ETIO"]
cohorts = meta["cohorts"]


def forward_wald(X, y, candidates, p_in=0.05, max_iter=200):
    """Forward selection: at each step add the candidate with the smallest Wald P < p_in."""
    selected, trace = [], []
    while True:
        best_var, best_p = None, 1.0
        for v in candidates:
            if v in selected:
                continue
            cols = selected + [v]
            try:
                m = sm.Logit(y, sm.add_constant(X[cols].astype(float))).fit(
                    disp=0, maxiter=max_iter, method="bfgs")
                p = m.pvalues.get(v, np.nan)
                if np.isfinite(p) and p < best_p:
                    best_p, best_var = p, v
            except Exception:
                continue
        if best_var is not None and best_p < p_in:
            selected.append(best_var)
            trace.append((best_var, best_p))
        else:
            break
    return selected, trace


def fit_final(X, y, cols):
    return sm.Logit(y, sm.add_constant(X[cols].astype(float))).fit(
        disp=0, maxiter=200, method="bfgs")


summary, coef_rows, results, pred_frames = [], [], [], []

queue = [("Overall", clinical)] + [(c, clinical[clinical["cohort"] == c]) for c in cohorts]

for name, sub in queue:
    y = sub["outcome"].values.astype(float)
    if name == "Overall":
        cand = [v for v in ALLPRED if v != "Other"]      # Other = reference aetiology
    else:
        cand = [v for v in ALLPRED if v not in ETIO]     # constant within a subgroup
    X = sub[cand].astype(float)

    sel, trace = forward_wald(X, y, cand)
    m = fit_final(X, y, sel)
    pred = m.predict(sm.add_constant(X[sel].astype(float)))
    a_app = roc_auc_score(y, pred)

    # Bootstrap internal validation: optimism = mean(AUC_boot - AUC_orig)
    rng = np.random.default_rng(SEED)
    opt = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(sub), len(sub), replace=True)
        try:
            mb = sm.Logit(y[idx], sm.add_constant(X.iloc[idx][sel])).fit(
                disp=0, maxiter=200, method="bfgs")
        except Exception:
            continue
        try:
            a_boot = roc_auc_score(y[idx], mb.predict(sm.add_constant(X.iloc[idx][sel])))
            a_orig = roc_auc_score(y, mb.predict(sm.add_constant(X[sel])))
            opt.append(a_boot - a_orig)
        except Exception:
            continue
    optimism = float(np.mean(opt)) if opt else np.nan

    print(f"=== {name} (n={len(sub)}, events {int(y.sum())}) ===")
    print(f"selected ({len(sel)}): {sel}")
    print(f"apparent AUC = {a_app:.4f} | optimism = {optimism:.4f} | "
          f"corrected AUC = {a_app - optimism:.4f}")
    print(f"Wald overall chi2 = {m.llr:.2f}, df = {len(sel)}, P = {m.llr_pvalue:.3e}")
    print()

    summary.append({"cohort": name, "n": len(sub), "events": int(y.sum()),
                    "n_selected": len(sel), "selected": " + ".join(sel),
                    "auc_apparent": a_app, "optimism": optimism,
                    "auc_bootstrap_corrected": a_app - optimism,
                    "llr_chi2": m.llr, "llr_df": len(sel), "llr_P": m.llr_pvalue,
                    "pseudo_R2": m.prsquared})

    ci = m.conf_int()
    for v in sel:
        coef_rows.append({"cohort": name, "variable": v, "beta": m.params[v],
                          "se": m.bse[v], "OR": np.exp(m.params[v]),
                          "OR_CI_low": np.exp(ci.loc[v, 0]), "OR_CI_high": np.exp(ci.loc[v, 1]),
                          "Wald_z": m.tvalues[v], "P_value": m.pvalues[v]})
    for v, p in trace:
        results.append({"cohort": name, "step": len(results) + 1, "variable": v, "entry_P": p})

    tmp = pd.DataFrame({"case_id": sub["case_id"].values, "outcome": y, "clin_pred": pred})
    tmp["cohort_name"] = name
    pred_frames.append(tmp)

pd.DataFrame(summary).to_csv(os.path.join(OUT, "clinical_auc_summary.csv"), index=False)
coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(os.path.join(OUT, "clinical_coefs.csv"), index=False)
pd.DataFrame(results).to_csv(os.path.join(OUT, "clinical_selection_trace.csv"), index=False)

allp = pd.concat(pred_frames, ignore_index=True)
overall = allp[allp["cohort_name"] == "Overall"][["case_id", "outcome", "clin_pred"]]
overall.to_csv(os.path.join(OUT, "clinical_pred_full.csv"), index=False)
print("-> clinical_pred_full.csv (overall model, n =", len(overall), ")")

key300 = pd.read_csv(_need("outcome_key_300.csv"))
pred300 = key300.merge(overall[["case_id", "clin_pred"]], on="case_id", how="left")
pred300.to_csv(os.path.join(OUT, "clinical_pred_300.csv"), index=False)
print(f"300-case subsample AUC = {roc_auc_score(pred300['outcome'], pred300['clin_pred']):.4f}")

print()
print("=== Final model coefficients (OR, 95% CI, P) ===")
print(coef_df.to_string(index=False))
print("BASELINE_LR_DONE")
