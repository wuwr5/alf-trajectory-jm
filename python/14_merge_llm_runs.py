# -*- coding: utf-8 -*-
"""Merge the per-run LLM outputs into the wide files the comparison consumes.

* Hy4: the three independent runs were produced interactively as
  hy4_batch1/2/3.csv (columns: case_id, run1, run2, run3, values 0-100).
  They are concatenated, deduplicated and sorted into hy4_pred_full.csv.

* DeepSeek: 13_predict_deepseek.py writes one file per run with a `risk`
  column. This script folds them into a single wide table for diagnostics
  (ds_pred_full.csv). The 11-arm comparison reads the per-run files directly,
  so this step is optional but useful for the ICC / reproducibility checks.

Both are validated: case_id sets must match the cohort, there must be no
duplicates, and the per-run distributions are printed so that a degenerate run
(an API failure that collapsed everything to one value) is immediately visible.

Input : data/hy4_batch{1,2,3}.csv
        data/ds_pred_run{1,2,3}.csv   (optional)
Output: data/hy4_pred_full.csv
        data/ds_pred_full.csv         (optional)
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
RUNS = ("run1", "run2", "run3")


def _p(name):
    return os.path.join(OUT, name)


# ==================================================================== Hy4
hy4_parts = [_p(f"hy4_batch{i}.csv") for i in (1, 2, 3)]
have_hy4 = [p for p in hy4_parts if os.path.exists(p)]

if have_hy4:
    hy4 = pd.concat([pd.read_csv(p) for p in have_hy4], ignore_index=True)
    dup = int(hy4["case_id"].duplicated().sum())
    hy4 = hy4.drop_duplicates("case_id").sort_values("case_id").reset_index(drop=True)

    # Expected cohort size. The pilot batches shipped with this project cover only
    # the stratified 300-case subsample, so a naive merge would silently overwrite
    # the 2,508-case file that the comparison actually needs. Guard against that.
    if os.path.exists(_p("llm_input_full.csv")):
        n_expected = len(pd.read_csv(_p("llm_input_full.csv")))
    else:
        n_expected = len(hy4)

    print("=== Hy4 runs ===")
    print(f"merged cases: {len(hy4)} of expected {n_expected} "
          f"(duplicate case_id dropped: {dup})")

    print()
    print("--- per-run distribution (0-100 scale) ---")
    print(hy4[list(RUNS)].describe().round(2).to_string())
    print()

    out = hy4[["case_id"] + list(RUNS)] if "case_id" in hy4.columns else hy4[list(RUNS)]

    if len(out) >= n_expected:
        tgt = _p("hy4_pred_full.csv")
        out.to_csv(tgt, index=False)
        print(f"-> hy4_pred_full.csv ({len(out)} cases)")
    else:
        tgt = _p("hy4_pred_merged.csv")
        out.to_csv(tgt, index=False)
        print(f"!! Only {len(out)} of {n_expected} cases were merged, so "
              "hy4_pred_full.csv was NOT touched.")
        print(f"!! Partial merge written to {tgt} instead.")
        print("!! hy4_batch{1,2,3}.csv as shipped cover the 300-case pilot "
              "subsample, not the full cohort. Supply the full-cohort batches "
              "to regenerate hy4_pred_full.csv (see README 'Known gaps').")
else:
    print("=== Hy4 runs ===")
    print("hy4_batch{1,2,3}.csv not found; skipping. These files come from the "
          "interactive Hy4 runs and are required for the Hy4 arm "
          "(see README 'Known gaps').")

# ================================================================ DeepSeek
ds_parts = [_p(f"ds_pred_{r}.csv") for r in RUNS]
have_ds = [p for p in ds_parts if os.path.exists(p)]

print()
if len(have_ds) == len(ds_parts):
    frames = []
    for r, p in zip(RUNS, ds_parts):
        d = pd.read_csv(p)[["case_id", "risk"]].rename(columns={"risk": r})
        frames.append(d.set_index("case_id"))
    ds = pd.concat(frames, axis=1).reset_index()
    ds = ds.sort_values("case_id").reset_index(drop=True)
    print("=== DeepSeek runs ===")
    print(f"merged cases: {len(ds)} | valid per run: "
          f"{[int(ds[r].notna().sum()) for r in RUNS]}")
    print()
    print("--- per-run distribution (0-100 scale) ---")
    print(ds[list(RUNS)].describe().round(2).to_string())
    print()
    ds.to_csv(_p("ds_pred_full.csv"), index=False)
    print("-> ds_pred_full.csv")
elif have_ds:
    print(f"=== DeepSeek runs ===\nfound {len(have_ds)}/3 run files; "
          "skipping the merged table. Re-run 13_predict_deepseek.py to complete them.")
else:
    print("=== DeepSeek runs ===\nno ds_pred_<run>.csv found; skipping.")

print()
print("MERGE_LLM_RUNS_DONE")
