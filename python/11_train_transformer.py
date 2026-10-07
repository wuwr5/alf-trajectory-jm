# -*- coding: utf-8 -*-
"""Comparator arm 2b: FT-Transformer out-of-fold predictions.

Runs 5-fold cross-validation across three random seeds and averages the
out-of-fold predictions, which is the probability the 11-arm comparison uses.

This script drives the model implementation that lives in a separate repository:
    https://github.com/wuwr5/alf-transformer-mimic
Clone it and point ALF_TRANSFORMER_REPO at the checkout, or place it next to
this repository as ./repo_alf_transformer.

Usage:
    python python/11_train_transformer.py [arch] [n_folds] [seeds...]
    python python/11_train_transformer.py ft 5 42 7 2024   # full run
    python python/11_train_transformer.py ft 2 42          # smoke test

On Anaconda builds set KMP_DUPLICATE_LIB_OK=TRUE: torch and MKL both ship an
OpenMP runtime and will abort on conflict otherwise.

Input : data/transformer_cohort.csv
Output: data/transformer_oof.csv   (case_id, outcome, oof_mean, seed42, seed7, seed2024)
"""
import os
import sys
import time

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

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

# --- Locate the alf_transformer package -------------------------------------
REPO = os.environ.get(
    "ALF_TRANSFORMER_REPO",
    os.path.join(_ROOT, "repo_alf_transformer"),
)
if not os.path.exists(os.path.join(REPO, "src", "alf_transformer")):
    sys.exit(
        f"Cannot find the alf_transformer package under: {REPO}\n"
        "  Clone https://github.com/wuwr5/alf-transformer-mimic and set\n"
        "  ALF_TRANSFORMER_REPO=<path>, or place it at ./repo_alf_transformer."
    )
sys.path.insert(0, os.path.join(REPO, "src"))

from alf_transformer import DataSpec, load_dataset, run_seeds  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

torch.set_num_threads(min(12, os.cpu_count() or 4))

ARCH = sys.argv[1] if len(sys.argv) > 1 else "ft"
N_FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEEDS = tuple(int(s) for s in sys.argv[3:]) or (42, 7, 2024)

cohort_csv = os.path.join(DATA, "transformer_cohort.csv")
if not os.path.exists(cohort_csv):
    sys.exit(f"Missing {cohort_csv}. Run 10_prep_transformer_data.py first.")

spec = DataSpec(outcome="outcome")
ds = load_dataset(cohort_csv, None, spec)

print("=" * 60)
print("ALF FT-Transformer training")
print("=" * 60)
print(f"architecture: {ARCH} | folds: {N_FOLDS} | seeds: {SEEDS}")
print(f"n = {len(ds)} | events = {int(ds.y.sum())} ({ds.y.mean():.4f})")
print(f"numeric features: {ds.x_num.shape[1]} | binary features: {ds.x_bin.shape[1]} "
      f"| trajectory: {ds.x_traj.shape}")
print(f"torch {torch.__version__} | threads {torch.get_num_threads()}")
print("=" * 60)

t0 = time.time()
oof, mat = run_seeds(ds, ARCH, seeds=SEEDS, n_folds=N_FOLDS, verbose=True)
el = time.time() - t0

print()
print(f"total elapsed: {el / 60:.1f} min")
print("OOF NaNs:", int(np.isnan(oof).sum()))
print(f"OOF AUC (mean of three seeds): {roc_auc_score(ds.y, oof):.4f}")
for i, s in enumerate(SEEDS):
    print(f"  seed {s}: AUC = {roc_auc_score(ds.y, mat[:, i]):.4f} | "
          f"mean predicted risk = {mat[:, i].mean():.4f}")

out = pd.DataFrame({"case_id": np.arange(1, len(ds) + 1),
                    "outcome": ds.y.astype(int),
                    "oof_mean": oof})
for i, s in enumerate(SEEDS):
    out[f"seed{s}"] = mat[:, i]

if N_FOLDS == 5 and len(SEEDS) == 3:
    tgt = os.path.join(DATA, "transformer_oof.csv")
    out.to_csv(tgt, index=False)
    print(f"\n-> {tgt}")
else:
    tgt = os.path.join(DATA, "transformer_oof_smoke.csv")
    out.to_csv(tgt, index=False)
    print(f"\n(smoke test) -> {tgt}")
print("TRAIN_TRANSFORMER_DONE")
