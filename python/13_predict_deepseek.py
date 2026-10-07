# -*- coding: utf-8 -*-
"""Comparator arm 4: DeepSeek predictions, full cohort x 3 independent runs.

Anthropic-compatible endpoint, thinking disabled, single case per request,
16 concurrent workers, temperature 0. Checkpoints every 250 cases, so a
crashed or rate-limited run can be resumed by simply re-running the script.

Usage:
    DS_KEY=sk-... python python/13_predict_deepseek.py [run1 run2 run3]

Cost note: this is 2,508 cases x 3 runs = 7,524 paid API calls. The released
predictions in data/ds_pred_run{1,2,3}.csv can be used directly instead.

Input : data/ds_cards_full.csv  (from 12_make_llm_cards.py)
Output: data/ds_pred_<run>.csv  (case_id, risk)
"""
import os
import re
import sys
import threading
import time

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

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import SUFFIX_DS, SYS_DS  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

OUT = DATA

URL = os.environ.get("DS_ENDPOINT", "https://api.deepseek.com/anthropic/v1/messages")
MODEL = os.environ.get("DS_MODEL", "deepseek-flash")
KEY = os.environ.get("DS_KEY", "")
if not KEY:
    sys.exit("DS_KEY is not set. Export your DeepSeek API key and re-run.")
WORKERS = int(os.environ.get("DS_WORKERS", "16"))

H = {"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}

cards_csv = os.path.join(OUT, "ds_cards_full.csv")
if not os.path.exists(cards_csv):
    sys.exit(f"Missing {cards_csv}. Run 12_make_llm_cards.py first.")

cards = pd.read_csv(cards_csv)
if os.environ.get("DS_NMAX"):
    cards = cards.head(int(os.environ["DS_NMAX"])).reset_index(drop=True)
N = len(cards)
RUNS = sys.argv[1:] or ["run1", "run2", "run3"]

_lock = threading.Lock()
_stat = {"ok": 0, "fail": 0, "retry": 0}


def ask(text, max_tokens=128):
    """POST one case; retry on 429/5xx with exponential backoff."""
    payload = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0.0,
               "system": SYS_DS, "thinking": {"type": "disabled"},
               "messages": [{"role": "user", "content": text}]}
    for a in range(5):
        try:
            r = requests.post(URL, headers=H, json=payload, timeout=60)
        except Exception:
            with _lock:
                _stat["retry"] += 1
            time.sleep(1.5 + a * 2.5)
            continue
        if r.status_code == 200:
            j = r.json()
            return "".join(b.get("text", "") for b in j.get("content", [])
                           if b.get("type") == "text").strip()
        if r.status_code in (429, 500, 502, 503, 504, 529):
            with _lock:
                _stat["retry"] += 1
            time.sleep(1.5 + a * 3)
            continue
        return f"__HTTP{r.status_code}__"
    return "__ERR__"


def parse(txt):
    """Extract the first integer in 0-100 from the reply, or None."""
    if txt.startswith("__"):
        return None
    m = re.findall(r"\d{1,3}", txt)
    if not m:
        return None
    v = int(m[0])
    return v if 0 <= v <= 100 else None


def do_one(idx):
    cid = int(cards.loc[idx, "case_id"])
    txt = ask(str(cards.loc[idx, "card"]) + "\n" + SUFFIX_DS)
    v = parse(txt)
    with _lock:
        _stat["fail" if v is None else "ok"] += 1
    return idx, cid, v


def run_once(run):
    outp = os.path.join(OUT, f"ds_pred_{run}.csv")
    done = {}
    if os.path.exists(outp):
        old = pd.read_csv(outp)
        done = {int(r.case_id): (None if pd.isna(r.risk) else r.risk)
                for r in old.itertuples()}
        print(f"[{run}] existing checkpoint with {len(done)} cases")

    todo = [i for i in range(N)
            if int(cards.loc[i, "case_id"]) not in done
            or done[int(cards.loc[i, "case_id"])] is None]
    print(f"[{run}] {len(todo)} / {N} to predict")
    if not todo:
        return

    results = dict(done)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(do_one, i) for i in todo]
        for k, f in enumerate(as_completed(futs), 1):
            idx, cid, v = f.result()
            results[cid] = v
            if k % 250 == 0 or k == len(todo):
                el = time.time() - t0
                rate = k / el if el > 0 else 0
                print(f"[{run}] {k}/{len(todo)}  ok={_stat['ok']} fail={_stat['fail']} "
                      f"retry={_stat['retry']}  {rate:.1f} cases/s  "
                      f"{el / 60:.1f} min elapsed", flush=True)
                pd.DataFrame({"case_id": sorted(results),
                              "risk": [results[c] for c in sorted(results)]}
                             ).to_csv(outp, index=False)

    pd.DataFrame({"case_id": sorted(results),
                  "risk": [results[c] for c in sorted(results)]}
                 ).to_csv(outp, index=False)
    ok = sum(1 for v in results.values() if v is not None)
    print(f"[{run}] done: {ok}/{len(results)} valid, "
          f"{(time.time() - t0) / 60:.1f} min -> {outp}", flush=True)


if __name__ == "__main__":
    print(f"model: {MODEL} | endpoint: {URL} (thinking disabled) | workers: {WORKERS}")
    print(f"cases: {N} | runs: {RUNS}")
    print()
    tt = time.time()
    for run in RUNS:
        run_once(run)
    print(f"\nAll runs finished in {(time.time() - tt) / 60:.1f} min")
    print("stats:", _stat)
    print("PREDICT_DEEPSEEK_DONE")
