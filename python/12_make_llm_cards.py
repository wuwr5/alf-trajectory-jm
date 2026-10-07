# -*- coding: utf-8 -*-
"""Comparator arms 3-4: build the case cards fed to the two LLMs.

Two card formats are produced, one per model, because the two arms were run
under different input representations:

* Hy4  -- compact pipe-delimited cards, one patient per line, engineered to
         minimise token cost so that many cases fit in a single request:

             C<id>|<cohort>|<age><sex>|<8-bit comorbidity mask>|<15 labs>|
             <SpO2,MAP,MELD,SOFA>|<3-bit treatment mask>

         cohort: O=Other, A=Alcoholic_only, V=Viral_only, B=Alcoholic_plus_viral
         comorbidity order: Ascites, Sepsis, HE, HRS, EVB, SBP, Shock, Pneumonia
         lab order        : Hb, Plt, WBC, Alb, BUN, Cl, Cr, Na, ALT, AST, TBil,
                            PT, APTT, INR, Lac
         treatment order  : Liver_transplantation, Vasopressin, rrt
         missing values are rendered as "-"

* DeepSeek -- natural-language Chinese cards, one patient per line.

Neither card contains the outcome column: both LLM arms were scored blind.

The Chinese label dictionaries live in prompts.py and are intentionally left
untranslated (see the module docstring there).

Input : data/llm_input_full.csv, data/meta.json
Output: data/hy4_cards_full.txt
        data/ds_cards_full.txt, data/ds_cards_full.csv
"""
import json
import os
import sys

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

from prompts import UNIT, ZH  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

OUT = DATA

llm_csv = os.path.join(OUT, "llm_input_full.csv")
meta_json = os.path.join(OUT, "meta.json")
for p in (llm_csv, meta_json):
    if not os.path.exists(p):
        sys.exit(f"Missing {p}. Run 08_prepare_cohort.py first.")

full = pd.read_csv(llm_csv)
with open(meta_json, encoding="utf-8") as fh:
    meta = json.load(fh)

COMP = meta["COMP"]
LAB = meta["LAB"]
TREAT = meta["TREAT"]
COH = {"Other": "O", "Alcoholic_only": "A", "Viral_only": "V",
       "Alcoholic_plus_viral": "B"}


# =============================================================== Hy4: compact
def num_compact(v):
    if pd.isna(v):
        return "-"
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def card_compact(r):
    cid = int(r["case_id"])
    coh = COH[r["cohort"]]
    sex = "M" if r["gender_male"] == 1 else "F"
    comp = "".join("1" if r[c] == 1 else "0" for c in COMP)
    lab = ",".join(num_compact(r[c]) for c in LAB)
    vit = ",".join(num_compact(r[c]) for c in ["SpO2", "MAP", "MELD", "SOFA"])
    tx = "".join("1" if r[c] == 1 else "0" for c in TREAT)
    return f"C{cid}|{coh}|{int(r['Age'])}{sex}|{comp}|{lab}|{vit}|{tx}"


lines = [card_compact(r) for _, r in full.iterrows()]
txt = "\n".join(lines)
with open(os.path.join(OUT, "hy4_cards_full.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt)

print("=== Hy4 compact cards ===")
print("cases:", len(lines))
print("characters:", len(txt), "| estimated tokens (chars/3):", len(txt) // 3)
print("characters per case:", round(len(txt) / len(lines), 1))
print()
print("first 3 lines:")
print("\n".join(lines[:3]))
print()
print("field order -- comp:", ",".join(COMP))
print("field order -- lab :", ",".join(LAB))
print("field order -- vit : SpO2,MAP,MELD,SOFA")
print("field order -- tx  :", ",".join(TREAT))
print()


# ==================================================== DeepSeek: natural language
# The format literals below (岁 / 男 / 女 / 无 / 缺失 / ...) are part of the stimulus
# that was actually sent to the model, so they are kept verbatim for the same reason
# as prompts.py -- translating them would break the link between the released
# predictions and this code. Verified: regenerating with this script reproduces
# data/ds_cards_full.csv byte for byte (2,508/2,508 rows).


def num_nl(v, nd=1):
    if pd.isna(v):
        return None
    if float(v) == int(v):
        return str(int(v))
    return f"{v:.{nd}f}"


def card_nl(r):
    cid = int(r["case_id"])
    sex = "男" if r["gender_male"] == 1 else "女"
    etio = [ZH[c] for c in ["Alcoholic_only", "Viral_only",
                            "Alcoholic_plus_viral", "Other"] if r[c] == 1]
    etio = etio[0] if etio else "未分类"
    comp = [ZH[c] for c in COMP if r[c] == 1]
    tx = [ZH[c] for c in TREAT if r[c] == 1]

    labs = []
    for c in LAB:
        v = num_nl(r[c])
        if v is None:
            labs.append(f"{ZH[c]}缺失")
        else:
            u = UNIT[c]
            labs.append(f"{ZH[c]}{v}{u}" if u else f"{ZH[c]}{v}")

    vit = []
    for c in ["SpO2", "MAP", "MELD", "SOFA"]:
        v = num_nl(r[c])
        vit.append(f"{ZH[c]}{v}{UNIT[c]}" if v is not None else f"{ZH[c]}缺失")

    return (f"[C{cid}] {int(r['Age'])}岁{sex}，急性肝衰竭（{etio}）。"
            f"合并症：{'、'.join(comp) if comp else '无'}。"
            f"检验：{'，'.join(labs)}。"
            f"生命体征与评分：{'，'.join(vit)}。"
            f"治疗：{'、'.join(tx) if tx else '无'}。")


nl_lines = [card_nl(r) for _, r in full.iterrows()]
with open(os.path.join(OUT, "ds_cards_full.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(nl_lines))

n = len(nl_lines)
L = sum(len(x) for x in nl_lines)
print("=== DeepSeek natural-language cards ===")
print("cases:", n, "| characters:", L, "| per case:", round(L / n, 1))
print("estimated tokens (Chinese ~1.2 chars/token):", int(L / 1.2))
print()
print("first 2 cases:")
print(nl_lines[0])
print()
print(nl_lines[1])
print()

pd.DataFrame({"case_id": full["case_id"].astype(int), "card": nl_lines}).to_csv(
    os.path.join(OUT, "ds_cards_full.csv"), index=False)
print("-> hy4_cards_full.txt, ds_cards_full.txt, ds_cards_full.csv")
print("MAKE_LLM_CARDS_DONE")
