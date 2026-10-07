# -*- coding: utf-8 -*-
"""重建 case_id (1..N 行序, 队列 xlsx) 与 id (MIMIC 派生标识符) 之间的一一映射。

背景
----
本项目的各臂使用两套受试者标识：
  * 队列 xlsx 与全部 LLM / 传统模型臂：1..N 的行序号 case_id
  * 从 PostgreSQL 抽取的纵向与生存数据：MIMIC 派生的 id

二者无直接键可连，必须通过"复合精确匹配"重建双射。本脚本以
(年龄, 性别, 院内死亡, 肝移植, 血管加压素, 肾脏替代治疗) 为匹配键，
先做严格唯一匹配，再对残余个案用 (Age, MELD, SOFA) 三元组补齐。

产出
----
data/jm_id_to_caseid.csv : id, case_id, 以及参与匹配的字段

用法
----
    python python/04_link_identifiers.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# --- 路径解析：支持环境变量覆盖，默认仓库内 data/ ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.environ.get("ALF_DATA_DIR", os.path.join(_ROOT, "data"))
COHORT_XLSX = os.environ.get(
    "ALF_COHORT_XLSX",
    os.path.join(_ROOT, "cohort", "alf_icu_final_first_stay.xlsx"),
)
os.makedirs(DATA, exist_ok=True)

# ---------------------------------------------------------------- 输入
raw = pd.read_excel(COHORT_XLSX, sheet_name="data")
raw["case_id"] = range(1, len(raw) + 1)
raw["outcome"] = raw["outcome"].astype(int)
raw["gender_male"] = (raw["gender"] == "M").astype(int)

base = pd.read_csv(os.path.join(DATA, "jm_base.csv"))          # 含 id 与协变量
print("队列 xlsx:", raw.shape, " MIMIC 侧 jm_base:", base.shape)

# ---------------------------------------------------------------- 复合精确匹配
# 匹配键取两侧都存在的字段。gender_male 仅 xlsx 侧有，MIMIC 侧 jm_base 无此列，
# 故用作可选增强键（存在则纳入，不存在则跳过）。
BASE_KEY = ["Age", "outcome", "Liver_transplantation", "Vasopressin", "rrt"]
OPTIONAL_KEY = ["gender_male"]


def norm(df, cols, optional=()):
    out = df.copy()
    keep = []
    for c in cols:
        if c not in out.columns:
            raise KeyError(f"缺列 {c}; 现有: {list(out.columns)[:20]}")
        keep.append(c)
    for c in optional:
        if c in out.columns:
            keep.append(c)
    return out, keep


raw_n, keys_r = norm(raw, BASE_KEY, OPTIONAL_KEY)
bas_n, keys_b = norm(base, BASE_KEY, OPTIONAL_KEY)
# 仅用两侧共有的键构造匹配串
KEYS = [k for k in keys_r if k in keys_b]
print("匹配键:", KEYS)


# 用浮点数容差比较 Age（两侧精度可能不同）
def key_of(df):
    parts = []
    for c in KEYS:
        v = df[c]
        parts.append(v.round(4).astype(str) if v.dtype.kind == "f"
                     else v.astype(int).astype(str))
    return parts[0].str.cat(parts[1:], sep="|") if len(parts) > 1 else parts[0]

raw_n["_k"] = key_of(raw_n)
bas_n["_k"] = key_of(bas_n)

# 仅保留两侧键唯一的部分，避免多对多
rc = raw_n["_k"].value_counts()
bc = bas_n["_k"].value_counts()
uniq = set(rc[rc == 1].index) & set(bc[bc == 1].index)

m1 = (raw_n[raw_n["_k"].isin(uniq)][["case_id", "_k"]]
      .merge(bas_n[bas_n["_k"].isin(uniq)][["id", "_k"]], on="_k", how="inner")
      .drop(columns="_k"))
print(f"阶段1 严格唯一匹配: {len(m1)} 例")

# ---------------------------------------------------------------- 残余个案：Age+MELD+SOFA 补齐
done_x = set(m1["case_id"]); done_b = set(m1["id"])
r2 = raw_n[~raw_n["case_id"].isin(done_x)].copy()
b2 = bas_n[~bas_n["id"].isin(done_b)].copy()

m2 = pd.DataFrame(columns=["case_id", "id"])
if len(r2) and len(b2) and {"MELD", "SOFA"}.issubset(r2.columns) and {"MELD", "SOFA"}.issubset(b2.columns):
    def k3(df):
        return (df["Age"].round(3).astype(str) + "|" +
                df["MELD"].round(3).astype(str) + "|" +
                df["SOFA"].round(3).astype(str))
    r2["_k"] = k3(r2); b2["_k"] = k3(b2)
    rc2 = r2["_k"].value_counts(); bc2 = b2["_k"].value_counts()
    u2 = set(rc2[rc2 == 1].index) & set(bc2[bc2 == 1].index)
    m2 = (r2[r2["_k"].isin(u2)][["case_id", "_k"]]
          .merge(b2[b2["_k"].isin(u2)][["id", "_k"]], on="_k", how="inner")
          .drop(columns="_k"))
    print(f"阶段2 Age+MELD+SOFA 补齐: {len(m2)} 例")

mp = pd.concat([m1, m2], ignore_index=True).drop_duplicates(subset=["case_id"]).drop_duplicates(subset=["id"])

# ---------------------------------------------------------------- 校验
n_raw, n_base = len(raw), len(base)
ok = (len(mp) == n_raw == n_base) and mp["case_id"].is_unique and mp["id"].is_unique
print(f"\n映射结果: {len(mp)} 条  (xlsx {n_raw} / MIMIC {n_base})  双射一致={ok}")
if not ok:
    print("!! 非完美双射，请检查键定义或队列构成（残留未匹配个案如下）")
    print("  未匹配 case_id 数:", n_raw - mp["case_id"].nunique())
    print("  未匹配 id 数:", n_base - mp["id"].nunique())

out = mp.merge(base[["id"] + [c for c in ["Age", "MELD", "SOFA", "outcome"] if c in base.columns]],
               on="id", how="left")
out.to_csv(os.path.join(DATA, "jm_id_to_caseid.csv"), index=False, encoding="utf-8-sig")
print("-> jm_id_to_caseid.csv")
