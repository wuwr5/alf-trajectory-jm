# -*- coding: utf-8 -*-
"""把纵向长表按「天」聚合为宽表，供 R / JMbayes2 的 mvglmer 使用。
输出: data/jm_wide.csv（id, tday, 各 marker 列）
      data/jm_wide_core.csv（仅核心三指标，缺失更少的版本）
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import numpy as np
import pandas as pd
# --- 路径解析：支持环境变量覆盖，默认仓库内 data/ ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.environ.get("ALF_DATA_DIR", os.path.join(_ROOT, "data"))
COHORT_XLSX = os.environ.get("ALF_COHORT_XLSX",
                             os.path.join(_ROOT, "cohort", "alf_icu_final_first_stay.xlsx"))
os.makedirs(DATA, exist_ok=True)


OUT = DATA
lab = pd.read_csv(os.path.join(OUT, 'jm_long.csv'))
surv = pd.read_csv(os.path.join(OUT, 'jm_surv.csv'))
base = pd.read_csv(os.path.join(OUT, 'jm_base.csv'))

print(f"[1] 长表 {len(lab):,} 行，{lab['id'].nunique()} 例")

# 按天聚合：t_d -> 天索引（0 = ICU 第 1 天）
lab['tday'] = np.floor(lab['time_d']).astype(int)
wide = (lab.groupby(['id', 'tday', 'marker'])['log_value'].mean()
           .unstack('marker').reset_index())
print(f"[2] 按天聚合后 {len(wide):,} 行（人-天）")

# 限制在 0–14 天（联合模型主要窗口）
wide = wide[(wide['tday'] >= 0) & (wide['tday'] <= 14)]
print(f"    限制 tday 0–14: {len(wide):,} 行")

CORE = ['Bilirubin', 'INR', 'Creatinine']
EXT = ['Lactate', 'Platelet']
keep = ['id', 'tday'] + [c for c in CORE + EXT if c in wide.columns]
wide = wide[keep]

# 合并个体级不随时间变量
ind = surv[['id', 'event_time_d', 'event', 'Age', 'gender_male',
            'Liver_transplantation', 'Vasopressin', 'rrt']]
b2 = base[['id', 'MELD', 'SOFA', 'Ascites', 'Sepsis', 'HE', 'HRS', 'Shock']]
ind = ind.merge(b2, on='id', how='left')
wide = wide.merge(ind, on='id', how='left')

wide = wide.sort_values(['id', 'tday'])
wide.to_csv(os.path.join(OUT, 'jm_wide.csv'), index=False, encoding='utf-8-sig')
print(f"[3] -> data/jm_wide.csv  {len(wide):,} 行 / {wide['id'].nunique()} 例")
print("    列:", list(wide.columns))

# 每个 marker 在宽表中的缺失率
print("\n[4] 宽表中各指标缺失率（人-天层面）")
for c in CORE + EXT:
    if c in wide.columns:
        miss = wide[c].isna().mean()
        per_id = wide.groupby('id')[c].apply(lambda s: s.notna().sum())
        print(f"    {c:<12} 缺失 {100*miss:.1f}%  |  每人有值天数 中位 {per_id.median():.0f} "
              f"(P25–P75 {per_id.quantile(.25):.0f}–{per_id.quantile(.75):.0f})")

# 每人-天的事件指示（用于 JMbayes2 的 id 级生存数据已单独存放）
print("\n[5] 生存数据（id 级）:", surv.shape, "| 事件", int(surv['event'].sum()))
print("    -> R 端用 jm_wide.csv（纵向）+ jm_surv.csv（生存，id 唯一）")
print("\nDONE")
