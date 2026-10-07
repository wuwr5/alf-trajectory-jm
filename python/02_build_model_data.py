# -*- coding: utf-8 -*-
"""构建联合模型建模数据集（JMbayes2 友好格式）+ landmark 宽表。
输入: data/long_labs_clean.csv, data/long_surv.csv, 桌面原始 xlsx
输出:
    data/jm_long.csv        多变量纵向长表（id, marker, time_d, value, log_value）
    data/jm_surv.csv        生存数据（id, time_d, event, transplant）
    data/jm_base.csv        基线协变量（id, 人口学/并发症/治疗/评分）
    data/jm_landmark_L2.csv landmark Day2 宽表（用于 JM vs 基线模型对照）
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
XLSX = COHORT_XLSX

# 生理合理范围（超出置为缺失，防单位/录入错误）
RANGE = {
    'Bilirubin': (0.05, 80), 'INR': (0.5, 25), 'Creatinine': (0.05, 30),
    'Lactate': (0.1, 40), 'Platelet': (1, 1500), 'ALT': (1, 20000),
    'AST': (1, 20000), 'Albumin': (0.5, 8), 'BUN': (1, 200),
    'Sodium': (100, 180), 'WBC': (0.1, 200), 'Hemoglobin': (1, 25),
}
CORE = ['Bilirubin', 'INR', 'Creatinine']          # JM 纵向子模型核心
EXT = ['Lactate', 'Platelet']                       # 可选扩展

# ---------------------------------------------------------------- 读数据
lab = pd.read_csv(os.path.join(OUT, 'long_labs_clean.csv'))
surv = pd.read_csv(os.path.join(OUT, 'long_surv.csv'))
raw = pd.read_excel(XLSX, sheet_name='data')
print(f"[1] 纵向 {len(lab):,} 行 | 生存 {len(surv):,} 例 | 原始 {len(raw):,} 例")

# ---------------------------------------------------------------- 清洗
n0 = len(lab)
lab = lab[lab['t_d'] >= 0]                                  # 只保留 ICU 入科后
for m, (lo, hi) in RANGE.items():
    bad = (lab['marker'] == m) & ((lab['valuenum'] < lo) | (lab['valuenum'] > hi))
    if bad.sum():
        lab = lab[~bad]
        print(f"    {m}: 剔除超范围 {int(bad.sum())} 条")
print(f"    范围清洗后 {len(lab):,} 行（原 {n0:,}）")

# 同一 (id, marker, 时间) 重复 -> 取均值
lab = (lab.groupby(['hadm_id', 'marker', 't_d'], as_index=False)['valuenum'].mean()
          .rename(columns={'hadm_id': 'id', 't_d': 'time_d'}))
print(f"    去重后 {len(lab):,} 行")

# 对数变换（胆红素/INR/肌酐/乳酸/ALT/AST 右偏）
lab['log_value'] = np.where(lab['marker'].isin(['Bilirubin', 'INR', 'Creatinine',
                                                'Lactate', 'ALT', 'AST']),
                            np.log(lab['valuenum']), lab['valuenum'])
lab = lab.sort_values(['id', 'marker', 'time_d'])
lab.to_csv(os.path.join(OUT, 'jm_long.csv'), index=False, encoding='utf-8-sig')
print(f"    -> data/jm_long.csv  {len(lab):,} 行，{lab['id'].nunique()} 例")

# ---------------------------------------------------------------- 生存数据
s = surv[['hadm_id', 'event_time_d', 'event']].rename(columns={'hadm_id': 'id'})
s = s.merge(raw[['hadm_id', 'Liver_transplantation', 'Vasopressin', 'rrt',
                 'Age', 'gender']], left_on='id', right_on='hadm_id', how='left')
s['gender_male'] = (s['gender'] == 'M').astype(int)
s = s.drop(columns=['hadm_id', 'gender'])
s.loc[s['event_time_d'] <= 0, 'event_time_d'] = 0.05
s = s.sort_values('id')
s.to_csv(os.path.join(OUT, 'jm_surv.csv'), index=False, encoding='utf-8-sig')
print(f"[2] -> data/jm_surv.csv  n={len(s)}，事件 {int(s['event'].sum())} "
      f"({100*s['event'].mean():.1f}%)，肝移植 {int(s['Liver_transplantation'].sum())} 例")

# ---------------------------------------------------------------- 基线协变量
base_cols = ['hadm_id', 'Age', 'gender_male', 'Alcoholic_plus_viral', 'Alcoholic_only',
             'Viral_only', 'Other', 'Ascites', 'Sepsis', 'HE', 'HRS', 'EVB', 'SBP',
             'Shock', 'Pneumonia', 'MELD', 'SOFA', 'Liver_transplantation',
             'Vasopressin', 'rrt', 'outcome']
b = raw[[c for c in base_cols if c in raw.columns]].rename(columns={'hadm_id': 'id'})
b.to_csv(os.path.join(OUT, 'jm_base.csv'), index=False, encoding='utf-8-sig')
print(f"[3] -> data/jm_base.csv  n={len(b)}，{len(b.columns)-1} 个协变量")

# ---------------------------------------------------------------- Landmark 宽表
print("\n[4] Landmark 宽表（用于 JM 与静态模型公平对照）")
def landmark_table(L, window_lo=0.0):
    """取 t_d ∈ [window_lo, L] 内每个 marker 的：首次值、末次值、均值、斜率、测量次数"""
    sub = lab[(lab['time_d'] >= window_lo) & (lab['time_d'] <= L)]
    rows = []
    for m in CORE + EXT:
        s2 = sub[sub['marker'] == m].sort_values(['id', 'time_d'])
        g = s2.groupby('id')
        agg = pd.DataFrame({
            f'{m}_first': g['valuenum'].first(),
            f'{m}_last': g['valuenum'].last(),
            f'{m}_mean': g['valuenum'].mean(),
            f'{m}_n': g['valuenum'].size(),
            f'{m}_t_first': g['time_d'].first(),
            f'{m}_t_last': g['time_d'].last(),
        })
        # 简易斜率（(末-首)/(时间差)），时间差<0.5d 则置 0
        dt = agg[f'{m}_t_last'] - agg[f'{m}_t_first']
        agg[f'{m}_slope'] = np.where(dt >= 0.5, (agg[f'{m}_last'] - agg[f'{m}_first']) / dt, 0.0)
        rows.append(agg.drop(columns=[f'{m}_t_first', f'{m}_t_last']))
    w = pd.concat(rows, axis=1)
    return w

for L in [1, 2, 3, 5]:
    w = landmark_table(L)
    at_risk = set(s.loc[s['event_time_d'] > L, 'id'])
    w = w.reindex(sorted(at_risk))
    cov = w[[f'{m}_n' for m in CORE]].notna().all(axis=1).mean()
    print(f"    L={L}d: 风险集 n={len(w)}，核心三指标齐全 {100*cov:.1f}%")
    w.to_csv(os.path.join(OUT, f'jm_landmark_L{L}.csv'), encoding='utf-8-sig')

# 主分析 landmark = Day 2
w2 = landmark_table(2).reindex(sorted(set(s.loc[s['event_time_d'] > 2, 'id'])))
extra = raw.set_index('hadm_id')[['MELD', 'SOFA', 'Ascites', 'Sepsis', 'HE', 'HRS',
                                  'Shock', 'Liver_transplantation', 'Vasopressin', 'rrt']]
w2 = w2.join(s.set_index('id')[['event_time_d', 'event', 'Age', 'gender_male']]).join(extra)
w2.to_csv(os.path.join(OUT, 'jm_landmark_L2.csv'), encoding='utf-8-sig')
print(f"    主分析 L=2d 宽表 -> data/jm_landmark_L2.csv  n={len(w2)}")

# ---------------------------------------------------------------- 汇总
print("\n" + "=" * 72)
print("[5] 建模就绪检查")
print(f"    纵向长表   : {len(lab):,} 行 / {lab['id'].nunique()} 例 / {lab['marker'].nunique()} 指标")
for m in CORE:
    sub = lab[lab['marker'] == m]
    cnt = sub.groupby('id').size()
    print(f"      {m:<12} 覆盖 {100*len(cnt)/2508:.1f}%  中位 {cnt.median():.0f} 次  "
          f"P25–P75 {cnt.quantile(.25):.0f}–{cnt.quantile(.75):.0f}")
print(f"    生存数据   : n={len(s)}，事件 {int(s['event'].sum())}，"
      f"事件时间中位 {s.loc[s['event']==1,'event_time_d'].median():.2f}d，"
      f"删失中位 {s.loc[s['event']==0,'event_time_d'].median():.2f}d")
print(f"    肝移植     : {int(s['Liver_transplantation'].sum())} 例（竞争风险，需处理）")
print("\nDONE")
