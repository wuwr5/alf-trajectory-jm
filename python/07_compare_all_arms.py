# -*- coding: utf-8 -*-
"""全臂头对头汇总 (Day2 风险集): 传统统计 / 大模型 / 校准后大模型 / 联合模型 / 临床评分
臂: 基线LR, 基线FT, Hy4(raw), DeepSeek(raw), Hy4(Platt校准), DeepSeek(Platt校准),
    JM(单变量并集), JM-mv(三变量共享参数), MELD, SOFA, ALFSG
指标: DeLong AUC(95%CI) / Hosmer-Lemeshow / 校准斜率截距 / Brier / DCA
输出: data/all_arms_summary.csv, data/all_arms_dca.csv
      data/fig_auc_all.png, fig_calibration_all.png, fig_dca_all.png, fig_llm_calibration.png
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
from functools import reduce
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2 as chi2dist

OUT = DATA
XLSX = COHORT_XLSX
L = 2

# ---------- 真值 ----------
raw = pd.read_excel(XLSX, sheet_name="data")
raw["case_id"] = range(1, len(raw) + 1); raw["outcome"] = raw["outcome"].astype(int)
mp = pd.read_csv(os.path.join(OUT, "jm_id_to_caseid.csv"))
js = pd.read_csv(os.path.join(OUT, "jm_surv.csv")).merge(mp, on="id", how="left")
truth = raw[["case_id","outcome"]].merge(js[["case_id","event_time_d"]], on="case_id", how="left")
# Day2 landmark 风险集 (统一定义): 仅排除入科后 L 天内已死亡者。
# 不得用 event_time_d>L 筛选 —— 部分患者 <2 天转出 ICU、event=0, Day2 仍存活 (此前误剔 15 例)。
truth = truth[~((truth["outcome"] == 1) & (truth["event_time_d"] <= L))].copy()
print(f"Day2 风险集: n={len(truth)}  总院内死亡={int(truth['outcome'].sum())}")

# ---------- 各臂概率 ----------
frames = []
def add(df, cols):
    frames.append(df[cols].copy())

clin = pd.read_csv(os.path.join(OUT, "clinical_pred_full.csv")).rename(columns={"clin_pred": "p_LR"})
add(clin, ["case_id", "p_LR"])
ft = pd.read_csv(os.path.join(OUT, "transformer_oof.csv")).rename(columns={"oof_mean": "p_FT"})
add(ft, ["case_id", "p_FT"])

hy4 = pd.read_csv(os.path.join(OUT, "hy4_pred_full.csv"))
hy4["p_Hy4_raw"] = hy4[["run1","run2","run3"]].mean(axis=1) / 100.0
add(hy4, ["case_id", "p_Hy4_raw"])
d1 = pd.read_csv(os.path.join(OUT, "ds_pred_run1.csv")).rename(columns={"risk":"r1"})
d2 = pd.read_csv(os.path.join(OUT, "ds_pred_run2.csv")).rename(columns={"risk":"r2"})
d3 = pd.read_csv(os.path.join(OUT, "ds_pred_run3.csv")).rename(columns={"risk":"r3"})
ds = d1.merge(d2[["case_id","r2"]], on="case_id").merge(d3[["case_id","r3"]], on="case_id")
ds["p_DS_raw"] = ds[["r1","r2","r3"]].mean(axis=1) / 100.0
add(ds, ["case_id", "p_DS_raw"])

llc = pd.read_csv(os.path.join(OUT, "ll_calibrated.csv"))     # LLM 校准后概率 (Platt)
for h in (7, 14):
    add(llc.rename(columns={f"Hy4_pla_h{h}": f"p_Hy4_cal_h{h}", f"DeepSeek_pla_h{h}": f"p_DS_cal_h{h}"}),
        ["case_id", f"p_Hy4_cal_h{h}", f"p_DS_cal_h{h}"])

for h in (7, 14):
    f = os.path.join(OUT, f"jm_pred_L{L}_h{h}.csv")
    if os.path.exists(f):
        j = pd.read_csv(f).merge(mp, on="id", how="left").rename(columns={"prob_JM": f"p_JM_h{h}"})
        add(j, ["case_id", f"p_JM_h{h}"])
    else:
        print(f"!! 缺 {f}, 跳过 JM(单变量) h={h}")
    f2 = os.path.join(OUT, f"jm_mv_pred_L{L}_h{h}.csv")
    if os.path.exists(f2):
        j2 = pd.read_csv(f2).merge(mp, on="id", how="left").rename(columns={"prob_JMmv": f"p_JMmv_h{h}"})
        add(j2, ["case_id", f"p_JMmv_h{h}"])
    else:
        print(f"!! 缺 {f2}, 跳过 JM-mv h={h}")

sp = pd.read_csv(os.path.join(OUT, "scores_pred.csv"))
for h in (7, 14):
    add(sp.rename(columns={f"MELD_h{h}": f"p_MELD_h{h}", f"SOFA_h{h}": f"p_SOFA_h{h}",
                           f"ALFSG_h{h}": f"p_ALFSG_h{h}"}),
        ["case_id", f"p_MELD_h{h}", f"p_SOFA_h{h}", f"p_ALFSG_h{h}"])

common = reduce(lambda a, b: a.merge(b, on="case_id", how="inner"), [truth[["case_id","outcome","event_time_d"]]] + frames)
print(f"全臂合并后 n={len(common)}")

# ---------- 指标 ----------
def delong(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); m = np.isfinite(s); y = y[m]; s = s[m]
    pos = s[y==1]; neg = s[y==0]; n1, n0 = len(pos), len(neg)
    auc = (pos[:,None] > neg[None,:]).mean() + 0.5*(pos[:,None]==neg[None,:]).mean()
    V10 = (pos[:,None] >= neg[None,:]).mean(axis=1); V01 = (pos[:,None] >= neg[None,:]).mean(axis=0)
    S10 = V10-auc; S01 = V01-auc
    var = np.var(S10, ddof=1)/n1 + np.var(S01, ddof=1)/n0; se = np.sqrt(var)
    return float(auc), float(auc-1.96*se), float(auc+1.96*se)
def hl(y, p, g=10):
    br = pd.qcut(p, g, duplicates="drop")
    t = pd.DataFrame({"y":y,"p":p,"g":br}).groupby("g", observed=True).agg(n=("y","size"), ysum=("y","sum"), pm=("p","mean"))
    exp = t["n"]*t["pm"]
    num = (t["ysum"]-exp)**2/(t["pm"]*(1-t["pm"])*t["n"]+1e-9)
    c = float(num.sum()); return c, float(chi2dist.sf(c, max(g-2,1)))
def cal(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = np.isfinite(p)&(p>0)&(p<1); y=y[m]; p=p[m]
    lp = np.log(p/(1-p)).clip(-15,15)
    lr = LogisticRegression(C=1e6, max_iter=1000); lr.fit(lp.reshape(-1,1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])
def brier(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); m = np.isfinite(p)
    return float(np.mean((y[m]-p[m])**2))
def dca(y, p, thr=np.arange(0.02, 0.99, 0.02)):
    y = np.asarray(y, int); p = np.asarray(p, float); m = np.isfinite(p); y=y[m]; p=p[m]
    N = len(y); ev = y.sum()/N
    rows = [("treat_none",0.0,0.0), ("treat_all",0.0,ev-(1-ev)*0.99/0.01)]
    for t in thr:
        pp = p >= t
        tp = int((pp&(y==1)).sum()); fp = int((pp&(y==0)).sum())
        rows.append((round(float(t),2), float(t), tp/N - fp/N*(t/(1-t))))
    return pd.DataFrame(rows, columns=["label","threshold","net_benefit"])

groups = {
    "传统统计": ["基线LR","基线FT"],
    "大模型(原始)": ["Hy4(raw)","DeepSeek(raw)"],
    "大模型(Platt校准)": ["Hy4(校准)","DeepSeek(校准)"],
    "联合模型": ["JM(单变量)","JM-mv(三变量)"],
    "临床评分": ["MELD","SOFA","ALFSG"],
}
label = {"p_LR":"基线LR", "p_FT":"基线FT", "p_Hy4_raw":"Hy4(raw)", "p_DS_raw":"DeepSeek(raw)",
         "p_JM_h7":"JM(单变量)", "p_JM_h14":"JM(单变量)", "p_JMmv_h7":"JM-mv(三变量)", "p_JMmv_h14":"JM-mv(三变量)",
         "p_Hy4_cal_h7":"Hy4(校准)", "p_Hy4_cal_h14":"Hy4(校准)", "p_DS_cal_h7":"DeepSeek(校准)", "p_DS_cal_h14":"DeepSeek(校准)",
         "p_MELD_h7":"MELD", "p_MELD_h14":"MELD", "p_SOFA_h7":"SOFA", "p_SOFA_h14":"SOFA",
         "p_ALFSG_h7":"ALFSG", "p_ALFSG_h14":"ALFSG"}
grp_of = {a: g for g, arms in groups.items() for a in arms}
color = {"基线LR":"#1f77b4", "基线FT":"#9467bd", "Hy4(raw)":"#2ca02c", "DeepSeek(raw)":"#ff7f0e",
         "Hy4(校准)":"#98df8a", "DeepSeek(校准)":"#ffbb78", "JM(单变量)":"#d62728",
         "JM-mv(三变量)":"#8c564b", "MELD":"#7f7f7f", "SOFA":"#17becf", "ALFSG":"#e377c2"}
static = ["p_LR","p_FT","p_Hy4_raw","p_DS_raw"]

summary, dcas = [], []
for h in (7, 14):
    yh = ((common["outcome"]==1) & (common["event_time_d"] <= L+h)).astype(int).values
    cols = static + [c for c in common.columns if c.endswith(f"_h{h}") and c not in static]
    for c in cols:
        p = common[c].clip(1e-4, 1-1e-4).values
        m = np.isfinite(p); yy = yh[m].astype(int); pp = p[m]
        a, lo, hi = delong(yy, pp); c2, hp = hl(yy, pp); ci, cs = cal(yy, pp); br = brier(yy, pp)
        d = dca(yy, pp); d["arm"] = label[c]; d["horizon"] = h; dcas.append(d)
        summary.append({"horizon":h, "group":grp_of[label[c]], "arm":label[c], "n":int(m.sum()),
                        "events":int(yy.sum()), "AUC":round(a,3), "AUC_lo":round(lo,3), "AUC_hi":round(hi,3),
                        "HL_chi2":round(c2,2), "HL_p":round(hp,4), "cal_int":round(ci,3),
                        "cal_slope":round(cs,3), "Brier":round(br,4)})
        print(f"  h={h:2d} {label[c]:14s} AUC={a:.3f}[{lo:.3f},{hi:.3f}] cal(sl={cs:.2f},int={ci:.2f}) Brier={br:.4f} ev={int(yy.sum())}")

sdf = pd.DataFrame(summary)
sdf.to_csv(os.path.join(OUT, "all_arms_summary.csv"), index=False, encoding="utf-8-sig")
pd.concat(dcas, ignore_index=True).to_csv(os.path.join(OUT, "all_arms_dca.csv"), index=False, encoding="utf-8-sig")
print("-> all_arms_summary.csv / all_arms_dca.csv")

# ---------- 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# --- 路径解析：支持环境变量覆盖，默认仓库内 data/ ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.environ.get("ALF_DATA_DIR", os.path.join(_ROOT, "data"))
COHORT_XLSX = os.environ.get("ALF_COHORT_XLSX",
                             os.path.join(_ROOT, "cohort", "alf_icu_final_first_stay.xlsx"))
os.makedirs(DATA, exist_ok=True)

plt.rcParams.update({"font.size":10, "figure.dpi":130,
                     "font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"], "axes.unicode_minus":False})

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for i, h in enumerate((7, 14)):
    sub = sdf[sdf["horizon"]==h].sort_values("AUC")
    x = range(len(sub))
    ax[i].barh(list(x), sub["AUC"], xerr=[sub["AUC"]-sub["AUC_lo"], sub["AUC_hi"]-sub["AUC"]],
               capsize=4, color=[color[a] for a in sub["arm"]], alpha=0.9)
    ax[i].set_yticks(list(x)); ax[i].set_yticklabels(sub["arm"])
    ax[i].axvline(0.5, ls="--", c="grey"); ax[i].set_xlim(0.45, 0.85)
    ax[i].set_title(f"Day2 风险集 · 预测 Day2+{h}d 死亡 · AUC (DeLong)")
    ax[i].set_xlabel("AUC")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_auc_all.png")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, h in enumerate((7, 14)):
    axx = axes[i]
    yh = ((common["outcome"]==1) & (common["event_time_d"] <= L+h)).astype(int).values
    for c in static + [c for c in common.columns if c.endswith(f"_h{h}") and c not in static]:
        p = common[c].clip(1e-4, 1-1e-4).values; m = np.isfinite(p)
        d = pd.DataFrame({"p":p[m], "y":yh[m]}).sort_values("p")
        g = d.groupby(pd.qcut(d["p"], 10, duplicates="drop"), observed=True).agg(pp=("p","mean"), py=("y","mean"))
        axx.plot(g["pp"], g["py"], "o-", ms=3, lw=1.4, label=label[c], color=color[label[c]])
    axx.plot([0,1],[0,1],"k--",alpha=0.5); axx.set_title(f"校准曲线 · Day2+{h}d")
    axx.set_xlabel("预测风险"); axx.set_ylabel("实际事件率"); axx.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_calibration_all.png")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, h in enumerate((7, 14)):
    axx = axes[i]
    for d in dcas:
        if d["horizon"].iloc[0] == h:
            axx.plot(d["threshold"], d["net_benefit"], lw=1.4, label=d["arm"].iloc[0], color=color[d["arm"].iloc[0]])
    axx.axhline(0, color="grey", ls="--"); axx.set_ylim(-0.02, 0.45)
    axx.set_title(f"决策曲线 DCA · Day2+{h}d"); axx.set_xlabel("阈值概率"); axx.set_ylabel("净获益")
    axx.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_dca_all.png")); plt.close(fig)

# LLM 校准前后可靠性曲线 (专用图)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for i, h in enumerate((7, 14)):
    axx = axes[i]
    for arm, rc, cc in [("Hy4","p_Hy4_raw",f"p_Hy4_cal_h{h}"), ("DeepSeek","p_DS_raw",f"p_DS_cal_h{h}")]:
        yh = ((common["outcome"]==1) & (common["event_time_d"] <= L+h)).astype(int).values
        for nm, col, ls, al in [("原始", rc, "--", 0.55), ("Platt校准", cc, "-", 1.0)]:
            p = common[col].clip(1e-4, 1-1e-4).values; m = np.isfinite(p)
            d = pd.DataFrame({"p":p[m], "y":yh[m]}).sort_values("p")
            g = d.groupby(pd.qcut(d["p"], 10, duplicates="drop"), observed=True).agg(pp=("p","mean"), py=("y","mean"))
            axx.plot(g["pp"], g["py"], marker="o" if ls=="-" else "s", ms=3.5, lw=1.5, ls=ls, alpha=al,
                     label=f"{arm} {nm}", color=color[arm + "(raw)"])
    axx.plot([0,1],[0,1],"k--",alpha=0.4); axx.set_title(f"LLM 概率校准前后 · Day2+{h}d")
    axx.set_xlabel("预测风险"); axx.set_ylabel("实际事件率"); axx.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_llm_calibration.png")); plt.close(fig)
print("-> fig_auc_all.png / fig_calibration_all.png / fig_dca_all.png / fig_llm_calibration.png")
print("ALL_ARMS_DONE")
