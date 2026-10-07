# ALF 纵向轨迹分析：贝叶斯共享参数联合模型与轨迹表型

急性肝衰竭（ALF）ICU 患者院内死亡**动态**预测的复现代码：
从 MIMIC-IV 抽取纵向实验室指标 → 拟合贝叶斯共享参数联合模型（JMbayes2）→
Day 2 landmark 动态预测 → 基于个体随机效应的轨迹表型聚类 →
与传统统计模型、大语言模型、临床评分的**十一臂头对头比较**。

> **数据声明**：本仓库**不包含任何患者级数据**。MIMIC-IV 受 PhysioNet 数据使用协议约束，
> 使用者须自行完成 credential 认证并在本地准备数据库与队列文件。所有 `data/` 下的产物均由本仓库脚本生成。

---

## 1. 方法概览

```
MIMIC-IV (PostgreSQL)
      │  01_extract_longitudinal.py     精确 itemid 抽取，t0 = ICU 入科
      ▼
纵向长表 907,046 行 / 2,501 例 / 19 指标
      │  02_build_model_data.py         生存表 + 基线协变量
      │  03_make_wide.py                人-天宽表（tday 0–14），R 主输入
      ▼
jm_long.csv / jm_surv.csv / jm_base.csv / jm_wide.csv
      │
      ├── R/01_univariate_jm.R          3 个单变量 SPJM（胆红素 / INR / 肌酐）
      └── R/02_multivariate_jm.R        三变量共享随机效应 JM  ★推荐
      │
      │  R/03_landmark_prediction.R     单变量臂 Day2 动态预测（复用 rds）
      │  R/04_mv_repredict.R            三变量臂 Day2 动态预测（复用 rds）
      │  R/05_trajectory_clustering.R   随机效应 b̂ → k-means 轨迹表型
      ▼
jm_pred_L2_h{7,14}.csv / jm_mv_pred_L2_h{7,14}.csv / jm_mv_clusters.csv
      │
      │  python/05_calibrate_llm.py     LLM 概率再校准（Platt / isotonic，5 折 cross-fitting）
      │  python/06_clinical_scores.py   MELD / SOFA / ALFSG 评分臂
      │  python/07_compare_all_arms.py  十一臂 AUC-校准-Brier-DCA 汇总
      ▼
all_arms_summary.csv + 4 张图
```

**三个设计要点**

1. **landmark 框架**：主分析 landmark 定 Day 2（Day 0 覆盖率仅 29–35%，因入科即刻多未抽血；Day 2 达 94.5%）。
   预测目标为条件风险 `P(L < T ≤ L+h | T > L, 轨迹至 L)`，避免 immortal time bias 与终末期反向因果。
2. **风险集的精确定义**（易错点）：应排除 `outcome == 1 且 event_time_d ≤ L` 者。
   **不可**用 `event_time_d > L` 筛选——那会误剔入科 2 天内转出 ICU、结局为存活、Day 2 仍存活的患者。
3. **多变量优于"分别建模后合并"**：三条轨迹共享随机效应的三变量模型在区分度、校准、Brier 上同时优于
   三个单变量模型取均值的做法。

---

## 2. 环境

**R**（≥ 4.4，JMbayes2 建模）

```r
install.packages(c("JMbayes2", "splines", "survival", "nlme"))
# Windows 若无 Rtools，可从 CRAN 下载对应版本的 win.binary zip 后本地安装：
# install.packages("JMbayes2_0.6-0.zip", repos = NULL, type = "win.binary")
```

**Python**

```bash
pip install -r requirements.txt
```

已在 Python 3.13 + pandas 2.x / scikit-learn 1.7 / statsmodels 0.14 / R 4.4.2 + JMbayes2 0.6-0 下验证。

---

## 3. 配置

脚本路径均可通过环境变量覆盖，默认值指向仓库内的 `data/`：

| 环境变量 | 用途 | 默认 |
|---|---|---|
| `ALF_ROOT` | 仓库根 | 由脚本位置自动推断 |
| `ALF_DATA_DIR` | 数据/产物目录 | `$ALF_ROOT/data` |
| `ALF_COHORT_XLSX` | 队列文件 | `$ALF_ROOT/cohort/alf_icu_final_first_stay.xlsx` |
| `PGPASSWORD` | **数据库密码（唯一传入方式）** | 无 |

数据库主机/端口/用户在 `python/01_extract_longitudinal.py` 顶部修改，或参考 `config.example.yml`。
**密码绝不写入任何文件**，只经 `PGPASSWORD` 传入。

队列文件 `cohort/alf_icu_final_first_stay.xlsx` 需自备：每患者一行，含 `outcome`（院内死亡）、
`gender`、年龄、病因、并发症、器官支持及 MELD/SOFA。

---

## 4. 复现步骤

```bash
# (0) 准备：本机已可访问 MIMIC-IV，且已放置队列 xlsx
export PGPASSWORD=********          # 仅这一步，不要写入文件

# (1) 数据抽取与建模集构建
python python/01_extract_longitudinal.py   # 907,046 行纵向长表
python python/02_build_model_data.py       # jm_long / jm_surv / jm_base
python python/03_make_wide.py              # jm_wide（人-天宽表，R 主输入）
python python/04_link_identifiers.py       # case_id ↔ id 双射映射

# (2) 联合模型拟合（★ 耗时最长，约 45 分钟）
Rscript R/02_multivariate_jm.R             # 三变量 JM  → jm_mv_fit.rds
Rscript R/01_univariate_jm.R               # 单变量对照 → jm_fits.rds

# (3) Day2 动态预测与轨迹聚类（复用已保存的 rds，无需重跑 MCMC）
Rscript R/04_mv_repredict.R                # → jm_mv_pred_L2_h{7,14}.csv
Rscript R/03_landmark_prediction.R         # → jm_pred_L2_h{7,14}.csv
Rscript R/05_trajectory_clustering.R       # → jm_mv_clusters.csv + 肘部图/轨迹图

# (4) 对照臂与汇总
python python/05_calibrate_llm.py          # LLM Platt/isotonic 再校准
python python/06_clinical_scores.py        # MELD / SOFA / ALFSG
python python/07_compare_all_arms.py       # 十一臂 AUC-校准-Brier-DCA
```

脚本 05–07 依赖各对照臂的预测文件（`clinical_pred_full.csv`、`transformer_oof.csv`、
`hy4_pred_full.csv`、`ds_pred_run{1,2,3}.csv`）。若只需复现**联合模型与轨迹部分**，
跳过 05–07 即可；01–04 与 R/ 下全部脚本自洽。

---

## 5. 主要结果（n = 2,326，Day 2 风险集 2,331 例的全臂交集）

样本量的三个层次：**2,508**（全队列）→ **2,331**（Day 2 landmark 风险集）→ **2,326**（十一臂交集，
联合模型因 5 例纵向观测不足未能输出预测，均为存活病例）。

**十一臂区分度（+7d AUC）**

| 排名 | 臂 | AUC (95% CI) | 校准斜率 | Brier |
|---|---|---|---|---|
| 1 | 基线 LR | 0.753 (0.728–0.778) | 0.76 | 0.1731 |
| 2 | 基线 FT（Transformer） | 0.752 (0.728–0.777) | 0.76 | 0.1780 |
| 3 | ALFSG | 0.732 (0.705–0.759) | 0.96 | 0.1276 |
| 4 | **JM-mv（三变量）** | **0.722 (0.694–0.751)** | **1.02** | **0.1301** |
| 5 | Hy4（原始 / 校准） | 0.701 | 0.68 → 1.00 | 0.2748 → 0.1332 |
| 6 | JM（单变量并集） | 0.700 (0.671–0.729) | 1.34 | 0.1336 |
| 7 | DeepSeek（原始 / 校准） | 0.668 → 0.667 | 0.53 → 0.99 | 0.4834 → 0.1362 |
| 8 | SOFA | 0.632 | 0.99 | 0.1388 |
| 9 | MELD | 0.601 | 0.98 | 0.1411 |

**三条核心结论**

1. **LLM 的落后是概率标度失效，而非排序能力不足，且廉价可修复。**
   DeepSeek 校准斜率仅 0.53、Brier 0.483；Platt 校准后斜率 0.996、Brier 0.136（改善 3.5 倍），
   **而 AUC 纹丝不动（0.668 → 0.667）**。Brier 大幅改善而 AUC 不变，是"失效发生在标度环节"的诊断性证据。
2. **显式建模多指标相关演化优于分别建模后合并。**
   JM-mv 相对单变量并集：AUC +0.022，校准斜率 1.34 → 1.02（修正过度离散），Brier 0.1336 → 0.1301。
3. **轨迹表型与横断面评分正交。**
   基于随机效应 b̂ 的 k-means（K=3）得到 3 个表型，其基线 MELD 中位几乎相同
   （25.3 / 26.5 / 27.7），院内死亡率却相差近三倍（21.2% / 32.4% / **63.0%**）。
   这解释了为何 MELD 在本动态任务上 AUC 仅 0.601：它测量某一时点的状态，而非演化方向与速率。

---

## 6. 目录结构

```
alf-trajectory-jm/
├── README.md
├── LICENSE                  MIT（代码）；数据受 PhysioNet DUA 约束
├── requirements.txt
├── config.example.yml       配置模板（不含密钥）
├── .gitignore               排除 data/、*.rds、队列 xlsx
├── R/
│   ├── 01_univariate_jm.R          单变量 SPJM ×3 + 并集预测
│   ├── 02_multivariate_jm.R        三变量共享随机效应 JM
│   ├── 03_landmark_prediction.R    单变量臂 Day2 预测（复用 rds）
│   ├── 04_mv_repredict.R           三变量臂 Day2 预测（复用 rds）
│   └── 05_trajectory_clustering.R  随机效应 b̂ → 轨迹表型
├── python/
│   ├── 01_extract_longitudinal.py  MIMIC-IV 精确 itemid 抽取
│   ├── 02_build_model_data.py      生存表 + 基线协变量
│   ├── 03_make_wide.py             人-天宽表
│   ├── 04_link_identifiers.py      case_id ↔ id 复合精确匹配
│   ├── 05_calibrate_llm.py         LLM 概率再校准
│   ├── 06_clinical_scores.py       MELD / SOFA / ALFSG
│   └── 07_compare_all_arms.py      十一臂汇总 + 4 张图
├── cohort/                  放队列 xlsx（git 忽略）
└── data/                    全部产物（git 忽略，脚本生成）
```

---

## 7. 实现要点与已修复的坑

这些是本项目踩过的坑，写在 README 以免复现时重蹈：

- **`predict.jm` 报 "last available time is larger than the maximum time to predict"**
  根因是 `last_times` 取自 newdata 的**生存事件时间**而非纵向 `tday`。
  **正确写法**：newdata 的纵向只保留 `tday ≤ L`，并把生存列重置为 `event_time_d ← L, event ← 0`
  （风险集在 L 时必存活），再取 `p[p$tday == L + h, "pred_CIF"]` 即为条件累积发生率。

- **`mvglmer` 并非必需**。GLMMadaptive ≥ 0.9 移除了 `mvglmer`，但 JMbayes2 **原生支持多变量**：
  直接把 `list(fmB, fmI, fmC)`（三个 `lme` 对象）传给 `jm()` 即可，无需 Rtools 编译旧版。

- **`ranef()` 对 `jm` 对象没有方法，返回 NULL**。取个体随机效应的正确位置是
  `jm_obj$statistics$Mean$b`（矩阵 n_subj × q，行序对应 `levels(jm_obj$model_data$idT)`）。
  `model_info$frames` 是空 list，不是 lme 对象。

- **受试者 id 的类型不一致**：`jm_wide$id` 为 integer，`ranef` 行名与 `idT` 水平为字符。
  使用 `%in%` 或 `merge` 前统一 `as.character()`。

- **标志物 itemid 必须用精确映射，不可用子串匹配**：AST 子串会误伤 Gastrin/Blasts，
  Lactate 会误伤 LDH。已核实的映射见 `python/01_extract_longitudinal.py`。

---

## 8. 引用

若本代码对你的研究有帮助，请引用 MIMIC-IV 与 JMbayes2：

```bibtex
@article{johnson2023mimiciv,
  title  = {MIMIC-IV, a freely accessible electronic health record dataset},
  author = {Johnson, Alistair and Bulgarelli, Lucas and Shen, Lu and others},
  journal= {Scientific Data}, volume = {10}, number = {1}, pages = {1}, year = {2023}
}
@article{rizopoulos2024jm,
  title  = {The R package JMbayes2 for fitting joint models for longitudinal
            and time-to-event data},
  author = {Rizopoulos, Dimitris},
  journal= {Journal of Statistical Software}, volume = {108}, number = {7},
  pages  = {1--45}, year = {2024}
}
```
