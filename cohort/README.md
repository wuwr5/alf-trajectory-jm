# 队列文件放置目录

本目录**不入库**（见根目录 `.gitignore`），用于放置原始队列文件。

## 需要放置的文件

| 文件 | 来源 | 说明 |
|---|---|---|
| `alf_icu_final_first_stay.xlsx` | 自建队列（MIMIC-IV 衍生） | 每患者一行，n=2,508。含 `outcome`、`Liver_transplantation`、`Vasopressin`、`rrt`、`Age` 等列 |

## 生成方式

该队列由 MIMIC-IV（v2.x/3.x）按 ALF 入组标准构建：
1. 首次 ICU 入住（`mimiciv_icu.icustays` 取 `intime` 最早一条）
2. 满足急性肝衰竭判定（肝性脑病 + INR ≥ 1.5，或胆红素与凝血障碍组合）
3. 年龄 ≥ 18

因 MIMIC-IV 受 PhysioNet 数据使用协议（DUA）约束，**本仓库不分发任何患者级数据**。
完成 PhysioNet 认证后，可按上述标准在 `python/01_extract_longitudinal.py` 的数据抽取链路
上自行重建队列。

## 覆盖默认路径

脚本默认读取 `cohort/alf_icu_final_first_stay.xlsx`。如需放到别处：

```bash
export ALF_COHORT_XLSX="/path/to/your/alf_icu_final_first_stay.xlsx"
```
