# Cohort file directory

This directory is **not committed** (see the root `.gitignore`); it holds the raw cohort file.

## Required file

| File | Source | Description |
|---|---|---|
| `alf_icu_final_first_stay.xlsx` | self-built cohort (derived from MIMIC-IV) | one row per patient, n = 2,508. Contains `outcome`, `Liver_transplantation`, `Vasopressin`, `rrt`, `Age`, etc. |

## How it is built

The cohort is constructed from MIMIC-IV (v2.x/3.x) under ALF inclusion criteria:
1. First ICU stay (earliest `intime` in `mimiciv_icu.icustays`).
2. Meets acute liver failure criteria (hepatic encephalopathy plus INR >= 1.5, or a combination
   of bilirubin and coagulopathy).
3. Age >= 18.

Because MIMIC-IV is governed by the PhysioNet Data Use Agreement, **this repository distributes no
patient-level data**. After completing PhysioNet credentialing, you can rebuild the cohort yourself
using the extraction pipeline in `python/01_extract_longitudinal.py`.

## Overriding the default path

Scripts read `cohort/alf_icu_final_first_stay.xlsx` by default. To place it elsewhere:

```bash
export ALF_COHORT_XLSX="/path/to/your/alf_icu_final_first_stay.xlsx"
```
