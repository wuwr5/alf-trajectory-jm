# Methods

## 1. Cohort and time origin

- Adults meeting ALF diagnostic criteria, taking the **first ICU stay**: n = 2,508,
  in-hospital deaths 962 (38.4%), liver transplantation 104 (4.1%).
- **Time origin t0 = ICU admission** (not hospital admission). Median time to death is 8.0 days
  from hospital admission but 6.34 days from ICU admission -- **these must not be mixed up**.
- Longitudinal markers are taken from day 0 to day 30 after ICU admission; the primary modelling
  window is days 0-14.

## 2. Longitudinal markers

The primary analysis uses **bilirubin, INR, and creatinine**. Rationale:

| Marker | Median measurements, first 7 days | Coverage | Proportion with >=4 |
|---|---|---|---|
| Bilirubin | 7 | 98.6% | 82.7% |
| INR | 8 | 98.9% | 89.0% |
| Creatinine | 12 | 99.7% | 95.9% |
| Lactate | 7 | 90.2% | -- |

Lactate is missing in 57.2% of person-day rows and is therefore excluded from the main model.

**Itemids must use exact mapping** (never substring matching): 50885 bilirubin, 51237 INR,
50912 creatinine, 50813 lactate, 51265 platelets, 50861 ALT, 50878 AST, 50862 albumin,
51006 BUN, 50983 sodium, 51222 Hb, 51301 WBC, 51274 PT, 51275 PTT (MIMIC-IV calls APTT "PTT").
All restricted to `fluid = 'Blood'`.
Must be excluded: 50954 (LDH), 50928 (Gastrin), 51148 (Blasts), 52190, 51240, 50811, 50814.

## 3. Landmark framework

- **Primary landmark L = 2 days**. Day 0 coverage is only 29-35% (blood is rarely drawn right at
  ICU admission), Day 1 reaches 89.2%, **Day 2 reaches 94.5% (all three core markers available)**,
  Day 5 reaches 98.1%.
- Risk set definition (**easy to get wrong**):

  ```
  Include: ~(outcome == 1 & event_time_d <= L)
  ```

  That is, exclude only those who died within L days of ICU admission.
  Do **not** write `event_time_d > L` -- that additionally drops patients who left the ICU within
  2 days, have `event = 0`, and are still alive at Day 2.
  The two definitions differ by 15 subjects in this cohort (2,331 vs 2,316).

- Endpoint: `y = (outcome == 1) & (event_time_d <= L + h)`, h in {7, 14}.

## 4. Joint models

Cox survival submodel: `Surv(event_time_d, event) ~ Age_c + gender_male + MELD_c`
(age centred at 60 and scaled per 10 years; MELD centred at 20 and scaled per 5 points).

Longitudinal submodel (per marker):

```
marker ~ ns(tday, 3) * Age_c + ns(tday, 3) * MELD_c + ns(tday, 3) * gender_male
random = ~ ns(tday, 3) | id
```

- **Univariate arm**: one JM per marker; the Day-2 conditional risk is the mean of the three.
  This implicitly assumes the three trajectories evolve independently within a subject.
- **Trivariate arm (JM-mv)**: `jm(CoxFit, list(fmB, fmI, fmC), time_var = "tday")`, where the three
  trajectories share correlated random effects.
  MCMC: 3 chains x 3000 iterations, burn-in 1500, thin 2, seed 2024.

## 5. Dynamic prediction

Conditional cumulative incidence `P(L < T <= L + h | T > L, trajectory up to L)`,
obtained from `predict.jm`:

```r
nd <- wide[wide$tday <= L & id %in% risk$id, ]
nd$event_time_d <- L          # reset the survival origin to the landmark
nd$event        <- 0
p  <- predict(jm_fit, newdata = nd, process = "event", times = L + h, return_newdata = TRUE)
cif <- p[p$tday == L + h, "pred_CIF"]
```

## 6. Trajectory phenotypes

Standardised posterior means of the subject-specific random effects b_hat (12-dimensional for the
trivariate model) are clustered with k-means; K is chosen by the elbow method (K = 3 in this
cohort). Clustering operates on the **random effects** rather than the raw trajectories because
they are a parametrically compressed summary, produced by the model, of the direction and speed of
each subject's deviation from the population mean, and therefore carry less noise.

## 7. Comparator arms

All comparator arms share the same cohort, the same Day-2 risk set, and the same endpoint, so
that the only thing varying across arms is the prediction strategy.

| Arm | Implementation | Out-of-sample estimate |
|---|---|---|
| Baseline LR | forward stepwise logistic regression, Wald entry P < 0.05 | 500-replicate bootstrap optimism correction |
| Baseline FT | FT-Transformer, cross-sectional features | 5-fold CV x 3 seeds, out-of-fold, averaged |
| Hy4 | compact pipe-delimited cards, 3 independent runs | 3-run mean |
| DeepSeek | natural-language cards, temperature 0, thinking disabled | 3-run mean |
| MELD / SOFA | published scores, converted to probabilities | 5-fold cross-fitted logistic regression |
| ALFSG | US-ALFSG variable structure (HE grade, log bilirubin, log INR, vasopressin, aetiology); coefficients re-estimated in this cohort because the source paper publishes no portable coefficients | 5-fold cross-fitted logistic regression |

Two representational notes that matter for interpreting the comparison:

- **The two LLMs did not receive identical input.** Hy4 was given compact pipe-delimited cards
  designed to keep token cost low; DeepSeek was given natural-language cards. This is a
  deliberate consequence of how each arm was run, not an oversight, but it means the Hy4-versus-
  DeepSeek contrast conflates *model* with *input representation*.
- **Both LLM arms were scored blind**: neither card contains the outcome column.

## 8. Evaluation metrics

- **Discrimination**: DeLong AUC with 95% CI (normal approximation based on the Sun-Xu variance).
- **Calibration**: calibration intercept and slope from an unpenalised logistic regression of the
  outcome on `logit(p)`; Hosmer-Lemeshow as a supplement.
  At n around 2,300 the HL test is extremely sensitive to trivial deviations (most arms give
  p about 0), so **slope and intercept are the primary criteria**.
- **Overall performance**: Brier score.
- **Clinical utility**: decision curve analysis (DCA), thresholds 0.02-0.98 in steps of 0.02.
- Probabilistic calibration of the LLM and score arms uses **5-fold cross-fitting** to avoid
  optimistic bias.

## 9. Known limitations

1. Single centre (MIMIC-IV), no external validation.
2. The 104 liver transplantations are treated as censored; they are not modelled explicitly as a
   competing risk.
3. Early deaths (36.6% die within 4 days) have a very short usable longitudinal window, so dynamic
   prediction adds little for them; results should be interpreted in strata.
4. The good calibration of the score arms is a product of in-cohort cross-fitted probabilisation,
   not an inherent property of the raw scores.
5. The 5 subjects for which the joint model produced no prediction (insufficient longitudinal
   observations, all survivors) were excluded; this has no material effect on mortality estimates.
6. **The Hy4 arm is not script-reproducible.** Those predictions were generated interactively
   through a chat interface and no system prompt was persisted, so the released probabilities
   cannot be regenerated byte-identically from this repository. The DeepSeek arm *is* script-
   driven and its prompt is preserved verbatim in `python/prompts.py`.
7. **Re-running either LLM arm is a fresh replicate, not a verification.** Commercial endpoints
   are not bit-reproducible across time even at temperature 0.
8. **The two LLM arms differ in input representation as well as in model** (compact cards for Hy4,
   natural-language cards for DeepSeek), so their head-to-head AUC gap should not be read as a
   pure model comparison.
