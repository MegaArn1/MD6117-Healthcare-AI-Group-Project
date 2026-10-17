# Mortality presentation materials

This file describes the reviewed **11,833-stay XGBoost** experiment. Task numbers and the mortality result differ in the later combined presentation; do not mix them. See `../In_Hospital_Death.ipynb` for executed reproduction and `GAI_DECLARATION.md` for AI disclosure.

Every number below is copied from Table 1 or a named file in `results/`. Do not use numbers from any file in `trash_bin/`.

---

## The one-slide story

> Bedside scores reach AUROC 0.64. From the same 48 hours of data we reach 0.885.
> Most of the gap is closed by better features (eight well-chosen ones get to 0.82);
> the full model adds a further 0.06. The single most important thing the model
> learned is that the *last* GCS in the window predicts death and the *minimum* does not.

---

## Slide-ready tables

### Ladder (main results table)

| Rung | Model | AUROC [95% CI] | AUPRC [95% CI] | Se | PPV | Sp |
|---|---|---|---|---|---|---|
| L0 | Random | 0.474 [0.436, 0.513] | 0.135 [0.124, 0.151] | — | — | — |
| L1 | BUN max alone | 0.675 [0.635, 0.710] | 0.265 [0.231, 0.310] | — | — | — |
| L2 | SAPS-I | 0.640 | 0.213 | 0.819 | 0.171 | 0.348 |
| L2 | SOFA | 0.623 | 0.253 | 0.820 | 0.171 | 0.345 |
| L3 | Logistic, 8 feat. (clinical-prior stats) | 0.742 [0.708, 0.774] | 0.395 [0.343, 0.455] | 0.797 | 0.220 | 0.524 |
| **L3-rev** | **Logistic, 8 feat. (GCS-last)** | **0.821 [0.793, 0.846]** | **0.500 [0.445, 0.556]** | 0.840 | 0.297 | 0.666 |
| **L4** | **XGBoost, 1,474 feat.** | **0.885 [0.865, 0.903]** | **0.591 [0.535, 0.649]** | 0.785 | 0.387 | 0.791 |

Test set n = 1,775, prevalence 0.144. Se/PPV/Sp at the validation recall-0.80 threshold. L2 rows are on score-available subsets (n = 1,728 / 1,733). Full table with L1 Age/GCS rows and min(Se,PPV) column: `FINAL_REPORT.md` Table 1.

### Where the performance comes from (feature-count ablation, validation)

| Features | Val AUROC | Share of full |
|---|---|---|
| 8 | 0.813 | 93.5% |
| 50 | 0.857 | 98.6% |
| 100 | 0.861 | 99.0% |
| 1,474 | 0.869 | 100% |

Message: validation AUROC is 0.857 with 50 features versus 0.869 in the full-feature ablation; a compact model deserves independent follow-up.

### GCS: which statistic? (observed-only single-variable AUROC, train + validation, n=9,907)

| Statistic | AUROC |
|---|---|
| last | 0.757 |
| median | 0.722 |
| **min (what SOFA uses)** | **0.589** |

Message: minimum GCS is less discriminative here; sedation is one unverified explanation. Sedation and documentation may contribute; medication data are unavailable.

### Top SHAP features (test set)

| Rank | Feature | \|SHAP\| |
|---|---|---|
| 1 | GCS, last value 0–48 h | 0.694 |
| 2 | Age | 0.258 |
| 3 | Urine output, total 0–48 h | 0.141 |
| 4 | WBC slope | 0.099 |
| 5 | GCS, last value 24–48 h | 0.097 |

Figure: `results/figures/L4_shap_summary.png`. CatBoost independently ranks the same top three.

### Operating point (test, recall-0.80 threshold)

```
                    Flagged   Not flagged
Died (256)            201          55        sensitivity 0.785
Survived (1,519)      318       1,201        specificity 0.791
                   PPV 0.387
```

Flags 29% of patients; 38.7% of flagged die (base rate 14.4%, 2.7× enrichment). SAPS-I needs to flag 68% of patients to catch the same number of deaths.

### Calibration and subgroups

Brier 0.086; reliability is broadly better below ~0.35 predicted risk and over-confident at higher risks. This is internal evaluation. Figure: `results/figures/L4_calibration_curve.png`.

| ICU type | n | Deaths | AUROC |
|---|---|---|---|
| Coronary | 262 | 37 | 0.854 |
| Cardiac surgery recovery | 370 | 19 | 0.955 |
| Medical | 646 | 120 | 0.837 |
| Surgical | 497 | 80 | 0.897 |

### Challenge metric context

| | min(Se, PPV) |
|---|---|
| 2012 SAPS-I reference | 0.313 |
| 2012 winner | 0.535 |
| Ours, test, recall-0.80 threshold | 0.387 |
| Ours, validation, metric-optimal threshold | 0.518 |

Different test sets; approximate only. The metric-optimal point halves sensitivity, which we reject on clinical grounds.

---

## Figures to show

The [saved notebook output index](../results/notebook/README.md) includes previews and standalone PNG/CSV downloads. Additional plots for the data, model comparison and feature-count slides are `results/figures/notebook/eda_measurement_coverage.png`, `eda_outcomes_training.png`, `roc_pr_test.png` and `feature_count_validation.png`. `results/figures/notebook/calibration_test.png` is a freshly rendered view of the same frozen calibration data.

1. `results/figures/L4_shap_summary.png` — global importance, beeswarm.
2. `results/figures/L4_shap_waterfall_TP_high_risk_died.png` — a correct high-risk call: last GCS 7, lactate 11.6, low urine, platelets 25. Every system says the same thing.
3. `results/figures/L4_shap_waterfall_FP_high_risk_survived.png` — a false alarm: last GCS 6 dominates (+1.7 log-odds) in an 80-year-old with good urine output. Shows the GCS-last failure mode.
4. `results/figures/L4_shap_waterfall_FN_low_risk_died.png` — a missed death: normal lactate, 7 L urine, age 53. Looked like recovery at 48 h.
5. `results/figures/L4_calibration_curve.png`.

---

## Suggested mortality segment (8–10 min within the group’s 20-min maximum)

1. Clinical question and why AUPRC (prevalence 0.144, missed death ≫ false alarm)
2. Data: 11,833 stays, 1,474 features, one frozen split, test scored once
3. The ladder table
4. Where the gain comes from: feature-count ablation
5. GCS last vs min (the finding)
6. Top SHAP features + one waterfall (the TP case)
7. Operating point and what it means on a ward
8. Calibration and subgroups
9. Limitations: one cohort, no external validation, over-confident tail
10. Take-home

---

## Questions to expect

**"How does this compare with published work?"** The pooled cohort, exclusions and fixed split differ from the official challenge. Direct numerical superiority is not established. Use the documented challenge context in `REFERENCES.md` and avoid unverified literature AUROC comparisons.

**"Train AUROC is 1.0. Isn't that overfitting?"** There is a substantial train–validation gap. Validation and test support internal performance, but do not rule out overfitting or selection optimism. The separate early-stopping ablation changed validation AUROC only slightly; external validation remains necessary.

**"Is the complexity justified?"** Eight good features and logistic regression get 0.82. The full GBDT adds 0.064, with non-overlapping marginal intervals; no paired significance test was performed. The saved feature-count ablation gives validation AUROC 0.857 with 50 features versus 0.869 with all features. So: yes for the benchmark, and the deployment version should be compact.

**"How do you know GCS-last isn't leakage?"** Inputs are restricted to the pre-48h window and the statistic was selected on training/validation data. Similar charting spans and modest single-variable proxy performance are limited checks; they do not prove absence of leakage. Sedation and documentation remain unmeasured influences.

**"Why not deep learning?"** GBDT is a practical tabular-data baseline and permits SHAP interpretation. Deep sequence models were not tested, so we make no direct performance claim against them.
