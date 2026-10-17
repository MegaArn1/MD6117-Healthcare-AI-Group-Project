# In-Hospital Mortality Prediction from 48 Hours of ICU Data

**Course**: MD6117 Machine Learning for Healthcare AI
**Task**: In-hospital death prediction (任务 1 of the group project)
**Data**: PhysioNet/CinC Challenge 2012, all 12,000 stays pooled, 11,833 after exclusions
**Date**: 2026-09-25
**Scope**: In-hospital mortality only; this is the reviewed 11,833-stay XGBoost experiment. Preprocessing is in `preprocessing/README.md`. The later combined presentation uses different task numbers and a different mortality experiment.

**Repository revision (2026-10-17):** Executable analysis is in `In_Hospital_Death.ipynb`; AI disclosure is in `docs/GAI_DECLARATION.md`. Historical result values are preserved; interpretation and reproduction instructions below have been corrected.

Model results come from the named files under `results/`. The GCS univariate screen is reproduced in `In_Hospital_Death.ipynb`; external challenge context is referenced separately in `docs/REFERENCES.md`.

---

## 1. Summary

A gradient-boosted tree model (XGBoost, 1,474 features) predicts in-hospital death from the first 48 hours of an ICU stay with **test AUROC 0.885 [0.865, 0.903]** and **AUPRC 0.591 [0.535, 0.649]** (n = 1,775, prevalence 0.144). The bedside scores on the same patients reach AUROC 0.640 (SAPS-I) and 0.623 (SOFA). An eight-feature logistic regression with well-chosen summary statistics reaches 0.821, so most of the distance from the bedside scores is closed by feature engineering rather than by model complexity; the full GBDT adds a further 0.064, with non-overlapping marginal bootstrap intervals (a descriptive comparison, not a paired test).

Three findings carry the discussion:

1. **Compact summaries carry substantial predictive information.** Fifty features reach validation AUROC 0.857 versus 0.869 for the full ablation model; the eight-feature tree and revised logistic validation estimates are similar. Feature set and model family change together, so this does not isolate an algorithm effect.
2. **Which summary statistic you take matters as much as which variable.** The last recorded GCS in the window (AUROC 0.757 alone on train+validation) beats the minimum GCS (0.589) by a wide margin, with sedation and procedures as possible but unverified explanations. Swapping GCS min for last and urine min for total together raised logistic test AUROC from 0.742 to 0.821.
3. **The model has useful internal discrimination across ICU types, with imperfect calibration.** Brier 0.086; subgroup AUROC ranges 0.837 (Medical) to 0.955 (Cardiac surgery recovery), with no subgroup below 0.83.

At the clinically chosen operating point (validation recall 0.80), the model flags 29% of patients and 38.7% of those flagged die, against a base rate of 14.4%.

---

## 2. Data and protocol

### 2.1 Cohort

| | Rows | Deaths | Prevalence |
|---|---|---|---|
| Train | 8,283 | 1,195 | 0.1443 |
| Validation | 1,775 | 256 | 0.1442 |
| Test | 1,775 | 256 | 0.1442 |
| Total | 11,833 | 1,707 | 0.1443 |

Splits are 70/15/15, stratified by outcome, seed 20260907, fixed before any modelling. 167 stays with negative length of stay were excluded upstream; all were survivors, which is why prevalence is 0.1443 and not the 0.142 quoted in the original plan. Source: `data/split_summary.csv`.

### 2.2 Features

1,474 features per stay, built by the preprocessing team and frozen on 2026-09-21: 37 dynamic parameters × 3 windows (0–24 h, 24–48 h, 0–48 h) × 12 statistics (measured, count, first, last, min, max, mean, median, std, slope, delta, time span), plus merged blood-pressure streams, static descriptors, and record-level counters. Missing values are kept as NaN for tree models; the logistic models use train-median imputation. SAPS-I, SOFA, length of stay and survival time are excluded from the matrix.

### 2.3 Evaluation rules

- **Test set scored once per declared model.** All hyperparameter search, model selection and threshold choice used train + validation only.
- **Reporting: AUPRC and AUROC**. Historical hyperparameter search and model selection optimised AUROC; AUPRC is clinically relevant at this prevalence but was not the original selection objective. Both carry 1,000-resample stratified bootstrap 95% CIs from one shared `metrics.py`.
- **Operating point**: threshold chosen on validation for recall ≈ 0.80, then applied to test. Missing a death costs more than an unnecessary review.
- **min(Se, PPV)** reported for comparability with the 2012 challenge.
- **Bedside scores** are evaluated on the patients for whom they were computable (SAPS-I n = 1,728, SOFA n = 1,733). 47 and 42 test patients respectively have no score; those patients have roughly twice the average mortality, so treating the sentinel as a low score would invert them. The model is also scored on those same subsets for the head-to-head.
- **Post-hoc analyses** (calibration, subgroups, SHAP, case explanations) run on the frozen test prediction vector. They were listed before being computed and involve no further choices.

---

## 3. Baseline ladder

**Table 1. Test-set performance, all rungs.** Se/PPV/Sp at the validation recall-0.80 threshold. Threshold metrics are omitted for L0/L1: the old gap-fill used a different random draw sequence and an inconsistent GCS risk direction, so it must not be combined with their ranking metrics.

| Rung | Model | Features | AUROC [95% CI] | AUPRC [95% CI] | Se | PPV | Sp | min(Se,PPV) | n |
|---|---|---|---|---|---|---|---|---|---|
| L0 | Random | 0 | 0.474 [0.436, 0.513] | 0.135 [0.124, 0.151] | — | — | — | — | 1,775 |
| L1 | Age | 1 | 0.627 [0.590, 0.667] | 0.224 [0.196, 0.265] | — | — | — | — | 1,775 |
| L1 | GCS min | 1 | 0.553 [0.517, 0.586] | 0.155 [0.143, 0.168] | — | — | — | — | 1,775 |
| L1 | BUN max | 1 | 0.675 [0.635, 0.710] | 0.265 [0.231, 0.310] | — | — | — | — | 1,775 |
| L2 | SAPS-I | score | 0.640 | 0.213 | 0.819 | 0.171 | 0.348 | 0.171 | 1,728 |
| L2 | SOFA | score | 0.623 | 0.253 | 0.820 | 0.171 | 0.345 | 0.171 | 1,733 |
| L3 | Logistic, 8 features (clinical-prior statistics) | 8 | 0.742 [0.708, 0.774] | 0.395 [0.343, 0.455] | 0.797 | 0.220 | 0.524 | 0.220 | 1,775 |
| **L3-revised** | **Logistic, 8 features (GCS-last, urine-total)** | 8 | **0.821 [0.793, 0.846]** | **0.500 [0.445, 0.556]** | 0.840 | 0.297 | 0.666 | 0.297 | 1,775 |
| **L4** | **XGBoost, 1,474 features** | 1,474 | **0.885 [0.865, 0.903]** | **0.591 [0.535, 0.649]** | 0.785 | 0.387 | 0.791 | 0.387 | 1,775 |

Sources: `L0_random_baseline.json`, `L1_*.json`, `L2_clinical_scores_analysis.json`, `L3_threshold_analysis.json`, `L3_logistic_regression_8features.json`, `L3_revised_test_evaluation.json`, `L4_test_evaluation.json`. Historical threshold gap-fill caveats are recorded in `docs/AI_USE_LOG.md`.

**Reading the ladder.**
- A single lab value (BUN max, 0.675) already matches or beats both bedside scores. The scores were designed for population stratification, not individual prediction, and each takes only a few dozen distinct values across 1,775 patients.
- L3 and L3-revised use the same eight parameters and the same recipe (L2 logistic, C = 1, balanced weights). They differ only in two summary statistics (§5). That swap is worth 0.079 AUROC.
- L4 over L3-revised: +0.064 AUROC, +0.091 AUPRC; marginal CIs do not overlap. No paired significance test was performed.

### 3.1 Head-to-head with the bedside scores on their own patients

| Comparison | Baseline AUROC [CI] | L4 AUROC [CI] | Gain | CIs overlap |
|---|---|---|---|---|
| vs SAPS-I (n = 1,728) | 0.640 [0.601, 0.676] | 0.886 [0.864, 0.905] | +0.246 | No |
| vs SOFA (n = 1,733) | 0.623 [0.586, 0.660] | 0.890 [0.870, 0.909] | +0.267 | No |

Source: `L4_supplementary_evaluation.json`.

---

## 4. The L4 model

### 4.1 Training and selection

Two GBDT implementations were trained on the unimputed matrix with native NaN handling and `scale_pos_weight` for class imbalance (no resampling). Hyperparameters were chosen by 5-fold stratified CV on the training set, 50 random candidates each. LightGBM was attempted but could not use the GPU on the available machine and was dropped before producing a result.

| Model | CV AUROC | Validation AUROC | Validation AUPRC | Training time |
|---|---|---|---|---|
| **XGBoost** (selected) | 0.871 | **0.866** | **0.538** | 13 min (GPU) |
| CatBoost | 0.867 | 0.860 | 0.505 | 89 min (GPU) |

XGBoost selected on validation AUROC. Chosen configuration: `max_depth=7, learning_rate=0.05, n_estimators=300, min_child_weight=3, subsample=0.9, colsample_bytree=0.9, gamma=0, max_bin=127`. Sources: `L4_xgboost_training.json`, `L4_catboost_training.json`, `L4_model_selection.json`.

### 4.2 Capacity and generalisation

| Model | Trees used | Train AUROC | Validation AUROC | Gap |
|---|---|---|---|---|
| XGBoost | 300 of 300 | 1.000 | 0.866 | 0.134 |
| CatBoost | 224 (early-stopped) | 0.957 | 0.860 | 0.097 |
| L3 logistic (reference) | — | 0.733 | 0.722 | 0.012 |

XGBoost nearly perfectly discriminates the training rows and has a substantial train–validation AUROC gap. Validation 0.866 and test 0.885 provide internal held-out evidence, but do not rule out overfitting, selection optimism, or failure at another site. The saved early-stopping ablation changes validation AUROC by approximately 0.0002. These are descriptive checks on one cohort and split. Sources: `L4_overfit_comparison.json`, `L4_xgboost_es_ablation.json`.

### 4.3 How many features are needed

Features ranked by mean |SHAP| on the **training** set, then XGBoost refit on the top k with early stopping on validation. Validation metrics only.

| Features | Validation AUROC | Validation AUPRC | Share of full-model AUROC |
|---|---|---|---|
| 8 | 0.813 | 0.422 | 93.5% |
| 20 | 0.835 | 0.452 | 96.1% |
| 50 | 0.857 | 0.491 | 98.6% |
| 100 | 0.861 | 0.506 | 99.0% |
| 200 | 0.862 | 0.530 | 99.1% |
| 500 | 0.867 | 0.517 | 99.8% |
| 1,474 | 0.869 | 0.532 | 100% |

Two things follow. Fifty features carry nearly all of the signal, so a compact model is the natural deployment candidate. And XGBoost on its own top eight (0.813) is within noise of the eight-feature logistic model (L3-revised, validation 0.808): at that scale the trees add essentially nothing over a linear fit, suggesting feature breadth contributes, although a matched-feature experiment would be needed to isolate model-family effects. Source: `L4_feature_count_ablation.json`.

### 4.4 Calibration

Brier score **0.086** on test. The ten-bin reliability curve tracks the diagonal up to about 0.35 predicted probability and then falls below it: patients predicted at 0.65 die at a rate of 0.54, and patients predicted at 0.95 die at 0.83. The model is slightly over-confident at the high end, which may be related to class weighting and the small numbers in high-risk bins. Ranking (AUROC) is unaffected; if calibrated probabilities are needed for communication with families, validation-fitted isotonic or Platt scaling would need separate evaluation; correction is not guaranteed. Source: `L4_calibration_and_subgroup.json`; figure `results/figures/L4_calibration_curve.png`.

### 4.5 Subgroups by ICU type

| ICU type | n | Deaths | Prevalence | AUROC [95% CI] |
|---|---|---|---|---|
| Coronary | 262 | 37 | 0.141 | 0.854 [0.789, 0.909] |
| Cardiac surgery recovery | 370 | 19 | 0.051 | 0.955 [0.913, 0.986] |
| Medical | 646 | 120 | 0.186 | 0.837 [0.799, 0.871] |
| Surgical | 497 | 80 | 0.161 | 0.897 [0.860, 0.929] |
| All | 1,775 | 256 | 0.144 | 0.885 [0.865, 0.903] |

No subgroup drops below 0.83. The Medical ICU, the largest and most heterogeneous, is hardest; the cardiac-surgery unit, with only 19 deaths, has the highest point estimate but only 19 deaths. Coronary has a wider AUROC CI; event counts and interval widths should both be considered. `static_icutype_2` is the 11th most important SHAP feature, so the model uses unit type directly. Source: `L4_subgroup_by_icutype.json`.

---

## 5. What the model learned

### 5.1 Global feature importance

Mean |SHAP| on the test set, XGBoost. Top 10 of 1,474:

| Rank | Feature | \|SHAP\| | Reading |
|---|---|---|---|
| 1 | gcs__0_48h__last | 0.694 | Last recorded consciousness level in the window |
| 2 | static_age_years | 0.258 | Age |
| 3 | urine__0_48h__total | 0.141 | Total urine output, 48 h |
| 4 | wbc__0_48h__slope_per_hour | 0.099 | Trend in white-cell count |
| 5 | gcs__24_48h__last | 0.097 | Last GCS, second day |
| 6 | bun__0_48h__min | 0.088 | Best BUN in the window |
| 7 | mechvent__0_48h__time_span_hours | 0.085 | Hours between first and last ventilator record |
| 8 | urine__24_48h__total | 0.083 | Urine output, second day |
| 9 | gcs__24_48h__median | 0.082 | Median GCS, second day |
| 10 | bun__24_48h__last | 0.079 | Last BUN, second day |

GCS-last is 2.7× the next feature. Seven of the top twenty are GCS statistics; four are renal (BUN, urine); two are trends (`slope`); two are measurement-behaviour features (`time_span_hours`). CatBoost, trained independently, puts the same three features at ranks 1–3 and shares 12 of the top 20, so the ranking is not an artefact of one implementation. Sources: `L4_feature_importance.json`, `L4_shap_validation_comparison.json`; figure `results/figures/L4_shap_summary.png`.

Leakage was checked: record-length proxies (`time_span_hours`, parameter counts) have single-variable AUROC 0.50–0.62; GCS charting span is identical for deaths and survivors (median 44.5 h vs 44.0 h); no ID or outcome-derived column is in the matrix.

### 5.2 Last GCS versus minimum GCS

The original L3 used `gcs__0_48h__min` because the SOFA neurological component uses the worst GCS in the window. Single-variable screen on observed train + validation rows (test untouched; n=9,907 for most GCS summaries, 9,906 for slope). The main notebook also shows the separate train-median-imputed estimates over all 10,058 development stays:

| GCS statistic, 0–48 h | AUROC |
|---|---|
| last | 0.757 |
| median | 0.722 |
| mean | 0.707 |
| max | 0.686 |
| slope / delta | 0.666 / 0.665 |
| **min** | **0.589** |

GCS minimum is less discriminative in this cohort. Sedation, intubation and procedures could contribute to low minima; medication information is unavailable, so this mechanism and whether its effects have worn off by the last measurement cannot be confirmed. Refitting the L3 recipe with `min → last` and `urine min → urine total` raised validation AUROC from 0.722 to 0.808; that feature set was then declared as L3-revised and scored on test once (0.821). Source: `L3_gcs_statistic_ablation.json`, `L3_revised_test_evaluation.json`.

The result illustrates how summary-statistic choice changes predictive performance; physiology and documentation effects cannot be separated causally here.

### 5.3 Three patients

SHAP waterfalls on the test set for one correct high-risk call, one false alarm, and one missed death. Figures in `results/figures/L4_shap_waterfall_*.png`; contributions in `L4_case_examples.json`.

**Correct high-risk call (RecordID 103032, predicted 0.997, died).** Last GCS 7 (+0.91 log-odds), last lactate 11.6 mmol/L (+0.33), second-day urine only 414 mL (+0.32), platelets 25 (+0.23), mean systolic BP 97 (+0.21). Several measured organ-system signals contribute toward higher model risk; this does not establish clinical certainty.

**False alarm (RecordID 100571, predicted 0.976, survived).** Last GCS 6 (+1.74), age 80 (+0.23), lactate 3.0 (+0.23). Urine output 5.5 L pulls the other way (−0.17) but cannot offset the GCS. Sedation or a post-ictal state are possible explanations, not verified patient facts; the model is reading depressed consciousness as prognostic although the actual reason for this patient’s survival is not established. This is the failure mode of the GCS-last feature and the case for pairing it with a less timing-dependent summary such as the median.

**Missed death (RecordID 102169, predicted 0.004, died).** Last GCS 8 pushes risk up (+0.48), but 7.2 L urine, age 53, a falling white count, lactate 0.6 and a short ventilation span all push it down, and the remaining 1,460 features contribute a further −4.5 log-odds. On the numbers available at 48 h this patient looked like a recovering one. Later deterioration is a possible explanation, but its timing and cause are unavailable; this case does not establish an unavoidable modelling limit.

---

## 6. Challenge context and limits of comparison

The PhysioNet/CinC 2012 challenge used a different data partition and min(Se, PPV) as its event-1 score. Its SAPS-I reference scored 0.313 and the best entry 0.535 on the official evaluation. Dataset and organiser references are listed in `docs/REFERENCES.md`. Our pooled and re-split cohort is not a direct benchmark comparison; unsupported specific literature AUROC claims from earlier notes are not used here.

Our test min(Se, PPV) is 0.387 at the validation recall-0.80 operating point. A separate validation-only sweep selects threshold 0.345 with min(Se, PPV) approximately 0.518 (Se and PPV about 0.52). This illustrates a sensitivity/precision trade-off, not a claim to beat the challenge. The test set was not re-thresholded for that objective. Source: `L4_challenge_metric_validation.json`.

---

## 7. Clinical reading of the operating point

At the recall-0.80 threshold on test (n = 1,775):

| | Flagged | Not flagged |
|---|---|---|
| Died (256) | 201 | 55 |
| Survived (1,519) | 318 | 1,201 |

The model flags 519 patients (29%). Of those, 201 die: a 38.7% event rate against a 14.4% base rate, a 2.7-fold enrichment. It misses 55 of 256 deaths. SAPS-I at its own recall-0.80 threshold flags 1,167 of 1,728 patients (68%) to catch a similar number of deaths, so the model delivers the same sensitivity with less than half the alert burden.

---

## 8. Limitations

1. **Single cohort, one split, no external validation.** All results come from PhysioNet 2012. `gcs__0_48h__last` in particular depends on charting cadence and needs testing at another site.
2. **Probabilities are over-confident above ~0.35** (§4.4). Fine for ranking and alerting; recalibrate before quoting risks to families.
3. **Two L3 coefficients have the wrong sign** (creatinine, platelets), attributable to collinearity with BUN and weak signal. Left as is because fixing them would be a third test-set look.
4. **The full 1,474-feature model is the research benchmark, not the deployment candidate.** A 50–100 feature version keeps 98.6–99% of validation AUROC and is far easier to audit.
5. **The outcome is in-hospital death, not ICU death**, so deaths after ICU discharge count as positives; survival time after 48 h is not modelled.
6. **Fairness by sex or age band was not assessed.** Age is the second most important feature.

---

## 9. Reproducing this work

See `README.md` for the portable environment and commands. Run `python tools/execute_notebook.py` from this package with Python 3.11 and `requirements.txt` installed. The notebook refits both fixed L3 recipes, loads `models/L4_xgboost_best.json`, verifies the reviewed metrics, and writes published tables/checks to `results/notebook/`, figures to `results/figures/notebook/`, and L3 pipelines/metadata to `models/notebook/`. The output index is `results/notebook/README.md`; optional new development fits use `artifacts/notebook/`.

Modules are `code/data.py` and `code/metrics.py`; historical entry points are under `code/baselines/`. All paths resolve inside this package. The data snapshot is `data/` and is checked against its SHA-256 manifest. Optional full training and raw preprocessing are documented separately; default execution does neither. The raw cleaning-policy discrepancy and unavailable historical GAI history are disclosed in `docs/DATA_README.md` and `docs/GAI_DECLARATION.md`.
