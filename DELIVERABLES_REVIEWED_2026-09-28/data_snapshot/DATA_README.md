# Frozen mortality data snapshot

The package **includes** the processed data in `../data/`. No external experiment directory is needed. Model code always reads the snapshot, independently of the current working directory.

| File in `data/` | Content |
|---|---|
| `patient_features_unimputed.csv.gz` | 11,833 rows × (RecordID + 1,474 features); NaN retained for trees |
| `X_train_imputed.csv.gz` | 8,283 rows, train-median imputed |
| `X_validation_imputed.csv.gz`, `X_test_imputed.csv.gz` | 1,775 rows each, same training-derived fill values |
| `labels_and_baselines.csv` | RecordID, binary death, fixed split, separate SAPS-I/SOFA and availability flags |
| `labels_and_splits.csv` | Original labels and split without bedside scores |
| `feature_dictionary.csv` | Exact order, source parameter, window, statistic, unit and missing semantics |
| `preprocessing_state.json` | Column order, schema hash and training fill values |
| `quality_flags_by_stay.csv`, `feature_missingness.csv`, `parameter_distributions.csv` | Quality and missingness audits |
| `source_manifest.json`, `preprocessing_summary.json`, `independent_verification.json` | Historical provenance and QA |
| `manifest.sha256` | SHA-256 for all 16 frozen snapshot files |

The split is 8,283 / 1,775 / 1,775, outcome-stratified, seed 20260907; 1,707 deaths, prevalence 0.1443. `RecordID` is never a predictor. Only stay IDs exist, so repeat admissions cannot be grouped by patient.

## Feature inventory

| Family | Count |
|---|---:|
| 37 dynamic parameters × 3 windows × 12 statistics | 1,332 |
| Urine total and ventilation-ever flags, each in 3 windows | 6 |
| Merged BP summaries (3 streams × 3 windows × 12 statistics) | 108 |
| Static descriptors including BMI, missing indicators and one-hot categories | 16 |
| Record counters (3 per window) | 9 |
| Arterial-line indicator (1 per window) | 3 |
| Total | 1,474 |

Windows are [0,24), [24,48), [0,48) h. Base statistics are measured, count, first, last, min, max, mean, median, std, slope, delta and time span. The exact dictionary is authoritative.

## Provenance and limits

This is the reviewed 11,833-stay cohort, taken from PhysioNet/CinC 2012 after exclusion of 167 negative-LOS records. Source raw paths and hashes are recorded in `source_manifest.json`. The course raw release is not bundled; do not assume another download or the later 12,000-stay combined project is identical.

The supplied preprocessing files contain an unresolved documentation discrepancy: `config.json` and the frozen summary describe retaining finite outliers, whereas `mortality_preprocess.py` calls physiological bounds plus pH/height corrections. The source manifest's pipeline hash matches the supplied script, but that alone cannot resolve which historical description is correct. We therefore do not claim an independently verified raw-to-snapshot byte-identical rebuild. Default model reproduction uses the bundled frozen bytes and verifies every manifest entry. The original preprocessing outputs and manifests are retained for audit.

Training medians are recomputed from unimputed training rows in the main notebook and compared with the stored state (CSV rounding tolerance 2e-6). No normalisation was applied to the snapshot; logistic standardisation is train-fitted. SAPS-I, SOFA, LOS and survival are excluded from the feature matrix.

## Handling traps

- Missing bedside scores are NaN with an availability flag. On test, SAPS-I is missing for 47 and SOFA for 42 stays. Evaluate each score only where available, and compare the model on those same patients.
- ICU type is one-hot encoded (`static_icutype_1` through `_4`). A single binary column is not a category code. The complete canonical subgroup result is `results/L4_subgroup_by_icutype.json`.
- `patient_features_48h_inclusive_affected_rows.csv.gz` contains 232 boundary-affected stays for a sensitivity analysis; it is not extra training data.
- “No measurement” is distinct from normal and from observed zero urine output.
- Shuffling or merging must preserve RecordID alignment. The loader explicitly aligns labels to the feature order.
