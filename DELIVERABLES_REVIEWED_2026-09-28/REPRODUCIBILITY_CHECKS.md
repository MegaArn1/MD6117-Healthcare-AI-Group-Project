# Reproducibility verification — 17 October 2026

Scope: reviewed in-hospital mortality experiment, 11,833 stays, 1,474 features. No other task was evaluated.

| Check | Observed result |
|---|---|
| Main notebook, default CPU Run All | 18 code cells executed with no error outputs |
| Frozen input manifest | All 16 SHA-256 entries match |
| L0 / L1 / original L3 / revised L3 / selected L4 | All 7 principal AUROC/AUPRC comparisons match reviewed records |
| XGBoost test AUROC | 0.884880060895326 |
| XGBoost test AUPRC | 0.5911575114623904 |
| Validation-derived XGBoost threshold | 0.10822707414627075 |
| Test sensitivity / PPV / specificity | 0.78515625 / 0.3872832369942196 / 0.7906517445687953 |
| Confusion counts (TN / FP / FN / TP) | 1,201 / 318 / 55 / 201 |
| Brier score | 0.08626776029039518 |
| New alignment/metric regression tests | 4 passed |
| Existing preprocessing unit tests | 18 passed |
| Existing phase/schema validation | 66 of 66 checks passed |
| Independent standalone script replay from `/tmp` | L3 fit, L4 evaluation, calibration/subgroups and native-SHAP case plots all passed in a copied package |
| L3 interval regression | Re-executed script uses bootstrap bounds, not point estimate as a bound |
| Subgroup regression | Four one-hot masks cover all 1,775 test stays |
| Submission verifier | Required artifacts, notebook syntax, saved execution, links and README word limit pass |
| Exported L3 pipeline reload | Both saved bundles reproduce their pre-save validation probabilities exactly |
| Published output manifest | All 13 published data/figure/model files match their recorded SHA-256 hashes; CSV totals agree with replay report |

Execution environment: Python 3.11.7; NumPy 1.26.4; pandas 2.1.4; SciPy 1.11.4; scikit-learn 1.2.2; XGBoost 3.2.0; CatBoost 1.2.10; Matplotlib 3.8.0; joblib 1.3.2. Native TreeSHAP is used in regeneration scripts because SHAP 0.49.1 TreeExplainer cannot parse the saved XGBoost vector-form base score.

`tools/execute_notebook.py` uses the active interpreter through a temporary kernelspec. Default runs publish tables and verification records to `results/notebook/`, five figures to `results/figures/notebook/`, and fitted L3 bundles plus metadata to `models/notebook/`. These reviewed outputs and the executed notebook are committed. `results/notebook/manifest.json` records SHA-256 hashes for the 13 exported files. Only optional new search/refit outputs use the Git-ignored `artifacts/notebook/` directory. See the saved output index in `results/notebook/README.md`.

Not executed in this revision: full GPU hyperparameter searches or a raw-to-feature rebuild. The original L4 models and frozen snapshot are preserved. The existing preprocessing QA notebook was checked for valid saved execution; its underlying outputs passed the separate regression/schema checks. Current cleaning-policy provenance and unavailable earlier individual AI-use history are disclosed, not silently assumed resolved.
