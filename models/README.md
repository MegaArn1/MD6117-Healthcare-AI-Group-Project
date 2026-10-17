# Trained model inventory

| File | Role and format |
|---|---|
| [L4_xgboost_best.json](L4_xgboost_best.json) | Historical selected XGBoost, all 1,474 engineered features, native NaN handling |
| [L4_catboost_best.cbm](L4_catboost_best.cbm) | Historical validation comparator, retained for cross-model explanation |
| [notebook/L3_pipeline.joblib](notebook/L3_pipeline.joblib) | CPU reproduction of the original eight-feature logistic recipe |
| [notebook/L3-revised_pipeline.joblib](notebook/L3-revised_pipeline.joblib) | CPU reproduction with GCS-last and urine-total substitutions |
| [notebook/model_metadata.json](notebook/model_metadata.json) | L3 ordered features, thresholds, recipe, seeds, training counts, environment and saved-model reload checks |

Each L3 bundle is a dictionary with `features`, `pipeline`, `threshold` and `purpose`. The pipeline packages train-median imputation, train-fitted standardisation and the fitted logistic estimator. It expects the listed **engineered per-stay columns**, not raw timestamped ICU observations. Use the pinned Python 3.11 environment in [requirements.txt](../requirements.txt) for the saved scikit-learn/joblib bundles.

From the package root, an inference example is:

```python
import joblib
import pandas as pd

bundle = joblib.load('models/notebook/L3-revised_pipeline.joblib')
stays = pd.read_csv('data/patient_features_unimputed.csv.gz')
risk = bundle['pipeline'].predict_proba(stays[bundle['features']])[:, 1]
flagged = risk >= bundle['threshold']  # threshold chosen on validation at recall 0.80
```

The notebook verifies that both exported bundles reproduce their validation probabilities exactly after reloading. They supplement the selected L4 artifact with executable logistic baselines; they do not change the historical selection. The [published output manifest](../results/notebook/manifest.json) records the L3 bundle hashes alongside their tables and figures.

Regenerate these L3 files with `python tools/execute_notebook.py` from the package root. Optional newly trained L4/search files go to ignored `artifacts/notebook/` until deliberately reviewed. The notebook's default replay retains the historical L4 artifacts at their existing paths.
