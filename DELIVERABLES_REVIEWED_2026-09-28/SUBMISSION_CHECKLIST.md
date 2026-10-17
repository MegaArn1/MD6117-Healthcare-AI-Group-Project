# Project code repository — requirement mapping

Source: `Group_Presentation/MD6117 Project Assessment Rubrics STUDENT.pdf`, page 5 (deliverables), page 2 (code marking criteria), page 1 (GAI policy). Scope: **in-hospital mortality component only**. The full group assessment requires two tasks; the other task and PowerPoint are maintained separately.

| Required item | Evidence in this package | Status / qualification |
|---|---|---|
| Declaration on Use of GAI at start | README opening section; first notebook cell; `GAI_DECLARATION.md` | Confirmed revision disclosed; earlier individual use requires factual team confirmation |
| Executable, annotated Python Jupyter notebook | `In_Hospital_Death.ipynb` | EDA, preprocessing audit, feature choice, logistic fits, optional L4 search/refit, saved L4 replay |
| Task definition, tables and plots | Notebook §§1–9 | Mortality only; other task deliberately excluded |
| Analysis and discussion of results/plots | Notebook interpretation cells and §10; `FINAL_REPORT.md` | Includes cohort/split, clinical trade-offs, calibration, uncertainty and limitations |
| Trained models | `models/L4_xgboost_best.json`, `models/L4_catboost_best.cbm` | Original selected model and validation comparator; notebook saves fitted L3 pipelines separately |
| Preprocessing files | `preprocessing/` and its QA notebook; `data/` snapshot | Raw rebuild optional; raw release not included; provenance discrepancy disclosed |
| README ≤1,500 words | `README.md` | Dataset, scope, environment, commands, output paths and run modes |
| AI use log, including incorrect outputs | `AI_USE_LOG.md` | Confirmed tool use plus traceable errors and fixes; no invented historical attribution |

| Code marking criterion (60 marks total) | Where to inspect |
|---|---|
| Problem definition and clinical framing (6) | Notebook §1; report §§1–2 and 7 |
| EDA and interpretation (12) | Notebook §3 training-only coverage and distributions; §4 quality audit |
| Preprocessing, feature engineering and selection (12) | Notebook §§4–5 and 9; feature dictionary; raw preprocessing and tests |
| Hyperparameter tuning (6) | Notebook §6; retained XGBoost/CatBoost training records and CatBoost checkpoint; optional full search; GPU scripts |
| Model diversity, comparison, selection, metrics and clinical insight (12) | Notebook §§5–10; score-available comparisons, ROC/PR, threshold, calibration, subgroup and case discussions |
| Code clarity/documentation (6) | Annotated notebook; loader/metrics; portable README; `tools/` |
| AI use/error log (6) | Opening declaration, `GAI_DECLARATION.md`, `AI_USE_LOG.md` |

## Execution checks

From the package root in the pinned environment:

```bash
python tools/verify_submission.py
python -m unittest discover -s tests -v
python -m unittest discover -s preprocessing/tests -v
python code/validate_data_loading.py
python tools/execute_notebook.py
python tools/verify_submission.py --require-executed
```

The verifier checks required files, README length, frozen hashes, notebook syntax and saved successful outputs. Notebook execution asserts the principal reviewed metrics. Full GPU retraining and raw-data preprocessing are not claimed as part of default verification.

## Assembly outside this mortality repository

Combine this component with the separately maintained task and the PowerPoint (all-member photo and contribution slide), then submit a single `<Group_Number>.zip` via NTU-Learn. The group number and individual contribution/AI declarations must come from the team. A GitHub URL alone is not the compressed submission required by the rubric.
