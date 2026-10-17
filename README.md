# In-hospital Death — MD6117 project code

## Declaration on Use of GAI

OpenAI Codex assisted with the 17 October 2026 repository audit, notebook, reproducibility fixes and documentation. See [GAI_DECLARATION.md](docs/GAI_DECLARATION.md) and [AI_USE_LOG.md](docs/AI_USE_LOG.md) for verified use and corrections. Earlier individual AI use must be confirmed by the team; it is not inferred or invented here.

## Scope and dataset

This repository contains **only in-hospital mortality prediction**. It preserves the reviewed 11,833-stay experiment: 1,474 features, XGBoost test AUROC **0.885** (95% CI 0.865–0.903), AUPRC **0.591** (0.535–0.649). The other group task and the later combined 12,000-stay CatBoost experiment are outside this package. Task numbers changed in later presentation documents; use the clinical task name to identify this experiment.

Source: PhysioNet / Computing in Cardiology Challenge 2012, sets A/B/C pooled. This historical cohort excludes 167 negative-length-of-stay records and contains 1,707 deaths (14.4%). Predict `In-hospital_death` at hour 48 from observations in **[0, 48 h)**. This is in-hospital death, including deaths after ICU discharge.

The bundled `data/` snapshot supplies all inputs for model replay. `data/manifest.sha256` records its hashes. The split is fixed by stay, stratified by outcome, seed 20260907: train 8,283 / validation 1,775 / test 1,775. Repeat admissions cannot be grouped because patient IDs are unavailable.

## Reading guide and document archive

The GitHub repository root is the project root: it directly contains this README, [In_Hospital_Death.ipynb](In_Hospital_Death.ipynb), [FINAL_REPORT.md](FINAL_REPORT.md), code, data, models and results. Start commands here; there is no dated deliverables folder to open first.

Supporting documents are collected under [docs/](docs/README.md):

| Document | Purpose |
|---|---|
| [GAI declaration](docs/GAI_DECLARATION.md) / [AI use and error log](docs/AI_USE_LOG.md) | Confirmed AI use, corrections and outstanding team history |
| [Submission checklist](docs/SUBMISSION_CHECKLIST.md) | Mapping to the course requirements |
| [Reproducibility checks](docs/REPRODUCIBILITY_CHECKS.md) | Executed validation and its limits |
| [Presentation materials](docs/PRESENTATION_MATERIALS.md) | Mortality slide tables, figure choices and Q&A |
| [References](docs/REFERENCES.md) | Dataset and method sources |
| [Data snapshot guide](docs/DATA_README.md) | Feature layout, provenance and handling traps |
| [Historical archive](docs/history/) | Earlier process notes, plans and original environment freeze; not current run instructions |

## Install and run

Use **Python 3.11**, at least 4 GB of free RAM and a CPU. No GitHub token, raw-data download or GPU is needed for default execution. From this directory:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/verify_submission.py
python tools/execute_notebook.py
```

The final command executes and saves [In_Hospital_Death.ipynb](In_Hospital_Death.ipynb) with the active Python interpreter. To use Jupyter interactively:

```bash
python -m ipykernel install --user --name md6117_mortality --display-name "MD6117 Mortality"
python -m notebook In_Hospital_Death.ipynb
```

Select **MD6117 Mortality**, then Restart Kernel and Run All. The committed notebook contains executed outputs for reading on GitHub.

Default Run All verifies data hashes, performs training-only EDA, fits both logistic pipelines, loads the reviewed XGBoost, recomputes test metrics and 1,000-resample confidence intervals, and draws ROC/PR, calibration and subgroup summaries. It displays retained SHAP plots. Expect minutes on a typical CPU; hardware affects runtime.

The [saved output index](results/notebook/README.md) provides figures, downloadable tables and verification records without opening the notebook:

| Output | Published location |
|---|---|
| Five PNG figures | `results/figures/notebook/` |
| Four CSV result tables | `results/notebook/tables/` |
| Replay report and output SHA-256 manifest | `results/notebook/verification.json`, `manifest.json` |
| Two fitted L3 pipelines and metadata | `models/notebook/` |

Default Run All refreshes these committed outputs. The replay asserts that the principal metrics match the reviewed records and that saved L3 pipelines reproduce their validation predictions after reloading. It is a repeatability check of fixed choices, not a new independent test.

## Models, development and evaluation

The ladder is random → age/GCS/BUN alone → available SAPS-I/SOFA → eight-feature logistic → revised logistic (GCS-last and urine-total) → XGBoost. L4 was selected over CatBoost by **validation AUROC**, following 50-candidate, 5-fold training CV searches. AUPRC is also reported because prevalence is low. The original selection objective was AUROC; it is not changed retrospectively.

The notebook's `RETRAIN_L4=False` and `RUN_FULL_SEARCH=False` switches expose an optional CPU refit and full training-only XGBoost search. Optional new fits write to the Git-ignored `artifacts/notebook/` directory and report validation results. GPU training entry points remain available:

```bash
python code/baselines/run_L4_xgboost_gpu.py
python code/baselines/run_L4_catboost_gpu.py
```

These legacy scripts overwrite their named training/model outputs when intentionally rerun. GPU searches originally took about 13 and 89 minutes. GPU-enabled XGBoost/CatBoost and compatible NVIDIA drivers are needed for the recorded GPU training route. LightGBM was attempted but yielded no result and is not required.

Thresholds target recall 0.80 on validation only. Reported test sensitivity is 0.785, PPV 0.387 and specificity 0.791. Test denominators differ for bedside scores; the notebook also compares L4 on exactly the score-available subsets. AUPRC uses `average_precision_score`; bootstrap resampling is stratified, seed 42. No paired significance test was performed.

## Files and preprocessing

| Path | Purpose |
|---|---|
| `In_Hospital_Death.ipynb` | Executable analysis, training, plots and discussion |
| `preprocessing/mortality_data_preprocessing.ipynb` | Separate executed QA notebook |
| `preprocessing/` | Raw preprocessing code/configuration, tests and copied outputs |
| `code/data.py`, `code/metrics.py` | Package-relative loader and shared metrics |
| `code/baselines/` | Historical model and analysis entry points |
| `models/` | Reviewed L4 models; `notebook/` contains refitted L3 bundles and metadata |
| `results/` | Historical metrics plus indexed notebook tables, figures and verification |
| `data/` | Frozen inputs, schema, quality summaries and input hashes |
| `FINAL_REPORT.md` | Main mortality report |
| `docs/` | Supporting documentation and historical archive; see the reading guide above |
| `tools/` | Notebook execution/generation and submission checks |

Raw preprocessing is optional and needs the course `release/README.md`, `release/outcomes.csv` and `release/icu_records/*.csv` placed inside this package. See [preprocessing/README.md](preprocessing/README.md). The raw release is not bundled. Rebuilt features must be audited separately; do not overwrite the frozen modelling snapshot. The supplied code/configuration and historical cleaning descriptions have a documented provenance discrepancy.

## Checks and interpretation

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s preprocessing/tests -v
python code/validate_data_loading.py
python tools/verify_submission.py --require-executed
```

This is single-cohort internal validation. Calibration is imperfect at high risk; GCS may reflect sedation/charting; external validation, patient-level grouping, fairness audits and clinical utility are untested. It is a research experiment.

For the final NTU-Learn submission, combine this mortality component with the separately maintained other task and PowerPoint, then package once as `<Group_Number>.zip`. This repository is the mortality component, not the complete two-task group submission.
