# AI use and error log

This log implements the code-repository requirement on page 5 and the “AI Use & Error Log” criterion on page 2 of `MD6117 Project Assessment Rubrics STUDENT.pdf`. It covers the mortality component only.

## Confirmed tool use

| Date | Tool | Purpose | Verification / output |
|---|---|---|---|
| 2026-10-17 | OpenAI Codex | Read the local rubric and inspect the mortality repository | `SUBMISSION_CHECKLIST.md`; scoped to reviewed 11,833-stay experiment |
| 2026-10-17 | OpenAI Codex | Draft executable annotated EDA/modelling notebook and reproduction helpers | `In_Hospital_Death.ipynb`; full default execution; generated `artifacts/notebook/verification.json` |
| 2026-10-17 | OpenAI Codex | Repair paths, bootstrap indexing and misleading interpretation | Changes in `code/`, report and presentation notes; regression checks |
| 2026-10-17 | OpenAI Codex | Draft declaration, usage log, README and artifact guide | Claims cross-checked against files, manifest and executed metrics |

## Errors found and corrections

These are **artifact errors found in an AI-assisted audit**; the source files do not establish that AI originally caused them.

| Issue | Evidence / effect | Correction and check |
|---|---|---|
| Imports and paths pointed to missing `src/`, external `experiment/`, or `code/results/` | A clean clone could not run baseline scripts or load models | Resolve paths relative to the submitted package; remove `src` imports; loader and script smoke checks |
| L3 bootstrap tuple read as `(low, high)` | Shared helper returns `(point, low, high)`; rerunning L3 would overwrite already-corrected CI with wrong values | Use elements 1 and 2 for bounds; notebook recomputes intervals with 1,000 resamples |
| L4 comparison embedded the old truncated L3 interval | Canonical L3 JSON had corrected bounds but comparison retained old bounds | Correct comparison bounds with an audit note; headline L4 performance unchanged |
| A single one-hot ICU column interpreted as category codes 1–4 | Historical calibration artifact called cardiac-surgery stays “Coronary” | Use four one-hot masks; canonical subgroup artifact is `L4_subgroup_by_icutype.json`; historical result retained and explicitly marked superseded for subgroup analysis |
| L0/L1 threshold gap-fill used a different random sequence and the wrong GCS direction | Those values did not describe exactly the same ranking-score vectors as the original L0/L1 rows | Do not combine them with reproduced ranking metrics; notebook fixes direction and documents historical gap-fill limitations |
| SHAP TreeExplainer failed on the saved XGBoost 3.2 base score | Reproduction raised `ValueError` for vector-form base score in the pinned SHAP environment | Use native XGBoost `pred_contribs` / CatBoost `ShapValues` and retain SHAP plotting; checked on frozen model |
| Pip freeze contained local `file:///croot/...` paths and unrelated packages | Not portable; GPU conda file omitted CatBoost | Provide minimal pinned Python 3.11 requirements; preserve original freeze as historical evidence |
| README described a different layout and said data were absent | Bundled `data/` and top-level code modules were present | Replace with current layout, CPU execution steps and explicit output paths |
| GCS univariate screen denominators were implicit | Observed-only last-GCS AUROC 0.757 differs from train-median-imputed AUROC 0.752 | Notebook now displays both definitions, their sample sizes and recomputed estimates; primary model results unchanged |
| Later combined presentation used another cohort/model and different task numbers | 12,000-stay CatBoost result could be confused with 11,833-stay XGBoost | Name the clinical task and reviewed version explicitly; exclude other task |
| Report called CI non-overlap a significance test and denied overfitting | No paired test was implemented; large train-validation gap remains | Label CI comparison descriptive and acknowledge capacity/generalisation uncertainty |
| Sedation and individual-patient explanations stated as facts | No medication or later-course evidence supplied | Describe these as possible explanations; avoid causal claims |
| Preprocessing descriptions conflict about physiological cleaning | `config.json`/frozen summary say retain outliers; supplied code contains range checks and targeted corrections | Preserve frozen bytes; disclose discrepancy; raw rebuild is a separately audited operation |

## Historical correction evidence

`results/L3_logistic_regression_8features.json` records a previous CI correction dated 2026-09-25. `results/L4_subgroup_by_icutype.json` already supersedes the single-column subgroup. The present revision repairs runnable code and documentation so those errors are not silently reintroduced. The original training models, data snapshot, and headline metric values remain unchanged.

## Team confirmation still needed

Earlier tools, member-specific use and prompt history cannot be recovered reliably from the delivered files. Members should add only factual prior use and corrections here before the combined submission. No unverified tools or invented prompt transcripts have been added.
