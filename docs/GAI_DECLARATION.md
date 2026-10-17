# Declaration on Use of GAI

**Assessment:** MD6117 Machine Learning for Healthcare AI — in-hospital mortality component.
**Declaration updated:** 17 October 2026.

## Confirmed use

OpenAI Codex was used during this repository revision to inspect the supplied project files and assessment requirements; identify reproducibility defects; draft an annotated modelling/EDA notebook; repair package-relative paths and bootstrap interval handling; draft documentation and this log; organise standalone figures, result tables and L3 model exports; simplify the repository root and archive supporting documentation; and execute automated validation against the existing artifacts.

AI assistance did not create the clinical observations, outcome labels, cohort split, or original reviewed XGBoost/CatBoost models. New notebook tables and figures are calculated from the submitted data and frozen model; the two inexpensive logistic recipes are refitted from training rows. Existing SHAP figures are identified as retained artifacts, rather than presented as freshly computed.

The model-selection protocol and original results are preserved. AI suggestions are checked against code, hashes and executed outputs. Hypotheses about sedation or reasons for individual deaths are identified as unverified explanations. See [AI_USE_LOG.md](AI_USE_LOG.md) for errors and their corrections.

## Limit of this declaration

The files available for this revision do **not** establish all tools used by each member during earlier preprocessing, modelling or writing. This statement covers only the confirmed revision above. Each group member must confirm any earlier GAI tool, its specific role and material corrections before the final group submission. Missing history must not be replaced with an invented tool list, model version, prompt transcript or claim of no prior use.

Academic responsibility for the submission and its final declaration remains with the team. This factual disclosure is placed at the start of the README and the main notebook to match the submission requirement.
