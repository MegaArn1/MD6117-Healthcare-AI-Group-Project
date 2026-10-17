# MD6117 — In-hospital Death prediction

## Declaration on Use of GAI

OpenAI Codex assisted with the 17 October 2026 repository revision, executable notebook, reproducibility fixes and documentation. See the [declaration](DELIVERABLES_REVIEWED_2026-09-28/GAI_DECLARATION.md) and [AI use/error log](DELIVERABLES_REVIEWED_2026-09-28/AI_USE_LOG.md). Earlier individual use requires factual team confirmation.

This repository contains the **in-hospital mortality component only**, preserving the reviewed **11,833-stay / 1,474-feature / XGBoost** experiment. The other group task and the later combined 12,000-stay mortality experiment are outside its scope.

Start with the [package README](DELIVERABLES_REVIEWED_2026-09-28/README.md) and the [executed notebook](DELIVERABLES_REVIEWED_2026-09-28/In_Hospital_Death.ipynb). The [submission checklist](DELIVERABLES_REVIEWED_2026-09-28/SUBMISSION_CHECKLIST.md) maps the mortality component to the course's code-repository requirements.

```bash
cd DELIVERABLES_REVIEWED_2026-09-28
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/verify_submission.py --require-executed
python tools/execute_notebook.py
```

Default execution runs on CPU with bundled processed data and saved models. It refits the fixed logistic baselines and reproduces XGBoost AUROC **0.884880**, AUPRC **0.591158**. See [verification results](DELIVERABLES_REVIEWED_2026-09-28/REPRODUCIBILITY_CHECKS.md) for checks and limitations.

The final NTU-Learn submission still needs assembly with the separately maintained task and PowerPoint into one `<Group_Number>.zip`.
