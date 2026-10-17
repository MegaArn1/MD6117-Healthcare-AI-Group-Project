# Results index

This directory covers the reviewed in-hospital mortality experiment only. Its two result sources are separated explicitly:

| Location | Source and purpose |
|---|---|
| `L0_*.json` through `L4_*.json` | Retained historical baseline, training, evaluation and ablation records |
| [notebook/README.md](notebook/README.md) | Published outputs from the executed CPU reproduction notebook: tables, figures and checks |
| [figures/README.md](figures/README.md) | Inventory of historical and notebook-generated PNG figures |

The default notebook replays fixed choices. Its outputs reproduce the reviewed experiment; they are not a new cohort or an independent experiment. Original L4 records keep their existing names. New CPU reproduction exports are grouped under `notebook/` so their origin is visible.

For ICU-specific historical results, use `L4_subgroup_by_icutype.json`: it supersedes the single-column subgroup in `L4_calibration_and_subgroup.json`. That earlier file still supplies the historical calibration record. Other interpretation corrections are documented in [AI_USE_LOG.md](../AI_USE_LOG.md).

Regenerate the published exports from the package root with `python tools/execute_notebook.py`, then verify them with `python tools/verify_submission.py --require-executed`. Optional new full searches and development refits write to the ignored `artifacts/notebook/` directory for later review.
