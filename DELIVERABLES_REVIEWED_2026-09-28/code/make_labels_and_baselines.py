"""Build labels_and_baselines.csv for the In-hospital Death modelling experiments.

Why this file exists
--------------------
`labels_and_baselines.csv` carries RecordID, the label, the frozen split, and the
two clinical severity scores (SAPS-I, SOFA) needed for the L2 baseline rung of
实验方案 §5.5 -- and NOTHING else.

The preprocessing pipeline deliberately keeps SAPS-I/SOFA out of the feature
matrix (they are outcome-adjacent and excluded by the leakage guard). But the L2
baseline needs them as *prediction scores*, not features. Without this file the
modelling session must reach into release/outcomes.csv, which also contains
Length_of_stay and Survival -- a single careless merge there leaks the outcome.
This file makes the safe path the easy path.

Sentinel handling
-----------------
Raw SAPS-I/SOFA use -1 for "not computable". Left as a numeric -1 they become the
LOWEST score, i.e. the model ranks those patients as lowest risk. Measured on this
cohort those patients are in fact HIGHER risk (SAPS-I -1 rows: 20.7% mortality vs
14.43% cohort-wide), so keeping -1 actively anti-correlates and depresses the
baseline. We therefore emit NaN plus an explicit availability flag and leave the
evaluation policy to the caller. See experiment/README.md for the recommendation.

Run:  python experiment/src/make_labels_and_baselines.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTCOMES = ROOT / "release" / "outcomes.csv"
LABELS = ROOT / "experiment" / "data" / "labels_and_splits.csv"
TARGET = ROOT / "experiment" / "data" / "labels_and_baselines.csv"

FORBIDDEN = {"Length_of_stay", "Survival"}


def main() -> None:
    labels = pd.read_csv(LABELS)
    outcomes = pd.read_csv(OUTCOMES)

    merged = labels.merge(
        outcomes[["RecordID", "SAPS-I", "SOFA", "In-hospital_death"]],
        on="RecordID", how="left", suffixes=("", "_src"), validate="one_to_one",
    )
    if len(merged) != len(labels):
        raise AssertionError(f"merge changed rowcount: {len(labels)} -> {len(merged)}")
    if not (merged["In-hospital_death"] == merged["In-hospital_death_src"]).all():
        raise AssertionError("label disagreement between splits file and outcomes.csv")
    merged = merged.drop(columns=["In-hospital_death_src"])

    for name, raw in (("saps_i", "SAPS-I"), ("sofa", "SOFA")):
        merged[f"{name}_available"] = (merged[raw] != -1).astype(int)
        merged[name] = merged[raw].where(merged[raw] != -1)
    merged = merged.drop(columns=["SAPS-I", "SOFA"])

    ordered = merged[[
        "RecordID", "In-hospital_death", "split",
        "saps_i", "saps_i_available", "sofa", "sofa_available",
    ]]

    leaked = FORBIDDEN & set(ordered.columns)
    if leaked:
        raise AssertionError(f"outcome-derived columns must not be exported: {leaked}")

    ordered.to_csv(TARGET, index=False, lineterminator="\n")
    print(f"wrote {TARGET.relative_to(ROOT)}  rows={len(ordered)}  cols={list(ordered.columns)}")
    for name in ("saps_i", "sofa"):
        n_missing = int(ordered[name].isna().sum())
        print(f"  {name}: available={int(ordered[f'{name}_available'].sum())}  NaN={n_missing}")


if __name__ == "__main__":
    main()
