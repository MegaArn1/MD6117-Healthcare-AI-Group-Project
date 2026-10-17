"""Data loading utilities for In-hospital Death modelling experiments.

This module provides clean access to the frozen 1,474-feature snapshot and the
three-way split. All paths are relative to the package data/ directory, the byte-identical
frozen snapshot taken from preprocessing/outputs/ on 2026-09-21.

Usage:
    from data import load_data, get_split, get_feature_names
    
    labels_df, X_unimputed, X_imputed_splits = load_data()
    X_train, y_train = get_split('train')
    feature_names = get_feature_names()
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SplitName = Literal["train", "validation", "test"]

_FEATURE_NAMES = None
_LABELS_DF = None
_X_UNIMPUTED = None
_X_IMPUTED_SPLITS = None


def get_feature_names() -> list[str]:
    """Return the 1,474 feature column names (excludes RecordID).
    
    Returns:
        List of feature column names in the order they appear in the data files.
    """
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        _FEATURE_NAMES = _load_feature_names()
    return _FEATURE_NAMES


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[SplitName, pd.DataFrame]]:
    """Load all data files: labels, unimputed features, and imputed splits.
    
    Returns:
        tuple of:
            - labels_df: DataFrame with RecordID, In-hospital_death, split, and baselines
            - X_unimputed: DataFrame with RecordID + 1,474 unimputed features
            - X_imputed_splits: dict mapping split name to imputed feature DataFrame
    """
    global _LABELS_DF, _X_UNIMPUTED, _X_IMPUTED_SPLITS
    
    if _LABELS_DF is None:
        _LABELS_DF = pd.read_csv(DATA_DIR / "labels_and_baselines.csv")
    
    if _X_UNIMPUTED is None:
        _X_UNIMPUTED = pd.read_csv(DATA_DIR / "patient_features_unimputed.csv.gz")
    
    if _X_IMPUTED_SPLITS is None:
        _X_IMPUTED_SPLITS = {
            "train": pd.read_csv(DATA_DIR / "X_train_imputed.csv.gz"),
            "validation": pd.read_csv(DATA_DIR / "X_validation_imputed.csv.gz"),
            "test": pd.read_csv(DATA_DIR / "X_test_imputed.csv.gz"),
        }
    
    return _LABELS_DF, _X_UNIMPUTED, _X_IMPUTED_SPLITS


def get_split(split_name: SplitName) -> tuple[pd.DataFrame, pd.Series]:
    """Get X and y for a specific split.
    
    Args:
        split_name: One of 'train', 'validation', 'test'
    
    Returns:
        tuple of:
            - X: DataFrame with 1,474 imputed features (no RecordID)
            - y: Series with In-hospital_death labels
    """
    labels_df, _, X_imputed_splits = load_data()
    
    X = X_imputed_splits[split_name].copy()
    record_ids = X["RecordID"]
    X = X.drop(columns=["RecordID"])
    
    if not record_ids.is_unique or not labels_df["RecordID"].is_unique:
        raise ValueError("RecordID must be unique in labels and features")
    split_labels = labels_df.loc[labels_df["split"].eq(split_name)].set_index("RecordID")
    if set(record_ids) != set(split_labels.index):
        raise ValueError(f"label/feature RecordID mismatch for {split_name}")
    y = split_labels.loc[record_ids, "In-hospital_death"].reset_index(drop=True)

    return X, y


def _load_feature_names() -> list[str]:
    """Load feature names from the unimputed data header (internal use)."""
    import gzip
    with gzip.open(DATA_DIR / "patient_features_unimputed.csv.gz", "rt") as f:
        header = f.readline().strip()
        cols = header.split(",")
    return cols[1:]  # Exclude RecordID
