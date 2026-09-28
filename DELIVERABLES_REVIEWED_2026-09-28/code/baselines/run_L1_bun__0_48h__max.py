"""
L1 Single-Feature Baseline - bun__0_48h__max as predictor for In-hospital Death.

Uses the raw bun__0_48h__max feature value as the risk score after median imputation.
Tests whether a single clinical variable (blood urea nitrogen max in first 48h) has 
discriminative power for mortality prediction.

Median imputation is trained on the training set and applied to validation and test.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.metrics import compute_metrics, bootstrap_ci


FEATURE_NAME = "bun__0_48h__max"


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load unimputed data
    print(f"Loading data for L1 baseline: {FEATURE_NAME}")
    print("="*60)
    labels_df, X_unimputed, _ = load_data()
    
    # Merge features with labels
    data = X_unimputed[["RecordID", FEATURE_NAME]].merge(
        labels_df[["RecordID", "In-hospital_death", "split"]], 
        on="RecordID"
    )
    
    # Split by train/validation/test
    train_data = data[data["split"] == "train"].copy()
    val_data = data[data["split"] == "validation"].copy()
    test_data = data[data["split"] == "test"].copy()
    
    print(f"Train:      {len(train_data):,} samples")
    print(f"Validation: {len(val_data):,} samples")
    print(f"Test:       {len(test_data):,} samples")
    
    # Compute median imputation value from training set
    train_feature = train_data[FEATURE_NAME]
    median_value = train_feature.median()
    missing_frac_train = train_feature.isna().mean()
    
    print(f"\nFeature: {FEATURE_NAME}")
    print(f"  Training median: {median_value:.4f}")
    print(f"  Missing in train: {missing_frac_train:.2%}")
    
    # Apply median imputation to all splits
    train_data[FEATURE_NAME] = train_data[FEATURE_NAME].fillna(median_value)
    val_data[FEATURE_NAME] = val_data[FEATURE_NAME].fillna(median_value)
    test_data[FEATURE_NAME] = test_data[FEATURE_NAME].fillna(median_value)
    
    # Extract test set for evaluation
    y_test = test_data["In-hospital_death"].values
    y_score = test_data[FEATURE_NAME].values
    
    n_test = len(y_test)
    prevalence = y_test.mean()
    missing_frac_test = data[data["split"] == "test"][FEATURE_NAME].isna().mean()
    
    print(f"  Missing in test: {missing_frac_test:.2%}")
    print(f"\nTest set: {n_test:,} samples, prevalence = {prevalence:.4f}")
    
    # Compute point estimates
    print("\nComputing AUPRC and AUROC...")
    metrics = compute_metrics(y_test, y_score)
    auprc_point = metrics['auprc']
    auroc_point = metrics['auroc']
    
    # Compute bootstrap confidence intervals
    print("Computing 95% bootstrap confidence intervals (1000 iterations)...")
    
    auprc_estimate, auprc_lower, auprc_upper = bootstrap_ci(
        y_test, y_score,
        metric_fn=lambda y_t, y_s: compute_metrics(y_t, y_s)['auprc'],
        n_bootstrap=1000,
        ci=0.95
    )
    
    auroc_estimate, auroc_lower, auroc_upper = bootstrap_ci(
        y_test, y_score,
        metric_fn=lambda y_t, y_s: compute_metrics(y_t, y_s)['auroc'],
        n_bootstrap=1000,
        ci=0.95
    )
    
    # Package results
    results = {
        'model': 'L1_bun__0_48h__max',
        'description': f'Single feature baseline: {FEATURE_NAME} with median imputation',
        'feature': FEATURE_NAME,
        'imputation': 'median',
        'imputation_value': float(median_value),
        'missing_fraction_train': float(missing_frac_train),
        'missing_fraction_test': float(missing_frac_test),
        'test_samples': int(n_test),
        'prevalence': float(prevalence),
        'auprc': float(auprc_point),
        'auprc_ci': [float(auprc_lower), float(auprc_upper)],
        'auroc': float(auroc_point),
        'auroc_ci': [float(auroc_lower), float(auroc_upper)]
    }
    
    # Save results
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "L1_bun__0_48h__max.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print formatted results
    print("\n" + "="*60)
    print(f"L1 BASELINE RESULTS: {FEATURE_NAME}")
    print("="*60)
    print(f"Feature:      {FEATURE_NAME}")
    print(f"Imputation:   median = {median_value:.4f}")
    print(f"Missing:      train={missing_frac_train:.2%}, test={missing_frac_test:.2%}")
    print(f"Test samples: {n_test:,}")
    print(f"Prevalence:   {prevalence:.4f}")
    print()
    print(f"AUPRC:        {auprc_point:.4f}  (95% CI: [{auprc_lower:.4f}, {auprc_upper:.4f}])")
    print(f"AUROC:        {auroc_point:.4f}  (95% CI: [{auroc_lower:.4f}, {auroc_upper:.4f}])")
    print("="*60)
    
    return results


if __name__ == '__main__':
    main()
