"""
L1 Static Age Baseline - Single-feature mortality prediction using static_age_years.

Uses median imputation trained on train set, evaluates static age as sole predictor
for In-hospital Death. Establishes the simplest clinical baseline.
"""
import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.metrics import compute_metrics, bootstrap_ci


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Loading data...")
    labels_df, X_unimputed, _ = load_data()
    
    # Extract static_age_years (column index 1, after RecordID)
    age_col_name = 'static_age_years'
    
    # Split by train/validation/test
    train_mask = labels_df['split'] == 'train'
    val_mask = labels_df['split'] == 'validation'
    test_mask = labels_df['split'] == 'test'
    
    # Merge labels with features on RecordID
    data = labels_df.merge(X_unimputed[['RecordID', age_col_name]], on='RecordID')
    
    # Extract age for each split
    age_train = data.loc[train_mask, age_col_name].values
    age_val = data.loc[val_mask, age_col_name].values
    age_test = data.loc[test_mask, age_col_name].values
    
    y_train = data.loc[train_mask, 'In-hospital_death'].values
    y_val = data.loc[val_mask, 'In-hospital_death'].values
    y_test = data.loc[test_mask, 'In-hospital_death'].values
    
    print(f"Train: {len(age_train)} samples, {np.isnan(age_train).sum()} missing age values")
    print(f"Validation: {len(age_val)} samples, {np.isnan(age_val).sum()} missing age values")
    print(f"Test: {len(age_test)} samples, {np.isnan(age_test).sum()} missing age values")
    
    # Compute median age from training set (excluding NaN)
    train_age_median = np.nanmedian(age_train)
    print(f"\nTrain set age median: {train_age_median:.2f} years")
    
    # Impute missing values with train median
    age_train_imputed = np.where(np.isnan(age_train), train_age_median, age_train)
    age_val_imputed = np.where(np.isnan(age_val), train_age_median, age_val)
    age_test_imputed = np.where(np.isnan(age_test), train_age_median, age_test)
    
    # Normalize age to [0, 1] range for use as scores
    # Use train min/max for normalization
    age_min = age_train_imputed.min()
    age_max = age_train_imputed.max()
    
    print(f"Train age range: [{age_min:.2f}, {age_max:.2f}]")
    
    # Normalize to [0, 1]
    y_score_test = (age_test_imputed - age_min) / (age_max - age_min)
    
    # Evaluate on test set
    n_samples = len(y_test)
    prevalence = y_test.mean()
    
    print(f"\nTest set: {n_samples} samples, prevalence = {prevalence:.4f}")
    
    # Compute point estimates
    print("\nComputing AUPRC and AUROC...")
    metrics = compute_metrics(y_test, y_score_test)
    auprc_point = metrics['auprc']
    auroc_point = metrics['auroc']
    
    # Compute bootstrap confidence intervals
    print("Computing 95% bootstrap confidence intervals (1000 iterations)...")
    
    auprc_estimate, auprc_lower, auprc_upper = bootstrap_ci(
        y_test, y_score_test,
        metric_fn=lambda y_t, y_s: compute_metrics(y_t, y_s)['auprc'],
        n_bootstrap=1000,
        ci=0.95
    )
    
    auroc_estimate, auroc_lower, auroc_upper = bootstrap_ci(
        y_test, y_score_test,
        metric_fn=lambda y_t, y_s: compute_metrics(y_t, y_s)['auroc'],
        n_bootstrap=1000,
        ci=0.95
    )
    
    # Compute missing fraction
    missing_fraction = np.isnan(age_test).sum() / len(age_test)
    
    # Package results
    results = {
        'model': 'L1_static_age_years',
        'description': 'Single-feature baseline using static_age_years with median imputation',
        'variable': 'static_age_years',
        'test_n': int(n_samples),
        'missing_fraction': float(missing_fraction),
        'prevalence': float(prevalence),
        'auprc': float(auprc_point),
        'auprc_ci': [float(auprc_lower), float(auprc_upper)],
        'auroc': float(auroc_point),
        'auroc_ci': [float(auroc_lower), float(auroc_upper)],
        'train_median': float(train_age_median),
        'train_min': float(age_min),
        'train_max': float(age_max)
    }
    
    # Save results
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "L1_static_age_years.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print formatted results
    print("\n" + "="*60)
    print("L1 STATIC AGE BASELINE RESULTS")
    print("="*60)
    print(f"Variable:     static_age_years")
    print(f"Test samples: {n_samples:,}")
    print(f"Missing:      {missing_fraction:.1%}")
    print(f"Prevalence:   {prevalence:.4f}")
    print()
    print(f"AUPRC:        {auprc_point:.4f}  (95% CI: [{auprc_lower:.4f}, {auprc_upper:.4f}])")
    print(f"AUROC:        {auroc_point:.4f}  (95% CI: [{auroc_lower:.4f}, {auroc_upper:.4f}])")
    print("="*60)
    
    return results


if __name__ == '__main__':
    main()
