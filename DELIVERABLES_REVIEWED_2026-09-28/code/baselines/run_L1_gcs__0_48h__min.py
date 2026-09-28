"""
L1 Single-Feature Baseline - GCS (Glasgow Coma Scale) minimum over 0-48h.

Uses gcs__0_48h__min as the sole predictor for In-hospital Death.
GCS ranges 3-15 (lower = worse neurological status).
We use -GCS as the risk score so lower GCS → higher predicted risk.

Handles missing values via median imputation fitted on training data.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import get_split
from src.metrics import compute_metrics, bootstrap_ci


def main():
    np.random.seed(42)
    
    print("="*70)
    print("L1 BASELINE: GCS (Glasgow Coma Scale) minimum over 0-48h")
    print("="*70)
    
    # Load splits
    print("\nLoading data splits...")
    X_train, y_train = get_split('train')
    X_val, y_val = get_split('validation')
    X_test, y_test = get_split('test')
    
    feature_name = 'gcs__0_48h__min'
    
    print(f"Train: {len(y_train)} samples, prevalence = {y_train.mean():.4f}")
    print(f"Val:   {len(y_val)} samples, prevalence = {y_val.mean():.4f}")
    print(f"Test:  {len(y_test)} samples, prevalence = {y_test.mean():.4f}")
    
    # Extract GCS feature
    gcs_train = X_train[feature_name].values.reshape(-1, 1)
    gcs_val = X_val[feature_name].values.reshape(-1, 1)
    gcs_test = X_test[feature_name].values.reshape(-1, 1)
    
    # Check missingness
    train_missing_pct = np.isnan(gcs_train).mean() * 100
    val_missing_pct = np.isnan(gcs_val).mean() * 100
    test_missing_pct = np.isnan(gcs_test).mean() * 100
    
    print(f"\nMissing GCS values:")
    print(f"  Train: {train_missing_pct:.2f}%")
    print(f"  Val:   {val_missing_pct:.2f}%")
    print(f"  Test:  {test_missing_pct:.2f}%")
    
    # Fit median imputer on training data
    print("\nFitting median imputer on training data...")
    imputer = SimpleImputer(strategy='median')
    imputer.fit(gcs_train)
    train_median = imputer.statistics_[0]
    
    print(f"Training median GCS: {train_median:.2f}")
    
    # Transform all splits
    gcs_train_imputed = imputer.transform(gcs_train).flatten()
    gcs_val_imputed = imputer.transform(gcs_val).flatten()
    gcs_test_imputed = imputer.transform(gcs_test).flatten()
    
    print(f"\nGCS statistics (after imputation):")
    print(f"  Train: min={gcs_train_imputed.min():.1f}, max={gcs_train_imputed.max():.1f}, "
          f"mean={gcs_train_imputed.mean():.2f}, std={gcs_train_imputed.std():.2f}")
    print(f"  Test:  min={gcs_test_imputed.min():.1f}, max={gcs_test_imputed.max():.1f}, "
          f"mean={gcs_test_imputed.mean():.2f}, std={gcs_test_imputed.std():.2f}")
    
    # Convert GCS to risk score: lower GCS = higher risk
    # Use -GCS so that higher score = higher predicted mortality risk
    print("\nConverting GCS to risk score (using -GCS)...")
    y_score_test = -gcs_test_imputed
    
    # Normalize to [0,1] range for interpretability
    score_min = y_score_test.min()
    score_max = y_score_test.max()
    y_score_test_normalized = (y_score_test - score_min) / (score_max - score_min)
    
    print(f"Risk score range: [{y_score_test_normalized.min():.4f}, {y_score_test_normalized.max():.4f}]")
    
    # Compute point estimates
    print("\nComputing AUPRC and AUROC on test set...")
    metrics = compute_metrics(y_test, y_score_test_normalized)
    auprc_point = metrics['auprc']
    auroc_point = metrics['auroc']
    
    print(f"  AUPRC: {auprc_point:.4f}")
    print(f"  AUROC: {auroc_point:.4f}")
    
    # Compute bootstrap confidence intervals
    print("\nComputing 95% bootstrap confidence intervals (1000 iterations)...")
    
    auprc_estimate, auprc_lower, auprc_upper = bootstrap_ci(
        y_test, y_score_test_normalized,
        metric_fn=lambda y_t, y_s: compute_metrics(y_t, y_s)['auprc'],
        n_bootstrap=1000,
        ci=0.95
    )
    
    auroc_estimate, auroc_lower, auroc_upper = bootstrap_ci(
        y_test, y_score_test_normalized,
        metric_fn=lambda y_t, y_s: compute_metrics(y_t, y_s)['auroc'],
        n_bootstrap=1000,
        ci=0.95
    )
    
    # Package results
    results = {
        'model': 'L1_gcs_0_48h_min',
        'description': 'Single feature: GCS (Glasgow Coma Scale) minimum over 0-48h',
        'feature': feature_name,
        'test_samples': int(len(y_test)),
        'prevalence': float(y_test.mean()),
        'missing_fraction': float(test_missing_pct / 100),
        'imputation_method': 'median',
        'train_median': float(train_median),
        'auprc': float(auprc_point),
        'auprc_ci': [float(auprc_lower), float(auprc_upper)],
        'auroc': float(auroc_point),
        'auroc_ci': [float(auroc_lower), float(auroc_upper)]
    }
    
    # Save results
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "L1_gcs__0_48h__min.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print formatted results
    print("\n" + "="*70)
    print("L1 GCS BASELINE RESULTS")
    print("="*70)
    print(f"Feature:      {feature_name}")
    print(f"Test samples: {len(y_test):,}")
    print(f"Prevalence:   {y_test.mean():.4f}")
    print(f"Missing data: {test_missing_pct:.2f}% (imputed with training median = {train_median:.2f})")
    print()
    print(f"AUPRC:        {auprc_point:.4f}  (95% CI: [{auprc_lower:.4f}, {auprc_upper:.4f}])")
    print(f"AUROC:        {auroc_point:.4f}  (95% CI: [{auroc_lower:.4f}, {auroc_upper:.4f}])")
    print("="*70)
    print("\nInterpretation:")
    print(f"  • AUROC {auroc_point:.4f} indicates {'weak' if auroc_point < 0.65 else 'moderate' if auroc_point < 0.75 else 'good'} discrimination")
    print(f"  • AUPRC {auprc_point:.4f} vs prevalence {y_test.mean():.4f}: "
          f"{'below' if auprc_point < y_test.mean() else 'above'} no-skill baseline")
    print(f"  • GCS alone {'has' if auroc_point > 0.7 else 'shows limited'} predictive value for in-hospital mortality")
    
    return results


if __name__ == '__main__':
    main()
