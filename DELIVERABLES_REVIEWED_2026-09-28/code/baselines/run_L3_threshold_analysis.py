"""
Stage 2: L3 Threshold Analysis

Find optimal threshold on validation set for target recall=0.80,
then evaluate on test set. Compare to clinical baselines.

Rationale from 实验方案 §5.5:
- L3 is the real opponent for L4 GBDT, not SAPS-I
- Missing a death (FN) costs far more than unnecessary review (FP)
- Target high recall (0.80) to minimize missed deaths
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data, get_split
from metrics import compute_metrics, find_threshold_for_recall, min_se_ppv

# Features used in L3 model (from previous training)
SELECTED_FEATURES = [
    'bun__0_48h__max',
    'static_age_years',
    'gcs__0_48h__min',
    'urine__0_48h__min',
    'lactate__0_48h__max',
    'platelets__0_48h__min',
    'creatinine__0_48h__max',
    'lactate__0_48h__measured',
]

def find_optimal_threshold_for_baseline(y_true, scores, target_recall=0.80):
    """
    Find threshold for SAPS-I or SOFA that achieves target recall.
    
    Clinical scores: higher score = higher risk, so threshold is "score >= threshold"
    """
    # Remove missing scores
    mask = ~np.isnan(scores)
    y_true_valid = y_true[mask]
    scores_valid = scores[mask]
    
    if len(y_true_valid) == 0:
        return None, {}
    
    # Try different thresholds
    unique_scores = np.sort(np.unique(scores_valid))
    best_threshold = None
    best_metrics = None
    min_diff = float('inf')
    
    for threshold in unique_scores:
        y_pred = (scores_valid >= threshold).astype(int)
        
        # Calculate recall
        tp = np.sum((y_pred == 1) & (y_true_valid == 1))
        fn = np.sum((y_pred == 0) & (y_true_valid == 1))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Find threshold that gets closest to target recall from above
        if recall >= target_recall:
            diff = abs(recall - target_recall)
            if diff < min_diff:
                min_diff = diff
                best_threshold = threshold
                
                # Calculate all metrics
                tn = np.sum((y_pred == 0) & (y_true_valid == 0))
                fp = np.sum((y_pred == 1) & (y_true_valid == 0))
                
                sensitivity = recall
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                best_metrics = {
                    'threshold': float(threshold),
                    'n_samples': int(len(y_true_valid)),
                    'sensitivity': float(sensitivity),
                    'ppv': float(ppv),
                    'specificity': float(specificity),
                    'min_se_ppv': float(min(sensitivity, ppv))
                }
    
    return best_threshold, best_metrics

def main():
    print("=" * 80)
    print("L3 THRESHOLD ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    labels_df, _, _ = load_data()
    
    # Get splits (already imputed)
    X_train, y_train = get_split("train")
    X_val, y_val = get_split("validation")
    X_test, y_test = get_split("test")
    
    print(f"Train: {len(y_train)} samples ({y_train.sum()} deaths, {y_train.mean():.4f} prevalence)")
    print(f"Validation: {len(y_val)} samples ({y_val.sum()} deaths, {y_val.mean():.4f} prevalence)")
    print(f"Test: {len(y_test)} samples ({y_test.sum()} deaths, {y_test.mean():.4f} prevalence)")
    print()
    
    # Select features
    X_train_selected = X_train[SELECTED_FEATURES].copy()
    X_val_selected = X_val[SELECTED_FEATURES].copy()
    X_test_selected = X_test[SELECTED_FEATURES].copy()
    
    # Impute missing values (median from training set)
    print("Imputing missing values...")
    imputation_values = {}
    for col in SELECTED_FEATURES:
        median_val = X_train_selected[col].median()
        imputation_values[col] = median_val
        X_train_selected[col].fillna(median_val, inplace=True)
        X_val_selected[col].fillna(median_val, inplace=True)
        X_test_selected[col].fillna(median_val, inplace=True)
    print()
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    print()
    
    # Train L3 model
    print("Training L3 logistic regression...")
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    print("Training complete.")
    print()
    
    # Get predictions on validation and test sets
    y_score_val = model.predict_proba(X_val_scaled)[:, 1]
    y_score_test = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate AUROC on both sets
    auroc_val = roc_auc_score(y_val, y_score_val)
    auroc_test = roc_auc_score(y_test, y_score_test)
    print(f"L3 AUROC (validation): {auroc_val:.4f}")
    print(f"L3 AUROC (test): {auroc_test:.4f}")
    print()
    
    # Find threshold on validation set for target recall=0.80
    print("=" * 80)
    print("THRESHOLD SELECTION ON VALIDATION SET")
    print("=" * 80)
    TARGET_RECALL = 0.80
    print(f"Target recall: {TARGET_RECALL}")
    print()
    
    threshold = find_threshold_for_recall(y_val, y_score_val, target_recall=TARGET_RECALL)
    print(f"Selected threshold: {threshold:.6f}")
    print()
    
    # Evaluate on validation set at this threshold
    val_metrics = compute_metrics(y_val, y_score_val, threshold=threshold)
    print("Validation set performance at threshold:")
    print(f"  Sensitivity: {val_metrics['sensitivity']:.4f}")
    print(f"  PPV:         {val_metrics['ppv']:.4f}")
    print(f"  Specificity: {val_metrics['specificity']:.4f}")
    print(f"  min(Se, PPV): {min(val_metrics['sensitivity'], val_metrics['ppv']):.4f}")
    print()
    
    # Apply threshold to test set
    print("=" * 80)
    print("TEST SET PERFORMANCE")
    print("=" * 80)
    print()
    
    test_metrics = compute_metrics(y_test, y_score_test, threshold=threshold)
    print(f"L3 Logistic Regression (threshold={threshold:.6f}):")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  PPV:         {test_metrics['ppv']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  min(Se, PPV): {min(test_metrics['sensitivity'], test_metrics['ppv']):.4f}")
    print()
    # Load and evaluate clinical baseline scores
    print("=" * 80)
    print("CLINICAL BASELINE COMPARISON")
    print("=" * 80)
    print()
    
    # Get baseline scores by split
    test_baseline = labels_df[labels_df['split'] == 'test'].copy()
    val_baseline = labels_df[labels_df['split'] == 'validation'].copy()
    
    # Verify label alignment
    y_test_baseline = test_baseline['In-hospital_death'].values
    y_val_baseline = val_baseline['In-hospital_death'].values
    assert len(y_test_baseline) == len(y_test), f"Test label count mismatch: {len(y_test_baseline)} vs {len(y_test)}"
    assert len(y_val_baseline) == len(y_val), f"Val label count mismatch: {len(y_val_baseline)} vs {len(y_val)}"
    
    # SAPS-I analysis
    print("SAPS-I:")
    saps_scores_test = test_baseline['saps_i'].values
    saps_scores_val = val_baseline['saps_i'].values
    
    n_saps_available = test_baseline['saps_i_available'].sum()
    print(f"  Available: {n_saps_available}/{len(y_test)} ({n_saps_available/len(y_test)*100:.1f}%)")
    
    # Find optimal SAPS-I threshold on validation set
    saps_threshold_val, _ = find_optimal_threshold_for_baseline(
        y_val_baseline, saps_scores_val, target_recall=TARGET_RECALL
    )
    
    if saps_threshold_val is not None:
        # Evaluate using validation threshold on test set
        mask = ~np.isnan(saps_scores_test)
        y_true_saps = y_test_baseline[mask]
        scores_saps = saps_scores_test[mask]
        y_pred_saps = (scores_saps >= saps_threshold_val).astype(int)
        
        tp = np.sum((y_pred_saps == 1) & (y_true_saps == 1))
        fp = np.sum((y_pred_saps == 1) & (y_true_saps == 0))
        tn = np.sum((y_pred_saps == 0) & (y_true_saps == 0))
        fn = np.sum((y_pred_saps == 0) & (y_true_saps == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        print(f"  Threshold (from validation): {saps_threshold_val:.1f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  PPV:         {ppv:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  min(Se, PPV): {min(sensitivity, ppv):.4f}")
        
        saps_result = {
            'threshold': float(saps_threshold_val),
            'n_samples': int(len(y_true_saps)),
            'sensitivity': float(sensitivity),
            'ppv': float(ppv),
            'specificity': float(specificity),
            'min_se_ppv': float(min(sensitivity, ppv))
        }
    else:
        print("  Could not find threshold achieving target recall")
        saps_result = None
    print()
    
    # SOFA analysis
    print("SOFA:")
    sofa_scores_test = test_baseline['sofa'].values
    sofa_scores_val = val_baseline['sofa'].values
    
    n_sofa_available = test_baseline['sofa_available'].sum()
    print(f"  Available: {n_sofa_available}/{len(y_test)} ({n_sofa_available/len(y_test)*100:.1f}%)")
    
    # Find optimal SOFA threshold on validation set
    sofa_threshold_val, _ = find_optimal_threshold_for_baseline(
        y_val_baseline, sofa_scores_val, target_recall=TARGET_RECALL
    )
    
    if sofa_threshold_val is not None:
        # Evaluate using validation threshold on test set
        mask = ~np.isnan(sofa_scores_test)
        y_true_sofa = y_test_baseline[mask]
        scores_sofa = sofa_scores_test[mask]
        y_pred_sofa = (scores_sofa >= sofa_threshold_val).astype(int)
        
        tp = np.sum((y_pred_sofa == 1) & (y_true_sofa == 1))
        fp = np.sum((y_pred_sofa == 1) & (y_true_sofa == 0))
        tn = np.sum((y_pred_sofa == 0) & (y_true_sofa == 0))
        fn = np.sum((y_pred_sofa == 0) & (y_true_sofa == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        print(f"  Threshold (from validation): {sofa_threshold_val:.1f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  PPV:         {ppv:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  min(Se, PPV): {min(sensitivity, ppv):.4f}")
        
        sofa_result = {
            'threshold': float(sofa_threshold_val),
            'n_samples': int(len(y_true_sofa)),
            'sensitivity': float(sensitivity),
            'ppv': float(ppv),
            'specificity': float(specificity),
            'min_se_ppv': float(min(sensitivity, ppv))
        }
    else:
        print("  Could not find threshold achieving target recall")
        sofa_result = None
    print()
    
    # Save results
    results = {
        'model': 'L3_logistic_regression_threshold_analysis',
        'description': 'Threshold analysis: find optimal threshold on validation, evaluate on test',
        'target_recall': TARGET_RECALL,
        'rationale': 'Missing a death (FN) costs far more than unnecessary review (FP), so we target high recall',
        'validation_recall_achieved': float(val_metrics['sensitivity']),
        'selected_threshold': float(threshold),
        'test_metrics_at_threshold': {
            'n_samples': int(len(y_test)),
            'sensitivity': float(test_metrics['sensitivity']),
            'ppv': float(test_metrics['ppv']),
            'specificity': float(test_metrics['specificity']),
            'min_se_ppv': float(min(test_metrics['sensitivity'], test_metrics['ppv']))
        },
        'clinical_baselines': {
            'saps_i': saps_result,
            'sofa': sofa_result
        }
    }
    
    output_path = Path(__file__).parent.parent / 'results' / 'L3_threshold_analysis.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print()
    print("=" * 80)
    print("L3 THRESHOLD ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()