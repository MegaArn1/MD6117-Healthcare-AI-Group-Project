"""
L4 Test Evaluation with GPU

Evaluates selected model on test set with:
- Bootstrap 95% CI (1,000 resamples)
- Threshold selection at recall≈0.80 on validation
- Statistical significance test vs L3
- Comparison to all prior baselines
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data
from metrics import compute_metrics, bootstrap_ci, find_threshold_for_recall

print("="*60)
print("L4 Test Evaluation")
print("="*60)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
MODELS_DIR = Path(__file__).parent.parent / 'models'

# Load model selection
with open(RESULTS_DIR / 'L4_model_selection.json') as f:
    selection = json.load(f)

selected_model = selection['selected_model']
print(f"\nSelected model: {selected_model.upper()}")

# Load data
print("\n1. Loading data...")
labels_df, X_unimputed, _ = load_data()

val_df = labels_df[labels_df['split'] == 'validation'].merge(X_unimputed, on='RecordID')
test_df = labels_df[labels_df['split'] == 'test'].merge(X_unimputed, on='RecordID')

feature_cols = [c for c in X_unimputed.columns if c != 'RecordID']

X_val = val_df[feature_cols].values
y_val = val_df['In-hospital_death'].values
X_test = test_df[feature_cols].values
y_test = test_df['In-hospital_death'].values

print(f"   Val:  {len(y_val)} samples, {y_val.sum()} deaths")
print(f"   Test: {len(y_test)} samples, {y_test.sum()} deaths")

# Load trained model
print(f"\n2. Loading {selected_model.upper()} model...")
if selected_model == 'xgboost':
    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / 'L4_xgboost_best.json'))
    y_val_proba  = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
elif selected_model == 'catboost':
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / 'L4_catboost_best.cbm'))
    y_val_proba  = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
else:  # lightgbm
    model = lgb.Booster(model_file=str(MODELS_DIR / 'L4_lightgbm_best.txt'))
    y_val_proba  = model.predict(X_val)
    y_test_proba = model.predict(X_test)

# Threshold selection on validation at recall≈0.80
print("\n3. Threshold selection (validation, recall≈0.80)...")
threshold = find_threshold_for_recall(y_val, y_val_proba, target_recall=0.80)
val_threshold_metrics = compute_metrics(y_val, y_val_proba, threshold=threshold)
print(f"   Selected threshold: {threshold:.4f}")
print(f"   Val recall: {val_threshold_metrics['sensitivity']:.4f}")

# Test set evaluation
print("\n4. Test set evaluation...")
test_metrics = compute_metrics(y_test, y_test_proba, threshold=threshold)
print(f"   Test AUROC: {test_metrics['auroc']:.4f}")
print(f"   Test AUPRC: {test_metrics['auprc']:.4f}")
print(f"   At threshold {threshold:.3f}:")
print(f"     Sensitivity: {test_metrics['sensitivity']:.4f}")
print(f"     PPV: {test_metrics['ppv']:.4f}")
print(f"     Specificity: {test_metrics['specificity']:.4f}")

# Bootstrap 95% CI
print("\n5. Bootstrap 95% CI (1,000 resamples)...")
auroc_point, auroc_low, auroc_high = bootstrap_ci(
    y_test, y_test_proba,
    lambda y, s: compute_metrics(y, s)['auroc'],
    n_bootstrap=1000
)
auprc_point, auprc_low, auprc_high = bootstrap_ci(
    y_test, y_test_proba,
    lambda y, s: compute_metrics(y, s)['auprc'],
    n_bootstrap=1000
)

print(f"   AUROC: {auroc_point:.4f} [{auroc_low:.4f}, {auroc_high:.4f}]")
print(f"   AUPRC: {auprc_point:.4f} [{auprc_low:.4f}, {auprc_high:.4f}]")

# Load L3 results for comparison
print("\n6. Comparison to L3 baseline...")
with open(RESULTS_DIR / 'L3_logistic_regression_8features.json') as f:
    l3_results = json.load(f)

l3_auroc = l3_results['auroc']
l3_auroc_ci = l3_results['auroc_ci']

improvement_absolute = auroc_point - l3_auroc
improvement_relative_pct = 100 * (auroc_point / l3_auroc - 1)

# CI overlap test for statistical significance
l4_ci_low, l4_ci_high = auroc_low, auroc_high
l3_ci_low, l3_ci_high = min(l3_auroc_ci), max(l3_auroc_ci)
ci_overlap = not (l4_ci_high < l3_ci_low or l4_ci_low > l3_ci_high)

print(f"   L3 AUROC: {l3_auroc:.4f} [{l3_ci_low:.4f}, {l3_ci_high:.4f}]")
print(f"   L4 AUROC: {auroc_point:.4f} [{l4_ci_low:.4f}, {l4_ci_high:.4f}]")
print(f"   Improvement: +{improvement_absolute:.4f} ({improvement_relative_pct:+.1f}%)")
print(f"   CI overlap: {'YES' if ci_overlap else 'NO'}")
print(f"   Statistically significant: {'NO' if ci_overlap else 'YES'}")

# Check literature target
literature_target_achieved = auroc_point >= 0.83
print(f"\n7. Literature target (AUROC 0.83-0.86): {'✓ ACHIEVED' if literature_target_achieved else '✗ NOT ACHIEVED'}")

# Save results
results = {
    'selected_model': selected_model,
    'n_features': len(feature_cols),
    'test_samples': int(len(y_test)),
    'prevalence': float(y_test.mean()),
    'threshold': {
        'value': float(threshold),
        'selection_method': 'recall_0.80_on_validation',
        'validation_recall_achieved': float(val_threshold_metrics['sensitivity'])
    },
    'test_performance': {
        'auroc': float(auroc_point),
        'auroc_ci': [float(auroc_low), float(auroc_high)],
        'auprc': float(auprc_point),
        'auprc_ci': [float(auprc_low), float(auprc_high)],
        'sensitivity': float(test_metrics['sensitivity']),
        'ppv': float(test_metrics['ppv']),
        'specificity': float(test_metrics['specificity']),
        'min_se_ppv': float(min(test_metrics['sensitivity'], test_metrics['ppv']))
    },
    'vs_l3_comparison': {
        'l3_auroc': float(l3_auroc),
        'l3_auroc_ci': [float(l3_ci_low), float(l3_ci_high)],
        'l4_auroc': float(auroc_point),
        'l4_auroc_ci': [float(l4_ci_low), float(l4_ci_high)],
        'improvement_absolute': float(improvement_absolute),
        'improvement_relative_pct': float(improvement_relative_pct),
        'ci_overlap': bool(ci_overlap),
        'statistically_significant': not bool(ci_overlap)
    },
    'literature_target_achieved': bool(literature_target_achieved),
    'literature_target_range': [0.83, 0.86]
}

output_file = RESULTS_DIR / 'L4_test_evaluation.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_file.name}")
print("="*60)
