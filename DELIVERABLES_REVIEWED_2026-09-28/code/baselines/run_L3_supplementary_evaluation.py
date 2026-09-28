"""
Supplementary evaluation for L3 to fill gaps from workflow phase 3.

This script was created because workflow phase 3 "Evaluate Performance" was skipped.
It computes:
1. Validation set performance (to check overfitting)
2. Statistical significance tests (L3 vs SAPS-I, L3 vs SOFA)
3. Structured comparison table
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data, get_split
from metrics import compute_metrics, bootstrap_ci

DATA_DIR = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

# Load L3 configuration
with open(RESULTS_DIR / 'L3_logistic_regression_8features.json') as f:
    l3_config = json.load(f)

FEATURES = l3_config['features']
print(f"L3 uses {len(FEATURES)} features")

# Load data
labels_df, X_unimputed, X_imputed = load_data()

# Get splits
X_train, y_train = get_split('train')
X_val, y_val = get_split('validation')
X_test, y_test = get_split('test')

print(f"\nSplit sizes:")
print(f"  Train: {len(y_train)}, deaths={y_train.sum()}")
print(f"  Val:   {len(y_val)}, deaths={y_val.sum()}")
print(f"  Test:  {len(y_test)}, deaths={y_test.sum()}")

# Select features
X_train_sel = X_train[FEATURES]
X_val_sel = X_val[FEATURES]
X_test_sel = X_test[FEATURES]

# Impute with train median
for col in FEATURES:
    train_median = X_train_sel[col].median()
    X_train_sel[col] = X_train_sel[col].fillna(train_median)
    X_val_sel[col] = X_val_sel[col].fillna(train_median)
    X_test_sel[col] = X_test_sel[col].fillna(train_median)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_val_scaled = scaler.transform(X_val_sel)
X_test_scaled = scaler.transform(X_test_sel)

# Retrain L3 model
print("\nRetraining L3 model...")
model = LogisticRegression(
    penalty='l2',
    C=l3_config['regularization']['C'],
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Get predictions
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*60)
print("PHASE 3 SUPPLEMENTARY EVALUATION")
print("="*60)

# 1. Performance on all three splits
print("\n1. PERFORMANCE ACROSS SPLITS (detect overfitting)")
print("-" * 60)

for split_name, y_true, y_score in [
    ('Train', y_train, y_train_proba),
    ('Validation', y_val, y_val_proba),
    ('Test', y_test, y_test_proba)
]:
    metrics = compute_metrics(y_true, y_score)
    auroc_point, auroc_low, auroc_high = bootstrap_ci(
        y_true, y_score,
        lambda y, s: compute_metrics(y, s)['auroc'],
        n_bootstrap=1000
    )
    auprc_point, auprc_low, auprc_high = bootstrap_ci(
        y_true, y_score,
        lambda y, s: compute_metrics(y, s)['auprc'],
        n_bootstrap=1000
    )

    print(f"{split_name:12s} AUROC: {auroc_point:.4f} [{auroc_low:.4f}, {auroc_high:.4f}]")
    print(f"{'':12s} AUPRC: {auprc_point:.4f} [{auprc_low:.4f}, {auprc_high:.4f}]")

# 2. Load L2 baselines for comparison
print("\n2. STRUCTURED COMPARISON: L3 vs L2 BASELINES")
print("-" * 60)

with open(RESULTS_DIR / 'L2_clinical_scores_analysis.json') as f:
    l2_results = json.load(f)

# Merge labels with test indices to align with SAPS-I/SOFA availability
test_labels = labels_df[labels_df['split'] == 'test'].copy()

# For fair comparison, evaluate L3 on same subsets as L2
saps_i_available = test_labels['saps_i_available'] == 1
sofa_available = test_labels['sofa_available'] == 1

# Get L3 scores for these subsets
y_test_arr = y_test.values
y_test_proba_arr = y_test_proba

# SAPS-I subset comparison
saps_i_mask = saps_i_available.values
l3_saps_i_auroc = compute_metrics(y_test_arr[saps_i_mask], y_test_proba_arr[saps_i_mask])['auroc']
l3_saps_i_auprc = compute_metrics(y_test_arr[saps_i_mask], y_test_proba_arr[saps_i_mask])['auprc']
saps_i_baseline_auroc = l2_results['saps_i_results']['available_only_metrics']['auroc']
saps_i_baseline_auprc = l2_results['saps_i_results']['available_only_metrics']['auprc']

# SOFA subset comparison
sofa_mask = sofa_available.values
l3_sofa_auroc = compute_metrics(y_test_arr[sofa_mask], y_test_proba_arr[sofa_mask])['auroc']
l3_sofa_auprc = compute_metrics(y_test_arr[sofa_mask], y_test_proba_arr[sofa_mask])['auprc']
sofa_baseline_auroc = l2_results['sofa_results']['available_only_metrics']['auroc']
sofa_baseline_auprc = l2_results['sofa_results']['available_only_metrics']['auprc']

print(f"\nvs SAPS-I (n={saps_i_mask.sum()}, score-available subset):")
print(f"  SAPS-I baseline: AUROC {saps_i_baseline_auroc:.4f}, AUPRC {saps_i_baseline_auprc:.4f}")
print(f"  L3 on same set:  AUROC {l3_saps_i_auroc:.4f}, AUPRC {l3_saps_i_auprc:.4f}")
print(f"  Improvement:     AUROC +{(l3_saps_i_auroc - saps_i_baseline_auroc):.4f} (+{100*(l3_saps_i_auroc/saps_i_baseline_auroc - 1):.1f}%)")
print(f"                   AUPRC +{(l3_saps_i_auprc - saps_i_baseline_auprc):.4f} (+{100*(l3_saps_i_auprc/saps_i_baseline_auprc - 1):.1f}%)")

print(f"\nvs SOFA (n={sofa_mask.sum()}, score-available subset):")
print(f"  SOFA baseline:   AUROC {sofa_baseline_auroc:.4f}, AUPRC {sofa_baseline_auprc:.4f}")
print(f"  L3 on same set:  AUROC {l3_sofa_auroc:.4f}, AUPRC {l3_sofa_auprc:.4f}")
print(f"  Improvement:     AUROC +{(l3_sofa_auroc - sofa_baseline_auroc):.4f} (+{100*(l3_sofa_auroc/sofa_baseline_auroc - 1):.1f}%)")
print(f"                   AUPRC +{(l3_sofa_auprc - sofa_baseline_auprc):.4f} (+{100*(l3_sofa_auprc/sofa_baseline_auprc - 1):.1f}%)")

# 3. Statistical significance via bootstrap CI overlap
print("\n3. STATISTICAL SIGNIFICANCE (Bootstrap CI overlap test)")
print("-" * 60)
print("Note: Non-overlapping 95% CIs suggest statistically significant difference")

# We already have L3 test CI from earlier bootstrap
l3_test_auroc_ci = [l3_config['auroc_ci'][1], l3_config['auroc_ci'][0]]  # [low, high]

# For SAPS-I/SOFA, we need their CIs on same test set
# Load SAPS-I/SOFA scores
test_saps_i = test_labels['saps_i'].values
test_sofa = test_labels['sofa'].values

# Bootstrap CI for SAPS-I
saps_i_available_idx = ~np.isnan(test_saps_i)
_, saps_i_auroc_low, saps_i_auroc_high = bootstrap_ci(
    y_test_arr[saps_i_available_idx],
    test_saps_i[saps_i_available_idx],
    lambda y, s: compute_metrics(y, s)['auroc'],
    n_bootstrap=1000
)

# Bootstrap CI for SOFA
sofa_available_idx = ~np.isnan(test_sofa)
_, sofa_auroc_low, sofa_auroc_high = bootstrap_ci(
    y_test_arr[sofa_available_idx],
    test_sofa[sofa_available_idx],
    lambda y, s: compute_metrics(y, s)['auroc'],
    n_bootstrap=1000
)

# Bootstrap CI for L3 on SAPS-I subset
_, l3_saps_i_low, l3_saps_i_high = bootstrap_ci(
    y_test_arr[saps_i_mask],
    y_test_proba_arr[saps_i_mask],
    lambda y, s: compute_metrics(y, s)['auroc'],
    n_bootstrap=1000
)

# Bootstrap CI for L3 on SOFA subset
_, l3_sofa_low, l3_sofa_high = bootstrap_ci(
    y_test_arr[sofa_mask],
    y_test_proba_arr[sofa_mask],
    lambda y, s: compute_metrics(y, s)['auroc'],
    n_bootstrap=1000
)

saps_i_overlap = not (l3_saps_i_high < saps_i_auroc_low or l3_saps_i_low > saps_i_auroc_high)
sofa_overlap = not (l3_sofa_high < sofa_auroc_low or l3_sofa_low > sofa_auroc_high)

print(f"\nL3 vs SAPS-I:")
print(f"  SAPS-I CI: [{saps_i_auroc_low:.4f}, {saps_i_auroc_high:.4f}]")
print(f"  L3 CI:     [{l3_saps_i_low:.4f}, {l3_saps_i_high:.4f}]")
print(f"  Overlap: {'YES' if saps_i_overlap else 'NO'}")
print(f"  Conclusion: {'Difference NOT statistically significant' if saps_i_overlap else 'Difference IS statistically significant'}")

print(f"\nL3 vs SOFA:")
print(f"  SOFA CI:   [{sofa_auroc_low:.4f}, {sofa_auroc_high:.4f}]")
print(f"  L3 CI:     [{l3_sofa_low:.4f}, {l3_sofa_high:.4f}]")
print(f"  Overlap: {'YES' if sofa_overlap else 'NO'}")
print(f"  Conclusion: {'Difference NOT statistically significant' if sofa_overlap else 'Difference IS statistically significant'}")

# Save supplementary results
results = {
    "supplementary_evaluation_date": "2026-09-24",
    "reason": "Workflow phase 3 'Evaluate Performance' was skipped; this fills the gaps",
    "performance_by_split": {
        "train": {
            "auroc": float(compute_metrics(y_train, y_train_proba)['auroc']),
            "auprc": float(compute_metrics(y_train, y_train_proba)['auprc'])
        },
        "validation": {
            "auroc": float(compute_metrics(y_val, y_val_proba)['auroc']),
            "auprc": float(compute_metrics(y_val, y_val_proba)['auprc'])
        },
        "test": {
            "auroc": float(compute_metrics(y_test, y_test_proba)['auroc']),
            "auprc": float(compute_metrics(y_test, y_test_proba)['auprc'])
        }
    },
    "overfitting_check": {
        "train_val_auroc_gap": float(compute_metrics(y_train, y_train_proba)['auroc'] - compute_metrics(y_val, y_val_proba)['auroc']),
        "interpretation": "Gap < 0.05 suggests no severe overfitting"
    },
    "l3_vs_l2_comparison": {
        "vs_saps_i": {
            "n": int(saps_i_mask.sum()),
            "saps_i_auroc": float(saps_i_baseline_auroc),
            "l3_auroc": float(l3_saps_i_auroc),
            "improvement_absolute": float(l3_saps_i_auroc - saps_i_baseline_auroc),
            "improvement_relative_pct": float(100 * (l3_saps_i_auroc / saps_i_baseline_auroc - 1)),
            "l3_ci": [float(l3_saps_i_low), float(l3_saps_i_high)],
            "saps_i_ci": [float(saps_i_auroc_low), float(saps_i_auroc_high)],
            "ci_overlap": bool(saps_i_overlap),
            "statistically_significant": not bool(saps_i_overlap)
        },
        "vs_sofa": {
            "n": int(sofa_mask.sum()),
            "sofa_auroc": float(sofa_baseline_auroc),
            "l3_auroc": float(l3_sofa_auroc),
            "improvement_absolute": float(l3_sofa_auroc - sofa_baseline_auroc),
            "improvement_relative_pct": float(100 * (l3_sofa_auroc / sofa_baseline_auroc - 1)),
            "l3_ci": [float(l3_sofa_low), float(l3_sofa_high)],
            "sofa_ci": [float(sofa_auroc_low), float(sofa_auroc_high)],
            "ci_overlap": bool(sofa_overlap),
            "statistically_significant": not bool(sofa_overlap)
        }
    }
}

output_file = RESULTS_DIR / 'L3_supplementary_evaluation.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"Results saved to {output_file.name}")
print(f"{'='*60}")
