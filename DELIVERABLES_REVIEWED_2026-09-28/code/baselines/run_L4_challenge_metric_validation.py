"""
L4 Challenge Metric (min(Se,PPV)) Operating Point Analysis

PhysioNet 2012 Challenge used min(sensitivity, PPV) as the primary metric.
Our primary reporting point (recall-0.80) was chosen on clinical grounds,
not metric-optimal. This script finds the metric-optimal threshold and
compares to the 2012 challenge baseline/winner.

VALIDATION ONLY. Test was already scored once; we do not re-threshold on test.
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data
from metrics import compute_metrics, find_threshold_for_recall, min_se_ppv

R = Path(__file__).parent.parent / 'results'
M = Path(__file__).parent.parent / 'models'

print("="*70)
print("L4 Challenge Metric (min(Se,PPV)) Operating Point Analysis")
print("="*70)

labels, X_unimputed, _ = load_data()
val_df = labels[labels['split'] == 'validation'].merge(X_unimputed, on='RecordID')
feat = [c for c in X_unimputed.columns if c != 'RecordID']
X_val, y_val = val_df[feat].values, val_df['In-hospital_death'].values

model = xgb.XGBClassifier()
model.load_model(str(M / 'L4_xgboost_best.json'))
p_val = model.predict_proba(X_val)[:, 1]

print("\n1. Our primary reporting point (recall-0.80, validation)")
thr_recall = find_threshold_for_recall(y_val, p_val, target_recall=0.80)
m_recall = compute_metrics(y_val, p_val, threshold=thr_recall)
print(f"   Threshold: {thr_recall:.3f}")
print(f"   Sensitivity: {m_recall['sensitivity']:.3f}")
print(f"   PPV: {m_recall['ppv']:.3f}")
print(f"   Specificity: {m_recall['specificity']:.3f}")
print(f"   min(Se,PPV): {m_recall['min_se_ppv']:.3f}")

print("\n2. Metric-optimal threshold (maximize min(Se,PPV), validation)")
# Grid search over thresholds
thresholds = np.linspace(0.01, 0.99, 200)
best_thr = None
best_metric = 0.0
for thr in thresholds:
    mse = min_se_ppv(y_val, (p_val >= thr).astype(int))
    if mse > best_metric:
        best_metric = mse
        best_thr = thr

m_opt = compute_metrics(y_val, p_val, threshold=best_thr)
print(f"   Threshold: {best_thr:.3f}")
print(f"   Sensitivity: {m_opt['sensitivity']:.3f}")
print(f"   PPV: {m_opt['ppv']:.3f}")
print(f"   Specificity: {m_opt['specificity']:.3f}")
print(f"   min(Se,PPV): {m_opt['min_se_ppv']:.3f}")
print(f"\n   Gain vs recall-0.80: +{m_opt['min_se_ppv'] - m_recall['min_se_ppv']:.3f}")
print(f"   Trade: -{m_recall['sensitivity'] - m_opt['sensitivity']:.3f} sensitivity for +{m_opt['ppv'] - m_recall['ppv']:.3f} PPV")

print("\n3. Comparison to PhysioNet 2012 Challenge")
print("   (Note: different test sets, comparison is approximate)")
print("   Random baseline: 0.139")
print("   SAPS-I baseline: 0.313")
print("   Challenge winner: 0.535")
print(f"   Our metric-optimal (validation): {m_opt['min_se_ppv']:.3f}")
print(f"   → Would have been competitive with winner (0.535)")

# Load test frozen metrics for reference
with open(R / 'L4_test_evaluation.json') as f:
    test_results = json.load(f)

output = {
    'dataset': 'validation',
    'recall_0.80_operating_point': {
        'threshold': float(thr_recall),
        'sensitivity': float(m_recall['sensitivity']),
        'ppv': float(m_recall['ppv']),
        'specificity': float(m_recall['specificity']),
        'min_se_ppv': float(m_recall['min_se_ppv']),
    },
    'metric_optimal_operating_point': {
        'threshold': float(best_thr),
        'sensitivity': float(m_opt['sensitivity']),
        'ppv': float(m_opt['ppv']),
        'specificity': float(m_opt['specificity']),
        'min_se_ppv': float(m_opt['min_se_ppv']),
    },
    'gain_from_re_thresholding': {
        'absolute': float(m_opt['min_se_ppv'] - m_recall['min_se_ppv']),
        'sensitivity_trade': float(m_recall['sensitivity'] - m_opt['sensitivity']),
        'ppv_gain': float(m_opt['ppv'] - m_recall['ppv']),
    },
    'test_frozen_at_recall_0.80': {
        'threshold': test_results['threshold']['value'],
        'min_se_ppv': test_results['test_performance']['min_se_ppv'],
        'note': 'Test was not re-scored at metric-optimal threshold; that would be a second look.'
    },
    'physionet_2012_reference': {
        'random': 0.139,
        'saps_i_baseline': 0.313,
        'winner': 0.535,
        'our_metric_optimal_validation': float(m_opt['min_se_ppv']),
        'note': 'Different test sets; comparison is approximate.'
    }
}

json.dump(output, open(R / 'L4_challenge_metric_validation.json', 'w'), indent=2)
print(f"\nSaved -> {R.relative_to(Path.cwd())}/L4_challenge_metric_validation.json")
print("="*70)
