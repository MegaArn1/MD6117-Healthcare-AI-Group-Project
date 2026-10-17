"""
L4 calibration and subgroup analysis on the frozen test predictions.

No model retraining, no hyperparameter choices. Just additional metrics on the
same prediction vector that was already scored once. Per 实验方案 §5.7, this
does not constitute a second look as long as we declare what we compute before
we run it.

What we compute:
  1. Calibration: reliability curve (10 bins) + Brier score
  2. Subgroup AUROC: all four one-hot ICU types
  3. Fill the min(Se,PPV) gaps: L0 random, L1 age/gcs/bun at recall-0.80 threshold
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_data, get_split
from metrics import compute_metrics, find_threshold_for_recall, min_se_ppv

R = Path(__file__).resolve().parents[2] / "results"
M = Path(__file__).resolve().parents[2] / "models"
FIG = R / 'figures'; FIG.mkdir(exist_ok=True)

print("="*70)
print("L4 Calibration, Subgroup, and min(Se,PPV) Gap-Fill")
print("="*70)

# ========== 1. Calibration on frozen L4 test predictions ==========
print("\n1. Calibration (L4 XGBoost, test set)")
labels, X_unimputed, _ = load_data()
test_df = labels[labels['split'] == 'test'].merge(X_unimputed, on='RecordID', validate='one_to_one')
feat = [c for c in X_unimputed.columns if c != 'RecordID']
X_test, y_test = test_df[feat].values, test_df['In-hospital_death'].values

model = xgb.XGBClassifier(); model.load_model(str(M / 'L4_xgboost_best.json'))
p_test = model.predict_proba(X_test)[:, 1]

frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy='uniform')
brier = brier_score_loss(y_test, p_test)
print(f"   Brier score: {brier:.4f}")
print(f"   Reliability curve: {len(frac_pos)} bins")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
ax.plot(mean_pred, frac_pos, 's-', lw=2, label=f'L4 XGBoost (Brier {brier:.3f})')
ax.set_xlabel('Mean predicted probability', fontsize=11)
ax.set_ylabel('Fraction of positives', fontsize=11)
ax.set_title('Calibration Curve (Test Set, n=1,775)', fontsize=12, pad=10)
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(FIG / 'L4_calibration_curve.png', dpi=150, bbox_inches='tight')
print(f"   Saved figure -> {FIG.relative_to(Path(__file__).resolve().parents[2])}/L4_calibration_curve.png")

calibration_result = {
    'model': 'L4_xgboost', 'dataset': 'test', 'n': int(len(y_test)),
    'brier_score': float(brier),
    'reliability_curve': {
        'n_bins': len(frac_pos),
        'mean_predicted_prob': mean_pred.tolist(),
        'fraction_of_positives': frac_pos.tolist(),
    }
}

# ========== 2. Subgroup analysis by ICUType ==========
print("\n2. Subgroup analysis (static_icutype_2, test set)")
# ICUType is represented by four one-hot columns, never integer codes in one column.
subgroups = []
icu_names = {1: 'Coronary', 2: 'Cardiac surgery recovery', 3: 'Medical', 4: 'Surgical'}
for code, name in icu_names.items():
    mask = test_df[f'static_icutype_{code}'].eq(1).to_numpy()
    y_sub, p_sub = y_test[mask], p_test[mask]
    m = compute_metrics(y_sub, p_sub)
    subgroups.append({'icutype': name, 'code': code, 'n': int(mask.sum()),
                      'deaths': int(y_sub.sum()), 'auroc': float(m['auroc'])})
subgroup_result = {
    'stratification_variable': 'static_icutype_1 through static_icutype_4 (one-hot)',
    'overall_test_auroc': float(compute_metrics(y_test, p_test)['auroc']),
    'subgroups': subgroups,
}

# ========== 3. Fill min(Se,PPV) gaps for L0/L1 ==========
print("\n3. Filling min(Se,PPV) for L0 random and L1 single-variable baselines")
Xte_full, yte_full = get_split('test')
Xva_full, yva_full = get_split('validation')

gap_fills = {}
# L0 random
# Match the original L0 test vector; draw validation from a separate fixed seed.
p_rand_test = np.random.RandomState(42).rand(len(yte_full))
p_rand_val = np.random.RandomState(43).rand(len(yva_full))
thr_rand = find_threshold_for_recall(yva_full, p_rand_val, target_recall=0.80)
mse_rand = min_se_ppv(yte_full, (p_rand_test >= thr_rand).astype(int))
gap_fills['L0_random'] = {'threshold': float(thr_rand), 'test_min_se_ppv': float(mse_rand)}
print(f"   L0 random:     threshold {thr_rand:.4f}, test min(Se,PPV) {mse_rand:.4f}")

# L1 single variables
for var in ['static_age_years', 'gcs__0_48h__min', 'bun__0_48h__max']:
    direction = -1 if var.startswith('gcs') else 1
    thr_v = find_threshold_for_recall(yva_full, direction * Xva_full[var].values, target_recall=0.80)
    mse_v = min_se_ppv(yte_full, (direction * Xte_full[var].values >= thr_v).astype(int))
    gap_fills[f'L1_{var}'] = {'threshold': float(thr_v), 'test_min_se_ppv': float(mse_v)}
    print(f"   L1 {var:24s}: threshold {thr_v:.4f}, test min(Se,PPV) {mse_v:.4f}")

# ========== Save ==========
output = {
    'calibration': calibration_result,
    'subgroup_analysis': subgroup_result,
    'min_se_ppv_gap_fills': gap_fills,
    '_note': ('Computed on frozen L4 test predictions. No model retraining or hyperparameter '
              'choices. Per 实验方案 §5.7, additional metrics on the same prediction vector '
              'do not constitute a second look.')
}
json.dump(output, open(R / 'L4_calibration_and_subgroup.json', 'w'), indent=2)
print(f"\nSaved -> {R.relative_to(Path(__file__).resolve().parents[2])}/L4_calibration_and_subgroup.json")
print("="*70)
