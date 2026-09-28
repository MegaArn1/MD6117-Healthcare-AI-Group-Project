"""
L3-revised: pre-declared rung, scored on test EXACTLY ONCE.

Background
----------
Stage 2's L3 chose gcs__0_48h__min (SOFA CNS uses the worst GCS in the window)
and urine__0_48h__min. Stage 3 found, on train+validation only, that
gcs__0_48h__last and urine__0_48h__total carry far more signal
(run_L3_gcs_statistic_ablation.py, variant E: validation AUROC 0.8081 vs
0.7217). Nothing about that selection used the test set.

Protocol for this script
------------------------
- The 8 features below are FIXED before this file was first executed. They are
  variant E of the ablation, copied verbatim. No search happens here.
- Recipe identical to the original L3: L2 logistic, C=1, class_weight balanced,
  StandardScaler fit on train only.
- Operating threshold picked on VALIDATION at recall 0.80, then applied to test.
- Test is scored once, in this run. If this file is re-run it must reproduce the
  same numbers; it must not be edited and re-run with different features.
- The original L3 rung (test AUROC 0.7417) is NOT replaced. Both rungs stand.
"""
import sys, json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import get_split
from metrics import compute_metrics, bootstrap_ci, find_threshold_for_recall, min_se_ppv

R = Path(__file__).parent.parent / 'results'

L3_ORIGINAL = ['bun__0_48h__max', 'static_age_years', 'gcs__0_48h__min', 'urine__0_48h__min',
               'lactate__0_48h__max', 'platelets__0_48h__min', 'creatinine__0_48h__max',
               'lactate__0_48h__measured']
L3_REVISED = ['bun__0_48h__max', 'static_age_years', 'gcs__0_48h__last', 'urine__0_48h__total',
              'lactate__0_48h__max', 'platelets__0_48h__min', 'creatinine__0_48h__max',
              'lactate__0_48h__measured']
SWAPS = {'gcs__0_48h__min': 'gcs__0_48h__last', 'urine__0_48h__min': 'urine__0_48h__total'}

Xtr, ytr = get_split('train'); Xva, yva = get_split('validation'); Xte, yte = get_split('test')
print(f"train {len(ytr)} / val {len(yva)} / test {len(yte)}   deaths {ytr.sum()}/{yva.sum()}/{yte.sum()}")

sc = StandardScaler().fit(Xtr[L3_REVISED])
model = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
model.fit(sc.transform(Xtr[L3_REVISED]), ytr)

p_tr = model.predict_proba(sc.transform(Xtr[L3_REVISED]))[:, 1]
p_va = model.predict_proba(sc.transform(Xva[L3_REVISED]))[:, 1]
p_te = model.predict_proba(sc.transform(Xte[L3_REVISED]))[:, 1]

m_tr, m_va = compute_metrics(ytr, p_tr), compute_metrics(yva, p_va)
print(f"\ntrain      AUROC {m_tr['auroc']:.4f}  AUPRC {m_tr['auprc']:.4f}")
print(f"validation AUROC {m_va['auroc']:.4f}  AUPRC {m_va['auprc']:.4f}   (matches ablation variant E: 0.8081)")

thr = find_threshold_for_recall(yva, p_va, target_recall=0.80)
print(f"\nthreshold {thr:.4f} from validation, recall achieved {compute_metrics(yva, p_va, threshold=thr)['sensitivity']:.4f}")

print("\n--- scoring test (single, pre-declared look) ---")
m_te = compute_metrics(yte, p_te, threshold=thr)
a_pt, a_lo, a_hi = bootstrap_ci(yte, p_te, lambda y, s: compute_metrics(y, s)['auroc'], n_bootstrap=1000)
p_pt, p_lo, p_hi = bootstrap_ci(yte, p_te, lambda y, s: compute_metrics(y, s)['auprc'], n_bootstrap=1000)
mse = min_se_ppv(yte, (p_te >= thr).astype(int))
print(f"test AUROC {a_pt:.4f} [{a_lo:.4f}, {a_hi:.4f}]")
print(f"test AUPRC {p_pt:.4f} [{p_lo:.4f}, {p_hi:.4f}]")
print(f"Se {m_te['sensitivity']:.4f}  PPV {m_te['ppv']:.4f}  Sp {m_te['specificity']:.4f}  min(Se,PPV) {mse:.4f}")

L3_ORIG_AUROC, L3_ORIG_CI = 0.7417, [0.7075, 0.7740]
L4_AUROC, L4_CI = 0.8849, [0.8653, 0.9031]
def overlap(a, b): return not (a[0] > b[1] or a[1] < b[0])
rev_ci = [a_lo, a_hi]
print(f"\nvs L3 original 0.7417 {L3_ORIG_CI}: {a_pt - L3_ORIG_AUROC:+.4f}  CI overlap {overlap(rev_ci, L3_ORIG_CI)}")
print(f"vs L4 0.8849 {L4_CI}:          {a_pt - L4_AUROC:+.4f}  CI overlap {overlap(rev_ci, L4_CI)}")

json.dump({
    'model': 'L3_revised_logistic_8features',
    'declaration': ('Feature set fixed before first execution; variant E of '
                    'run_L3_gcs_statistic_ablation.py, selected on train+validation only. '
                    'Test scored once in this run. Original L3 rung is retained, not replaced.'),
    'features': L3_REVISED, 'features_original_l3': L3_ORIGINAL, 'statistic_swaps': SWAPS,
    'recipe': {'penalty': 'l2', 'C': 1.0, 'class_weight': 'balanced',
               'scaler': 'StandardScaler fit on train only', 'imputation': 'train-median (pre-imputed matrices)'},
    'coefficients': {f: float(c) for f, c in zip(L3_REVISED, model.coef_[0])},
    'intercept': float(model.intercept_[0]),
    'train': {'auroc': float(m_tr['auroc']), 'auprc': float(m_tr['auprc'])},
    'validation': {'auroc': float(m_va['auroc']), 'auprc': float(m_va['auprc'])},
    'threshold': {'value': float(thr), 'selection': 'validation recall 0.80',
                  'validation_recall_achieved': float(compute_metrics(yva, p_va, threshold=thr)['sensitivity'])},
    'test': {'n': int(len(yte)), 'prevalence': float(yte.mean()),
             'auroc': float(a_pt), 'auroc_ci': [float(a_lo), float(a_hi)],
             'auprc': float(p_pt), 'auprc_ci': [float(p_lo), float(p_hi)],
             'sensitivity': float(m_te['sensitivity']), 'ppv': float(m_te['ppv']),
             'specificity': float(m_te['specificity']), 'min_se_ppv': float(mse)},
    'comparisons': {
        'vs_l3_original': {'their_auroc': L3_ORIG_AUROC, 'their_ci': L3_ORIG_CI,
                           'delta': float(a_pt - L3_ORIG_AUROC), 'ci_overlap': overlap(rev_ci, L3_ORIG_CI)},
        'vs_l4_xgboost': {'their_auroc': L4_AUROC, 'their_ci': L4_CI,
                          'delta': float(a_pt - L4_AUROC), 'ci_overlap': overlap(rev_ci, L4_CI)}},
    'overfitting': {'train_val_auroc_gap': float(m_tr['auroc'] - m_va['auroc'])},
}, open(R / 'L3_revised_test_evaluation.json', 'w'), indent=2)
print("\nsaved -> results/L3_revised_test_evaluation.json")
