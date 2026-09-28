"""
Model-level effect of the GCS statistic choice in L3. Train + validation only.

Stage 2 chose gcs__0_48h__min on clinical grounds (SOFA CNS uses worst GCS).
Stage 3 SHAP ranked gcs__0_48h__last first. The single-variable gap between
them does not translate 1:1 into model performance, so this refits the same
8-feature logistic regression with the statistic swapped and reports
VALIDATION metrics. L3's frozen test number is left untouched.
"""
import sys, json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import get_split
from metrics import compute_metrics

R = Path(__file__).parent.parent / 'results'
l3 = json.load(open(R / 'L3_logistic_regression_8features.json'))
orig = l3['features']
Xtr, ytr = get_split('train'); Xva, yva = get_split('validation')

def fit_eval(feats):
    sc = StandardScaler().fit(Xtr[feats])
    m = LogisticRegression(penalty='l2', C=l3['regularization']['C'],
                           class_weight='balanced', max_iter=1000, random_state=42)
    m.fit(sc.transform(Xtr[feats]), ytr)
    mv = compute_metrics(yva, m.predict_proba(sc.transform(Xva[feats]))[:, 1])
    mt = compute_metrics(ytr, m.predict_proba(sc.transform(Xtr[feats]))[:, 1])
    return dict(n_features=len(feats), train_auroc=float(mt['auroc']),
                val_auroc=float(mv['auroc']), val_auprc=float(mv['auprc']))

variants = {
    'A_original_gcs_min':        orig,
    'B_swap_gcs_min_to_last':    [f if f != 'gcs__0_48h__min' else 'gcs__0_48h__last' for f in orig],
    'C_add_gcs_last_keep_min':   orig + ['gcs__0_48h__last'],
    'D_swap_urine_min_to_total': [f if f != 'urine__0_48h__min' else 'urine__0_48h__total' for f in orig],
    'E_both_swaps':              [{'gcs__0_48h__min': 'gcs__0_48h__last',
                                   'urine__0_48h__min': 'urine__0_48h__total'}.get(f, f) for f in orig],
}
out = {}
print(f"{'variant':28s} {'nfeat':>5s} {'train':>7s} {'val AUROC':>9s} {'val AUPRC':>9s}  Δval vs A")
for k, fs in variants.items():
    out[k] = fit_eval(fs)
    d = out[k]['val_auroc'] - out['A_original_gcs_min']['val_auroc']
    print(f"{k:28s} {out[k]['n_features']:5d} {out[k]['train_auroc']:7.4f} "
          f"{out[k]['val_auroc']:9.4f} {out[k]['val_auprc']:9.4f}  {d:+.4f}")
out['_note'] = 'Validation-set metrics only; L3 test number in L3_logistic_regression_8features.json is unchanged.'
json.dump(out, open(R / 'L3_gcs_statistic_ablation.json', 'w'), indent=2)
print("\nsaved -> results/L3_gcs_statistic_ablation.json")
