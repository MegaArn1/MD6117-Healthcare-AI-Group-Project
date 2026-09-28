"""
L4 Supplementary Evaluation — fills gaps left by the GPU training run.

Three things the main L4 pipeline did not do:
1. Evaluate L4 on the SAPS-I / SOFA score-available subsets, which the stage-1
   sentinel decision requires for a fair head-to-head against L2.
2. Report train/validation/test together, so overfitting can be judged.
3. Re-derive SHAP feature categories on two orthogonal axes (organ domain and
   statistic type). The original categoriser tested organ keywords before
   statistic keywords, so every slope/measured/time_span feature was absorbed
   into an organ bucket and the "no temporal features" claim came out false.
"""
import sys, json
import numpy as np
from pathlib import Path
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data
from metrics import compute_metrics, bootstrap_ci

R = Path(__file__).parent.parent / 'results'
M = Path(__file__).parent.parent / 'models'

labels, X, _ = load_data()
df = labels.merge(X, on='RecordID', validate='one_to_one')
feat = [c for c in X.columns if c != 'RecordID']

model = xgb.XGBClassifier()
model.load_model(str(M / 'L4_xgboost_best.json'))

def pred(split):
    d = df[df['split'] == split]
    return d, model.predict_proba(d[feat].values)[:, 1], d['In-hospital_death'].values

print("=" * 66)
print("1. PERFORMANCE ACROSS SPLITS (overfitting check)")
print("=" * 66)
perf = {}
for s in ['train', 'validation', 'test']:
    _, p, y = pred(s)
    m = compute_metrics(y, p)
    perf[s] = {'auroc': float(m['auroc']), 'auprc': float(m['auprc'])}
    print(f"  {s:11s} AUROC {m['auroc']:.4f}   AUPRC {m['auprc']:.4f}")
gap_tr_val = perf['train']['auroc'] - perf['validation']['auroc']
gap_val_te = perf['test']['auroc'] - perf['validation']['auroc']
print(f"\n  train-val AUROC gap  = {gap_tr_val:+.4f}")
print(f"  test-val AUROC gap   = {gap_val_te:+.4f}")

print("\n" + "=" * 66)
print("2. FAIR HEAD-TO-HEAD ON SCORE-AVAILABLE SUBSETS (stage-1 protocol)")
print("=" * 66)
te, p_te, y_te = pred('test')
l2 = json.load(open(R / 'L2_clinical_scores_analysis.json'))
subsets = {}
for score, key in [('SAPS-I', 'saps_i'), ('SOFA', 'sofa')]:
    mask = (te[f'{key}_available'] == 1).values
    yb, pb = y_te[mask], p_te[mask]
    pt, lo, hi = bootstrap_ci(yb, pb, lambda a, b: compute_metrics(a, b)['auroc'], n_bootstrap=1000)
    ptp, lop, hip = bootstrap_ci(yb, pb, lambda a, b: compute_metrics(a, b)['auprc'], n_bootstrap=1000)
    base = l2[f'{key}_results']['available_only_metrics']
    # baseline CI on the same subset
    raw = te[key].values[mask]
    bpt, blo, bhi = bootstrap_ci(yb, raw, lambda a, b: compute_metrics(a, b)['auroc'], n_bootstrap=1000)
    overlap = not (lo > bhi or hi < blo)
    subsets[key] = {
        'score': score, 'n': int(mask.sum()),
        'baseline_auroc': float(bpt), 'baseline_auroc_ci': [float(blo), float(bhi)],
        'baseline_auprc': float(base['auprc']),
        'l4_auroc': float(pt), 'l4_auroc_ci': [float(lo), float(hi)],
        'l4_auprc': float(ptp), 'l4_auprc_ci': [float(lop), float(hip)],
        'improvement_auroc': float(pt - bpt),
        'ci_overlap': bool(overlap), 'statistically_significant': not bool(overlap),
    }
    print(f"\n  vs {score} (n={mask.sum()}):")
    print(f"    {score:7s} AUROC {bpt:.4f} [{blo:.4f}, {bhi:.4f}]   AUPRC {base['auprc']:.4f}")
    print(f"    L4      AUROC {pt:.4f} [{lo:.4f}, {hi:.4f}]   AUPRC {ptp:.4f}")
    print(f"    gain    AUROC {pt - bpt:+.4f}   CI overlap: {'YES' if overlap else 'NO'}")

print("\n" + "=" * 66)
print("3. SHAP RE-CATEGORISATION (statistic axis vs organ axis)")
print("=" * 66)
fi = json.load(open(R / 'L4_feature_importance.json'))
top = fi['top_20_features']

def stat_type(n):
    for k, lab in [('measured', 'measurement-behaviour'), ('__count', 'measurement-behaviour'),
                   ('slope_per_hour', 'temporal-trend'), ('__delta', 'temporal-trend'),
                   ('time_span_hours', 'measurement-behaviour'), ('__std', 'variability'),
                   ('__first', 'point-in-time'), ('__last', 'point-in-time'),
                   ('__min', 'extremum'), ('__max', 'extremum'),
                   ('__mean', 'central'), ('__median', 'central'), ('__total', 'cumulative')]:
        if k in n: return lab
    return 'static'

def organ(n):
    for keys, lab in [(['bun', 'creatinine', 'urine'], 'Renal'), (['gcs'], 'CNS'),
                      (['lactate', 'bp_', 'sysabp', 'diasabp', 'map', 'hr__'], 'Cardiovascular'),
                      (['platelets', 'wbc', 'hct'], 'Haematology'),
                      (['bilirubin', 'alt', 'ast', 'alp', 'albumin'], 'Hepatic'),
                      (['pao2', 'paco2', 'fio2', 'sao2', 'mechvent', 'resprate'], 'Respiratory'),
                      (['age', 'gender', 'icutype', 'bmi', 'height', 'weight'], 'Demographic/Context'),
                      (['record__'], 'Record-level')]:
        if any(k in n.lower() for k in keys): return lab
    return 'Other-lab'

from collections import Counter
st = Counter(); og = Counter()
print(f"\n  {'feature':42s} {'organ':22s} {'statistic':22s}")
for f in top:
    n = f['feature_name']; s = stat_type(n); o = organ(n)
    st[s] += 1; og[o] += 1
    print(f"  {n:42s} {o:22s} {s:22s}")
print(f"\n  statistic-type counts: {dict(st)}")
print(f"  organ-domain counts:   {dict(og)}")

l3_feats = set(json.load(open(R / 'L3_logistic_regression_8features.json'))['features'])
names20 = [f['feature_name'] for f in top]
l3_in20 = sum(n in l3_feats for n in names20)
l3_params = {n.split('__')[0] for n in l3_feats if '__' in n} | {'static_age_years'}
param_in20 = sum(1 for n in names20 if n.split('__')[0] in l3_params or n in l3_params)
print(f"\n  exact L3 features in top 20:            {l3_in20}/8")
print(f"  top-20 features whose PARAMETER L3 used: {param_in20}/20")

json.dump({
    'reason': 'Fills gaps in the L4 GPU run: subset head-to-head, overfitting check, SHAP re-categorisation',
    'performance_by_split': perf,
    'overfitting': {'train_val_auroc_gap': float(gap_tr_val), 'test_val_auroc_gap': float(gap_val_te)},
    'fair_subset_comparison': subsets,
    'shap_recategorised': {
        'statistic_type_counts': dict(st), 'organ_domain_counts': dict(og),
        'exact_l3_features_in_top20': int(l3_in20),
        'top20_sharing_l3_parameter': int(param_in20),
    },
}, open(R / 'L4_supplementary_evaluation.json', 'w'), indent=2)
print(f"\n  saved -> L4_supplementary_evaluation.json")
print("=" * 66)
