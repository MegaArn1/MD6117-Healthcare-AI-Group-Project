"""
L4 subgroup AUROC by all four ICU types, on the frozen test prediction vector.

The earlier run used only static_icutype_2 (a single one-hot column) and so
reported one subgroup. The matrix has static_icutype_1..4; this declares all
four up front and scores them together. No retraining, no threshold choice.
PhysioNet 2012 ICUType codes: 1 Coronary, 2 Cardiac Surgery Recovery,
3 Medical, 4 Surgical.
"""
import sys, json
import numpy as np
from pathlib import Path
import xgboost as xgb
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data
from metrics import compute_metrics, bootstrap_ci

R = Path(__file__).parent.parent / 'results'; M = Path(__file__).parent.parent / 'models'
labels, X, _ = load_data()
te = labels[labels['split'] == 'test'].merge(X, on='RecordID', validate='one_to_one')
feat = [c for c in X.columns if c != 'RecordID']
y = te['In-hospital_death'].values
m = xgb.XGBClassifier(); m.load_model(str(M / 'L4_xgboost_best.json'))
p = m.predict_proba(te[feat].values)[:, 1]

names = {1: 'Coronary', 2: 'Cardiac surgery recovery', 3: 'Medical', 4: 'Surgical'}
rows = []
print(f"{'ICUType':28s} {'n':>5s} {'deaths':>6s} {'prev':>6s} {'AUROC':>7s}  95% CI")
for k, nm in names.items():
    mask = te[f'static_icutype_{k}'].values == 1
    ys, ps = y[mask], p[mask]
    pt, lo, hi = bootstrap_ci(ys, ps, lambda a, b: compute_metrics(a, b)['auroc'], n_bootstrap=1000)
    apt, alo, ahi = bootstrap_ci(ys, ps, lambda a, b: compute_metrics(a, b)['auprc'], n_bootstrap=1000)
    rows.append(dict(code=k, icutype=nm, n=int(mask.sum()), deaths=int(ys.sum()), prevalence=float(ys.mean()),
                     auroc=float(pt), auroc_ci=[float(lo), float(hi)], auprc=float(apt), auprc_ci=[float(alo), float(ahi)]))
    print(f"{nm:28s} {mask.sum():5d} {int(ys.sum()):6d} {ys.mean():6.3f} {pt:7.4f}  [{lo:.4f}, {hi:.4f}]")
print(f"{'sum':28s} {sum(r['n'] for r in rows):5d} {sum(r['deaths'] for r in rows):6d}")
json.dump({'dataset': 'test', 'model': 'L4 XGBoost (frozen predictions)', 'overall_auroc': 0.8849,
           'subgroups': rows,
           '_note': 'All four ICUType one-hot columns declared and scored in one pass on the frozen test vector. '
                    'Supersedes the single-column subgroup in L4_calibration_and_subgroup.json.'},
          open(R / 'L4_subgroup_by_icutype.json', 'w'), indent=2)
print("saved -> results/L4_subgroup_by_icutype.json")
