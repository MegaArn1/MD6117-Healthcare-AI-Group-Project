"""
Early-stopping ablation for the selected XGBoost configuration.
Train + validation only. Test is not scored.

The GPU run's final fit passed eval_set but no early_stopping_rounds, so it
ran the full 300 trees and interpolated the training set (train AUROC 1.0).
Question: with identical hyperparameters, does early stopping on validation
change (a) trees used, (b) train AUROC, (c) validation AUROC?

Both arms are refit here on CPU (tree_method='hist') so the comparison is
like-for-like; CPU and GPU histograms differ marginally, so the no-ES arm
will not reproduce the saved model bit-for-bit.
"""
import sys, json, time
import numpy as np
from pathlib import Path
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data
from metrics import compute_metrics

R = Path(__file__).parent.parent / 'results'
labels, X, _ = load_data()
df = labels.merge(X, on='RecordID', validate='one_to_one')
feat = [c for c in X.columns if c != 'RecordID']
tr = df[df['split'] == 'train']; va = df[df['split'] == 'validation']
Xtr, ytr = tr[feat].values, tr['In-hospital_death'].values
Xva, yva = va[feat].values, va['In-hospital_death'].values

best = json.load(open(R / 'L4_xgboost_training.json'))['best_hyperparameters']
spw = (len(ytr) - ytr.sum()) / ytr.sum()
base = dict(objective='binary:logistic', eval_metric='auc', scale_pos_weight=spw,
            random_state=42, n_jobs=-1, device='cpu', tree_method='hist')
base.update({k: v for k, v in best.items() if k != 'n_estimators'})

def run(name, **kw):
    t = time.time()
    m = xgb.XGBClassifier(**base, **kw)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    p_tr = m.predict_proba(Xtr)[:, 1]; p_va = m.predict_proba(Xva)[:, 1]
    mt, mv = compute_metrics(ytr, p_tr), compute_metrics(yva, p_va)
    res = dict(trees_used=int(m.get_booster().num_boosted_rounds()),
               best_iteration=int(getattr(m, 'best_iteration', -1)),
               train_auroc=float(mt['auroc']), train_auprc=float(mt['auprc']),
               val_auroc=float(mv['auroc']), val_auprc=float(mv['auprc']),
               train_perfectly_separated=bool(p_tr[ytr == 1].min() > p_tr[ytr == 0].max()),
               fit_seconds=round(time.time() - t, 1))
    res['train_val_auroc_gap'] = res['train_auroc'] - res['val_auroc']
    print(f"{name:28s} trees={res['trees_used']:4d} best_it={res['best_iteration']:4d} "
          f"train={res['train_auroc']:.4f} val={res['val_auroc']:.4f} gap={res['train_val_auroc_gap']:+.4f} "
          f"perfect-sep={res['train_perfectly_separated']} ({res['fit_seconds']}s)")
    return res

out = {
    'no_early_stopping_300': run('no ES, n_estimators=300', n_estimators=300),
    'early_stopping_50_cap2000': run('ES=50, n_estimators<=2000', n_estimators=2000, early_stopping_rounds=50),
    '_note': 'Train/validation only; CPU hist refit with the GPU-selected hyperparameters.',
    'hyperparameters': base,
}
json.dump(out, open(R / 'L4_xgboost_es_ablation.json', 'w'), indent=2, default=str)
print("\nsaved -> results/L4_xgboost_es_ablation.json")
