"""
How many of the 1,474 features does L4 actually need? Train + validation only.

Protocol (no validation leakage into the selection step):
  1. SHAP on the TRAIN set for the selected XGBoost model -> feature ranking.
  2. For each k, refit XGBoost with the same hyperparameters on the top-k
     train-ranked features, early stopping on validation.
  3. Report validation AUROC/AUPRC. Test is not scored.
"""
import sys, json, time
import numpy as np
from pathlib import Path
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data
from metrics import compute_metrics

R = Path(__file__).parent.parent / 'results'; M = Path(__file__).parent.parent / 'models'
labels, X, _ = load_data()
df = labels.merge(X, on='RecordID', validate='one_to_one')
feat = [c for c in X.columns if c != 'RecordID']
tr = df[df['split'] == 'train']; va = df[df['split'] == 'validation']
Xtr, ytr = tr[feat].values, tr['In-hospital_death'].values
Xva, yva = va[feat].values, va['In-hospital_death'].values

xm = xgb.XGBClassifier(); xm.load_model(str(M / 'L4_xgboost_best.json'))
s_tr = xm.get_booster().predict(xgb.DMatrix(Xtr, feature_names=feat), pred_contribs=True)[:, :-1]
order = np.argsort(-np.abs(s_tr).mean(0)); ranked = [feat[i] for i in order]

best = json.load(open(R / 'L4_xgboost_training.json'))['best_hyperparameters']
spw = (len(ytr) - ytr.sum()) / ytr.sum()
base = dict(objective='binary:logistic', eval_metric='auc', scale_pos_weight=spw, random_state=42,
            n_jobs=-1, device='cpu', tree_method='hist', n_estimators=2000, early_stopping_rounds=50)
base.update({k: v for k, v in best.items() if k != 'n_estimators'})

out = {'ranking_source': 'mean |SHAP| on TRAIN set, selected XGBoost model', 'results': []}
print(f"{'k':>5s} {'trees':>5s} {'train AUROC':>11s} {'val AUROC':>9s} {'val AUPRC':>9s}  {'Δval vs all':>11s}")
full_val = None
for k in [8, 20, 50, 100, 200, 500, 1474]:
    idx = order[:k]; t = time.time()
    m = xgb.XGBClassifier(**base).fit(Xtr[:, idx], ytr, eval_set=[(Xva[:, idx], yva)], verbose=False)
    mt = compute_metrics(ytr, m.predict_proba(Xtr[:, idx])[:, 1])
    mv = compute_metrics(yva, m.predict_proba(Xva[:, idx])[:, 1])
    row = dict(k=k, trees_used=int(m.get_booster().num_boosted_rounds()), train_auroc=float(mt['auroc']),
               val_auroc=float(mv['auroc']), val_auprc=float(mv['auprc']), fit_seconds=round(time.time() - t, 1))
    if k == 1474: full_val = row['val_auroc']
    out['results'].append(row)
for row in out['results']:
    d = row['val_auroc'] - full_val
    print(f"{row['k']:5d} {row['trees_used']:5d} {row['train_auroc']:11.4f} {row['val_auroc']:9.4f} {row['val_auprc']:9.4f}  {d:+11.4f}")
out['top_50_train_ranked'] = ranked[:50]
out['_note'] = 'Validation metrics only. Ranking from train-set SHAP so validation is not used for selection.'
json.dump(out, open(R / 'L4_feature_count_ablation.json', 'w'), indent=2)
print("\nsaved -> results/L4_feature_count_ablation.json")
