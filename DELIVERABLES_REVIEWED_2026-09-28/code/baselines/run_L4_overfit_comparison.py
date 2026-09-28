"""
L4 capacity check across BOTH GBDT candidates. Train and validation only.
Test is deliberately NOT touched: CatBoost lost model selection and must
never be scored on test, and XGBoost's test number is already frozen in
L4_test_evaluation.json.

Answers one question: is train AUROC ≈ 1.0 an XGBoost-specific artefact, or a
property of the data shape (1,474 features vs 8,283 rows)?
Also records how many trees each final model actually used, because XGBoost's
final fit had no early stopping while CatBoost's did.
"""
import sys, json
import numpy as np
from pathlib import Path
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data
from metrics import compute_metrics

R = Path(__file__).parent.parent / 'results'
M = Path(__file__).parent.parent / 'models'

labels, X, _ = load_data()
df = labels.merge(X, on='RecordID', validate='one_to_one')
feat = [c for c in X.columns if c != 'RecordID']
tr = df[df['split'] == 'train'];  va = df[df['split'] == 'validation']
Xtr, ytr = tr[feat].values, tr['In-hospital_death'].values
Xva, yva = va[feat].values, va['In-hospital_death'].values

out = {}

xgbm = xgb.XGBClassifier(); xgbm.load_model(str(M / 'L4_xgboost_best.json'))
n_xgb = xgbm.get_booster().num_boosted_rounds()
p_tr = xgbm.predict_proba(Xtr)[:, 1]; p_va = xgbm.predict_proba(Xva)[:, 1]
out['xgboost'] = {
    'trees_used': int(n_xgb), 'trees_configured': 300, 'early_stopping_in_final_fit': False,
    'train': {k: float(v) for k, v in compute_metrics(ytr, p_tr).items()},
    'validation': {k: float(v) for k, v in compute_metrics(yva, p_va).items()},
    'train_min_score_among_deaths': float(p_tr[ytr == 1].min()),
    'train_max_score_among_survivors': float(p_tr[ytr == 0].max()),
}

cbm = CatBoostClassifier(); cbm.load_model(str(M / 'L4_catboost_best.cbm'))
n_cb = cbm.tree_count_
p_tr = cbm.predict_proba(Xtr)[:, 1]; p_va = cbm.predict_proba(Xva)[:, 1]
out['catboost'] = {
    'trees_used': int(n_cb), 'trees_configured': 2000, 'early_stopping_in_final_fit': True,
    'early_stopping_rounds': 50,
    'train': {k: float(v) for k, v in compute_metrics(ytr, p_tr).items()},
    'validation': {k: float(v) for k, v in compute_metrics(yva, p_va).items()},
    'train_min_score_among_deaths': float(p_tr[ytr == 1].min()),
    'train_max_score_among_survivors': float(p_tr[ytr == 0].max()),
}

for k, v in out.items():
    v['train_val_auroc_gap'] = v['train']['auroc'] - v['validation']['auroc']
    v['train_perfectly_separated'] = bool(v['train_min_score_among_deaths'] > v['train_max_score_among_survivors'])

print(f"{'model':10s} {'trees':>6s}  {'train AUROC':>11s} {'val AUROC':>9s} {'gap':>7s}  {'train AUPRC':>11s} {'val AUPRC':>9s}  perfect-sep")
for k, v in out.items():
    print(f"{k:10s} {v['trees_used']:6d}  {v['train']['auroc']:11.4f} {v['validation']['auroc']:9.4f} "
          f"{v['train_val_auroc_gap']:+7.4f}  {v['train']['auprc']:11.4f} {v['validation']['auprc']:9.4f}  "
          f"{v['train_perfectly_separated']}")

out['_note'] = ('Train/validation only. Test not scored for either model here. '
                'perfect-sep = every training death scored above every training survivor.')
json.dump(out, open(R / 'L4_overfit_comparison.json', 'w'), indent=2)
print("\nsaved -> results/L4_overfit_comparison.json")
