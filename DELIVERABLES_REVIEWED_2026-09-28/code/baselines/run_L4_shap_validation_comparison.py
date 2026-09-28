"""
Do XGBoost and CatBoost agree on what matters? SHAP on the VALIDATION set for
both (CatBoost was not selected, so it must not be scored on test; XGBoost's
test-set SHAP already exists in L4_feature_importance.json and is left alone).
Native SHAP: XGBoost pred_contribs, CatBoost type='ShapValues'.
"""
import sys, json
import numpy as np
from pathlib import Path
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data

R = Path(__file__).parent.parent / 'results'; M = Path(__file__).parent.parent / 'models'
labels, X, _ = load_data()
df = labels.merge(X, on='RecordID', validate='one_to_one')
feat = [c for c in X.columns if c != 'RecordID']
va = df[df['split'] == 'validation']; Xva = va[feat].values

xm = xgb.XGBClassifier(); xm.load_model(str(M / 'L4_xgboost_best.json'))
sx = xm.get_booster().predict(xgb.DMatrix(Xva, feature_names=feat), pred_contribs=True)[:, :-1]
cm = CatBoostClassifier(); cm.load_model(str(M / 'L4_catboost_best.cbm'))
sc = cm.get_feature_importance(Pool(Xva), type='ShapValues')[:, :-1]

def rank(s):
    imp = np.abs(s).mean(0); order = np.argsort(-imp)
    return [(feat[i], float(imp[i])) for i in order], {feat[i]: r + 1 for r, i in enumerate(order)}
xr, xrank = rank(sx); cr, crank = rank(sc)

K = 20
top_x = {f for f, _ in xr[:K]}; top_c = {f for f, _ in cr[:K]}
shared = sorted(top_x & top_c, key=lambda f: xrank[f])
print(f"{'rank':>4s}  {'XGBoost (val SHAP)':40s} {'|SHAP|':>7s}   {'CatBoost (val SHAP)':40s} {'|SHAP|':>7s}")
for i in range(15):
    print(f"{i+1:4d}  {xr[i][0]:40s} {xr[i][1]:7.3f}   {cr[i][0]:40s} {cr[i][1]:7.3f}")
print(f"\ntop-{K} overlap: {len(shared)}/{K}  ->  {shared}")
for f in ['gcs__0_48h__last', 'static_age_years', 'urine__0_48h__total', 'gcs__0_48h__min']:
    print(f"  {f:28s} XGB rank {xrank[f]:5d}   CatBoost rank {crank[f]:5d}")
r1x, r2x = xr[0][1], xr[1][1]; r1c, r2c = cr[0][1], cr[1][1]
print(f"\n#1/#2 importance ratio: XGBoost {r1x/r2x:.2f}x   CatBoost {r1c/r2c:.2f}x")

json.dump({
    'dataset': 'validation', 'top_k': K,
    'xgboost_top20': [{'feature': f, 'mean_abs_shap': v, 'rank': i + 1} for i, (f, v) in enumerate(xr[:K])],
    'catboost_top20': [{'feature': f, 'mean_abs_shap': v, 'rank': i + 1} for i, (f, v) in enumerate(cr[:K])],
    'top20_overlap_count': len(shared), 'top20_overlap_features': shared,
    'key_feature_ranks': {f: {'xgboost': xrank[f], 'catboost': crank[f]}
                          for f in ['gcs__0_48h__last', 'static_age_years', 'urine__0_48h__total', 'gcs__0_48h__min']},
    'rank1_to_rank2_ratio': {'xgboost': r1x / r2x, 'catboost': r1c / r2c},
}, open(R / 'L4_shap_validation_comparison.json', 'w'), indent=2)
print("saved -> results/L4_shap_validation_comparison.json")
