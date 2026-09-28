"""
L4 Individual Case Interpretability: SHAP waterfall for 3 representative patients.

Pick from the frozen test set:
  1. True positive (high risk, died): high predicted probability, outcome = 1
  2. False positive: high predicted probability, outcome = 0
  3. False negative: low predicted probability, outcome = 1

Per 实验方案 §8, individual explanations are a hard course requirement. This
completes the "translate technical output to clinical language" deliverable.
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data

R = Path(__file__).parent.parent / 'results'
M = Path(__file__).parent.parent / 'models'
FIG = R / 'figures'; FIG.mkdir(exist_ok=True)

print("="*70)
print("L4 Individual Case Interpretability (SHAP Waterfall)")
print("="*70)

labels, X_unimputed, _ = load_data()
test_df = labels[labels['split'] == 'test'].merge(X_unimputed, on='RecordID', validate='one_to_one')
feat = [c for c in X_unimputed.columns if c != 'RecordID']
X_test = test_df[feat].values
y_test = test_df['In-hospital_death'].values
record_ids = test_df['RecordID'].values

model = xgb.XGBClassifier()
model.load_model(str(M / 'L4_xgboost_best.json'))
p_test = model.predict_proba(X_test)[:, 1]

# SHAP explainer (TreeExplainer for XGBoost)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("\n1. Selecting representative cases")
# TP: high prob, died
tp_idx = np.where((p_test > 0.7) & (y_test == 1))[0]
if len(tp_idx) == 0:
    tp_idx = np.where((p_test > 0.5) & (y_test == 1))[0]
tp_i = tp_idx[np.argmax(p_test[tp_idx])]

# FP: high prob, survived
fp_idx = np.where((p_test > 0.5) & (y_test == 0))[0]
fp_i = fp_idx[np.argmax(p_test[fp_idx])] if len(fp_idx) > 0 else None

# FN: low prob, died
fn_idx = np.where((p_test < 0.3) & (y_test == 1))[0]
if len(fn_idx) == 0:
    fn_idx = np.where((p_test < 0.5) & (y_test == 1))[0]
fn_i = fn_idx[np.argmin(p_test[fn_idx])] if len(fn_idx) > 0 else None

cases = []
for label, idx in [('TP_high_risk_died', tp_i), ('FP_high_risk_survived', fp_i), ('FN_low_risk_died', fn_i)]:
    if idx is None:
        print(f"   {label}: not found")
        continue
    rec_id = int(record_ids[idx])
    pred_prob = float(p_test[idx])
    outcome = int(y_test[idx])
    print(f"   {label:25s}: RecordID {rec_id:5d}, pred={pred_prob:.3f}, outcome={outcome}")
    
    # SHAP waterfall plot
    shap_exp = shap.Explanation(values=shap_values[idx], 
                                base_values=explainer.expected_value,
                                data=X_test[idx],
                                feature_names=feat)
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_exp, max_display=15, show=False)
    plt.title(f"{label.replace('_', ' ').title()}\nRecordID {rec_id}, Pred={pred_prob:.3f}, Outcome={outcome}", 
              fontsize=11, pad=10)
    plt.tight_layout()
    fig_path = FIG / f'L4_shap_waterfall_{label}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved -> {fig_path.relative_to(Path.cwd())}")
    
    # Top 10 SHAP contributors for JSON
    top10_idx = np.argsort(-np.abs(shap_values[idx]))[:10]
    contributors = [{'feature': feat[i], 'value': float(X_test[idx, i]), 
                     'shap_value': float(shap_values[idx][i])} for i in top10_idx]
    
    cases.append({
        'label': label,
        'record_id': rec_id,
        'predicted_probability': pred_prob,
        'actual_outcome': outcome,
        'top_10_shap_contributors': contributors,
    })

output = {
    'model': 'L4_xgboost',
    'dataset': 'test',
    'cases': cases,
    '_note': ('Individual case explanations using SHAP waterfall. Cases selected from frozen '
              'test predictions: one true positive (high risk, died), one false positive '
              '(high risk, survived), one false negative (low risk, died).')
}
json.dump(output, open(R / 'L4_case_examples.json', 'w'), indent=2)
print(f"\nSaved -> {R.relative_to(Path.cwd())}/L4_case_examples.json")
print("="*70)
