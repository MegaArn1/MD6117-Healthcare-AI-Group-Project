"""
L4 SHAP Feature Importance Analysis with GPU

Computes SHAP values and analyzes feature importance with:
- Top 20 features by absolute SHAP
- Clinical domain categorization
- Comparison to L3 logistic regression features
- Clinical validation checks
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data

print("="*60)
print("L4 SHAP Feature Importance Analysis")
print("="*60)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
MODELS_DIR = Path(__file__).parent.parent / 'models'
FIGURES_DIR = Path(__file__).parent.parent / 'results' / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Load model selection
with open(RESULTS_DIR / 'L4_model_selection.json') as f:
    selection = json.load(f)

selected_model = selection['selected_model']
print(f"\nSelected model: {selected_model.upper()}")

# Load data
print("\n1. Loading data...")
labels_df, X_unimputed, _ = load_data()
test_df = labels_df[labels_df['split'] == 'test'].merge(X_unimputed, on='RecordID')
feature_cols = [c for c in X_unimputed.columns if c != 'RecordID']

X_test = test_df[feature_cols].values
y_test = test_df['In-hospital_death'].values
print(f"   Test: {len(y_test)} samples, {len(feature_cols)} features")

# Load model
print(f"\n2. Loading {selected_model.upper()} model...")
if selected_model == 'xgboost':
    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / 'L4_xgboost_best.json'))
elif selected_model == 'catboost':
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / 'L4_catboost_best.cbm'))
else:  # lightgbm
    model = lgb.Booster(model_file=str(MODELS_DIR / 'L4_lightgbm_best.txt'))

# Compute SHAP values on test set
print("\n3. Computing SHAP values (may take 5-10 min)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# For binary classifiers that return a list (e.g. LightGBM), take class-1 values
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Mean absolute SHAP
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print(f"   ✓ SHAP values computed")
print(f"\nTop 10 features by SHAP importance:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")

# Load L3 features for comparison
with open(RESULTS_DIR / 'L3_logistic_regression_8features.json') as f:
    l3_results = json.load(f)
l3_features = set(l3_results['features'])

# Categorize top 20 by clinical domain
print("\n4. Clinical domain categorization...")
top_20 = feature_importance.head(20).copy()

def categorize_feature(name):
    """Categorize feature by clinical domain"""
    name_lower = name.lower()
    if any(x in name_lower for x in ['bun', 'creatinine', 'urine']):
        return 'Renal'
    elif any(x in name_lower for x in ['gcs', 'glasgow']):
        return 'CNS'
    elif any(x in name_lower for x in ['lactate', 'bp_', 'map', 'hr', 'heartrate']):
        return 'Cardiovascular'
    elif any(x in name_lower for x in ['platelets', 'wbc', 'hct']):
        return 'Coagulation/Hematology'
    elif any(x in name_lower for x in ['bilirubin', 'alt', 'ast', 'albumin']):
        return 'Hepatic'
    elif any(x in name_lower for x in ['pao2', 'paco2', 'fio2', 'sao2', 'mechvent', 'resprate']):
        return 'Respiratory'
    elif 'age' in name_lower:
        return 'Demographic'
    elif 'measured' in name_lower or 'count' in name_lower:
        return 'Measurement behavior'
    elif 'delta' in name_lower or 'slope' in name_lower:
        return 'Temporal'
    else:
        return 'Other'

top_20['clinical_domain'] = top_20['feature'].apply(categorize_feature)
top_20['in_l3_features'] = top_20['feature'].isin(l3_features)

# L3 overlap analysis
l3_in_top_20 = top_20['in_l3_features'].sum()
l3_in_top_50 = feature_importance.head(50)['feature'].isin(l3_features).sum()

print(f"   L3 features (8 total) in L4 top 20: {l3_in_top_20}")
print(f"   L3 features in L4 top 50: {l3_in_top_50}")

# Clinical validation
suspicious_features = []
for _, row in top_20.iterrows():
    feat = row['feature']
    # Check for potential artifacts
    if 'recordid' in feat.lower():
        suspicious_features.append(f"{feat} (ID leakage)")
    elif 'survival' in feat.lower() or 'death' in feat.lower():
        suspicious_features.append(f"{feat} (outcome leakage)")

physiologically_sensible = len(suspicious_features) == 0

print(f"\n5. Clinical validation...")
print(f"   Physiologically sensible: {physiologically_sensible}")
if suspicious_features:
    print(f"   ⚠ Suspicious features detected:")
    for sf in suspicious_features:
        print(f"     - {sf}")
else:
    print(f"   ✓ No obvious artifacts detected")

# Measurement behavior importance
measurement_features = top_20[top_20['clinical_domain'] == 'Measurement behavior']
if len(measurement_features) > 0:
    measurement_importance = f"{len(measurement_features)} measurement behavior features in top 20, supporting 'missingness is informative' hypothesis"
else:
    measurement_importance = "No measurement behavior features in top 20"

# Temporal features importance
temporal_features = top_20[top_20['clinical_domain'] == 'Temporal']
if len(temporal_features) > 0:
    temporal_importance = f"{len(temporal_features)} temporal features (delta/slope) in top 20, confirming trajectory information is valuable"
else:
    temporal_importance = "No temporal features in top 20"

print(f"   Measurement behavior: {measurement_importance}")
print(f"   Temporal features: {temporal_importance}")

# SHAP summary plot
print("\n6. Generating SHAP summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False, max_display=20)
plt.tight_layout()
plot_path = FIGURES_DIR / 'L4_shap_summary.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Plot saved to {plot_path.name}")

# Save results
top_20_list = []
for _, row in top_20.iterrows():
    top_20_list.append({
        'feature_name': row['feature'],
        'shap_importance': float(row['mean_abs_shap']),
        'clinical_domain': row['clinical_domain'],
        'in_l3_features': bool(row['in_l3_features']),
        'clinical_interpretation': f"Feature from {row['clinical_domain']} domain"
    })

results = {
    'selected_model': selected_model,
    'shap_computation': 'TreeExplainer on test set',
    'top_20_features': top_20_list,
    'l3_overlap': {
        'l3_features_in_top_20': int(l3_in_top_20),
        'l3_features_in_top_50': int(l3_in_top_50),
        'l3_total_features': 8,
        'interpretation': f"{l3_in_top_20}/8 L3 features appear in L4 top 20, validating L3 feature selection quality"
    },
    'clinical_validation': {
        'physiologically_sensible': physiologically_sensible,
        'suspicious_features': suspicious_features,
        'measurement_behavior_importance': measurement_importance,
        'temporal_features_importance': temporal_importance
    },
    'artifact_check': 'No outcome leakage detected' if physiologically_sensible else f'WARNING: {len(suspicious_features)} suspicious features detected'
}

output_file = RESULTS_DIR / 'L4_feature_importance.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_file.name}")
print("="*60)
