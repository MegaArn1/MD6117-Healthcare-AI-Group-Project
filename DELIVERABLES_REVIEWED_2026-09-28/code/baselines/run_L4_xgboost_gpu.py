"""
L4 XGBoost Training with GPU Acceleration

Trains XGBoost on full 1,474-feature unimputed dataset with:
- GPU acceleration (device='cuda', XGBoost 2.0+ API)
- Native NaN handling
- scale_pos_weight for class imbalance
- Hyperparameter search via RandomizedSearchCV
- Early stopping on validation set
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data
from metrics import compute_metrics

print("="*60)
print("L4 XGBoost Training (GPU)")
print("="*60)

# Paths
RESULTS_DIR = Path(__file__).parent.parent / 'results'
MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

start_time = time.time()

# Load unimputed data for native NaN handling
print("\n1. Loading unimputed data...")
labels_df, X_unimputed, _ = load_data()

# Filter by split
train_df = labels_df[labels_df['split'] == 'train'].merge(X_unimputed, on='RecordID')
val_df = labels_df[labels_df['split'] == 'validation'].merge(X_unimputed, on='RecordID')
test_df = labels_df[labels_df['split'] == 'test'].merge(X_unimputed, on='RecordID')

# Extract features and labels
feature_cols = [c for c in X_unimputed.columns if c != 'RecordID']
print(f"   Features: {len(feature_cols)}")

X_train = train_df[feature_cols].values
y_train = train_df['In-hospital_death'].values
X_val = val_df[feature_cols].values
y_val = val_df['In-hospital_death'].values
X_test = test_df[feature_cols].values
y_test = test_df['In-hospital_death'].values

print(f"   Train: {len(y_train)} samples, {y_train.sum()} deaths")
print(f"   Val:   {len(y_val)} samples, {y_val.sum()} deaths")
print(f"   Test:  {len(y_test)} samples, {y_test.sum()} deaths")

# Calculate scale_pos_weight
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

# Check GPU availability
print("\n2. Checking GPU availability...")
try:
    # XGBoost 2.0+ uses device='cuda' for GPU
    test_params = {'device': 'cuda'}
    test_model = xgb.XGBClassifier(**test_params, n_estimators=1)
    test_model.fit(X_train[:100], y_train[:100])
    print("   ✓ GPU available and working")
    use_gpu = True
except Exception as e:
    print(f"   ✗ GPU not available, falling back to CPU: {e}")
    use_gpu = False

# Hyperparameter search space
param_distributions = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300, 500],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'max_bin': [63, 127, 255]  # Reduce for GPU memory
}

print("\n3. Hyperparameter search...")
print(f"   Search space: {sum(len(v) for v in param_distributions.values())} combinations")
print(f"   Using: {'GPU' if use_gpu else 'CPU'}")

# Base model
base_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'n_jobs': -1
}

if use_gpu:
    base_params['device'] = 'cuda'
else:
    base_params['device'] = 'cpu'
    base_params['tree_method'] = 'hist'

base_model = xgb.XGBClassifier(**base_params)

# RandomizedSearchCV with 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    base_model,
    param_distributions,
    n_iter=50,  # Try 50 random combinations
    scoring='roc_auc',
    cv=cv,
    verbose=2,
    n_jobs=1 if use_gpu else -1,  # GPU doesn't parallelize across folds well
    random_state=42
)

search.fit(X_train, y_train)

print(f"\n   Best CV AUROC: {search.best_score_:.4f}")
print(f"   Best params: {search.best_params_}")

# Train final model with early stopping on validation
print("\n4. Training final model with best params...")
best_model = xgb.XGBClassifier(**base_params, **search.best_params_)
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Validation performance
y_val_proba = best_model.predict_proba(X_val)[:, 1]
val_metrics = compute_metrics(y_val, y_val_proba)
print(f"   Val AUROC: {val_metrics['auroc']:.4f}")
print(f"   Val AUPRC: {val_metrics['auprc']:.4f}")

# Save model
model_path = MODELS_DIR / 'L4_xgboost_best.json'
best_model.save_model(str(model_path))
print(f"\n5. Model saved to {model_path.name}")

elapsed = time.time() - start_time
print(f"\nTotal training time: {elapsed/60:.1f} minutes")

# Save results
results = {
    'model': 'xgboost',
    'gpu_used': use_gpu,
    'n_features': len(feature_cols),
    'best_hyperparameters': search.best_params_,
    'cv_best_auroc': float(search.best_score_),
    'validation_auroc': float(val_metrics['auroc']),
    'validation_auprc': float(val_metrics['auprc']),
    'training_time_seconds': float(elapsed),
    'scale_pos_weight': float(scale_pos_weight),
    'train_samples': int(len(y_train)),
    'val_samples': int(len(y_val)),
    'test_samples': int(len(y_test))
}

output_file = RESULTS_DIR / 'L4_xgboost_training.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_file.name}")
print("="*60)
