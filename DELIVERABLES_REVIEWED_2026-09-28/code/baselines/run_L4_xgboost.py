"""
Stage 4: L4 Baseline - XGBoost with Full Feature Set (1,474 features)

Native NaN handling, scale_pos_weight for class imbalance.
Hyperparameter search on train+validation combined.
Target: AUROC 0.83-0.86 (literature level).
"""

import sys
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from time import time
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
import xgboost as xgb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data
from metrics import compute_metrics, bootstrap_ci

def main():
    print("=" * 80)
    print("L4 BASELINE: XGBOOST (1,474 FEATURES, NATIVE NaN HANDLING)")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    labels_df, X_unimputed, _ = load_data()
    
    # Merge features with labels to get splits
    X_unimputed = X_unimputed.merge(labels_df[['RecordID', 'split', 'In-hospital_death']], on='RecordID')
    
    # Separate splits
    train_data = X_unimputed[X_unimputed['split'] == 'train'].copy()
    val_data = X_unimputed[X_unimputed['split'] == 'validation'].copy()
    test_data = X_unimputed[X_unimputed['split'] == 'test'].copy()
    
    # Prepare feature matrices
    feature_cols = [col for col in X_unimputed.columns if col not in ['RecordID', 'split', 'In-hospital_death']]
    
    X_train = train_data[feature_cols].values
    y_train = train_data['In-hospital_death'].values
    
    X_val = val_data[feature_cols].values
    y_val = val_data['In-hospital_death'].values
    
    X_test = test_data[feature_cols].values
    y_test = test_data['In-hospital_death'].values
    
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Class distribution
    n_pos_train = y_train.sum()
    n_neg_train = len(y_train) - n_pos_train
    prevalence_train = n_pos_train / len(y_train)
    scale_pos_weight = n_neg_train / n_pos_train
    
    print(f"Training set class distribution:")
    print(f"  Positive: {n_pos_train} ({prevalence_train:.4f})")
    print(f"  Negative: {n_neg_train} ({1-prevalence_train:.4f})")
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}")
    print()
    
    # Combine train+validation for hyperparameter search
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    n_pos_train_val = y_train_val.sum()
    n_neg_train_val = len(y_train_val) - n_pos_train_val
    scale_pos_weight_combined = n_neg_train_val / n_pos_train_val
    
    print(f"Combined train+val set: {X_train_val.shape[0]} samples")
    print(f"  Positive: {n_pos_train_val} ({n_pos_train_val/len(y_train_val):.4f})")
    print(f"  scale_pos_weight (combined): {scale_pos_weight_combined:.4f}")
    print()
    
    # Define hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    
    print("Hyperparameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    print()
    
    # Base XGBoost model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight_combined,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc',
        tree_method='hist',  # Faster for large datasets
    )
    
    # GridSearchCV with 5-fold stratified CV
    print("Starting hyperparameter search (GridSearchCV, 5-fold CV)...")
    print("This may take a while...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(lambda y_true, y_pred: 
                        compute_metrics(y_true, y_pred)['auroc'])
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=1,  # XGBoost already uses multiple cores
        verbose=2,
        refit=True
    )
    
    start_time = time()
    grid_search.fit(X_train_val, y_train_val)
    training_time = time() - start_time
    
    print()
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print()
    
    # Best parameters
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print("=" * 80)
    print("BEST HYPERPARAMETERS")
    print("=" * 80)
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"\nBest CV AUROC: {best_cv_score:.6f}")
    print()
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on validation set (for comparison)
    y_val_pred = best_model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_pred)
    
    print("=" * 80)
    print("VALIDATION SET PERFORMANCE")
    print("=" * 80)
    print(f"AUROC: {val_metrics['auroc']:.6f}")
    print(f"AUPRC: {val_metrics['auprc']:.6f}")
    print()
    
    # Evaluate on test set
    print("=" * 80)
    print("TEST SET PERFORMANCE")
    print("=" * 80)
    
    y_test_pred = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred)
    
    print(f"AUROC: {test_metrics['auroc']:.6f}")
    print(f"AUPRC: {test_metrics['auprc']:.6f}")
    print()
    
    # Bootstrap confidence intervals
    print("Computing bootstrap 95% confidence intervals (1000 iterations)...")
    
    auroc_point, auroc_lower, auroc_upper = bootstrap_ci(
        y_test, y_test_pred,
        lambda y_t, y_p: compute_metrics(y_t, y_p)['auroc'],
        n_bootstrap=1000
    )
    
    auprc_point, auprc_lower, auprc_upper = bootstrap_ci(
        y_test, y_test_pred,
        lambda y_t, y_p: compute_metrics(y_t, y_p)['auprc'],
        n_bootstrap=1000
    )
    
    print(f"\nAUROC: {auroc_point:.6f} (95% CI: [{auroc_lower:.6f}, {auroc_upper:.6f}])")
    print(f"AUPRC: {auprc_point:.6f} (95% CI: [{auprc_lower:.6f}, {auprc_upper:.6f}])")
    print()
    
    # Save model
    model_path = Path(__file__).parent.parent / 'results' / 'L4_xgboost_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"Model saved to: {model_path}")
    print()
    
    # Save detailed results
    results = {
        'model': 'XGBoost',
        'n_features': X_train.shape[1],
        'feature_set': 'all_1474_features',
        'data_type': 'unimputed_native_nan_handling',
        'class_imbalance_handling': 'scale_pos_weight',
        'scale_pos_weight': float(scale_pos_weight_combined),
        'hyperparameter_search': {
            'method': 'GridSearchCV',
            'cv_folds': 5,
            'param_grid': {k: [float(x) if isinstance(x, (int, float)) else x 
                              for x in v] for k, v in param_grid.items()},
            'n_combinations': int(np.prod([len(v) for v in param_grid.values()])),
            'best_params': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in best_params.items()},
            'best_cv_auroc': float(best_cv_score),
        },
        'training': {
            'train_samples': int(len(y_train)),
            'validation_samples': int(len(y_val)),
            'train_val_samples': int(len(y_train_val)),
            'train_prevalence': float(prevalence_train),
            'training_time_seconds': float(training_time),
        },
        'validation_performance': {
            'auroc': float(val_metrics['auroc']),
            'auprc': float(val_metrics['auprc']),
        },
        'test_performance': {
            'n_samples': int(len(y_test)),
            'prevalence': float(y_test.mean()),
            'auroc': float(auroc_point),
            'auroc_95ci_lower': float(auroc_lower),
            'auroc_95ci_upper': float(auroc_upper),
            'auprc': float(auprc_point),
            'auprc_95ci_lower': float(auprc_lower),
            'auprc_95ci_upper': float(auprc_upper),
        },
        'model_path': str(model_path),
    }
    
    results_path = Path(__file__).parent.parent / 'results' / 'L4_xgboost_training.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    print()
    
    # Feature importance (top 20)
    print("=" * 80)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("=" * 80)
    
    feature_importance = best_model.feature_importances_
    feature_names = feature_cols
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(20).iterrows():
        print(f"{row['feature']:50s} {row['importance']:.6f}")
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    results = main()
