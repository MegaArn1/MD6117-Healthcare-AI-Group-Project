"""
Stage 2: L3 Baseline - Small Logistic Regression (5-10 features)

Clinical feature selection rationale (实验方案 §5.5):
- L3 is the real opponent for L4, not SAPS-I
- Features selected based on clinical meaning, not pure statistical screening
- Cover multiple organ systems aligned with SOFA domains
- Include L1 best performers where clinically appropriate
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data, get_split
from metrics import compute_metrics, bootstrap_ci, find_threshold_for_recall

# Selected features with clinical justification
SELECTED_FEATURES = [
    'bun__0_48h__max',              # 1. Renal - L1 best (AUROC=0.675)
    'static_age_years',              # 2. Demographic - L1 second best (AUROC=0.627)
    'gcs__0_48h__min',               # 3. CNS (SOFA neurological)
    'urine__0_48h__min',             # 4. Renal output - oliguria marker
    'lactate__0_48h__max',           # 5. Perfusion/shock marker
    'platelets__0_48h__min',         # 6. Coagulation (SOFA coagulation)
    'creatinine__0_48h__max',        # 7. Renal - AKI marker
    'lactate__0_48h__measured',      # 8. Measurement behavior
]

CLINICAL_RATIONALE = {
    'bun__0_48h__max': {
        'domain': 'Renal',
        'sofa_alignment': 'SOFA renal component',
        'rationale': 'Blood urea nitrogen elevation indicates renal dysfunction and catabolic state. Best L1 single variable (AUROC=0.675)',
        'direction': 'Higher BUN → higher mortality risk'
    },
    'static_age_years': {
        'domain': 'Demographic',
        'sofa_alignment': 'Risk stratification baseline',
        'rationale': 'Age is strongest demographic predictor of ICU mortality, captures physiologic reserve. L1 AUROC=0.627',
        'direction': 'Older age → higher mortality risk'
    },
    'gcs__0_48h__min': {
        'domain': 'CNS',
        'sofa_alignment': 'SOFA neurological component',
        'rationale': 'Glasgow Coma Scale minimum captures worst neurological status. Core ICU assessment, directly used in SOFA',
        'direction': 'Lower GCS → higher mortality risk (worse consciousness)'
    },
    'urine__0_48h__min': {
        'domain': 'Renal',
        'sofa_alignment': 'SOFA renal component',
        'rationale': 'Minimum urine output captures worst oliguria episode. Oliguria/anuria is strong mortality signal (实验方案 §3: Urine=0 is real, not missing)',
        'direction': 'Lower urine output → higher mortality risk'
    },
    'lactate__0_48h__max': {
        'domain': 'Cardiovascular / Perfusion',
        'sofa_alignment': 'SOFA cardiovascular proxy',
        'rationale': 'Peak lactate indicates tissue hypoperfusion and anaerobic metabolism. Gold standard shock marker, predicts multiple organ failure',
        'direction': 'Higher lactate → higher mortality risk (shock/hypoperfusion)'
    },
    'platelets__0_48h__min': {
        'domain': 'Coagulation',
        'sofa_alignment': 'SOFA coagulation component',
        'rationale': 'Minimum platelet count captures worst coagulopathy. Thrombocytopenia indicates consumption, DIC, or bone marrow failure',
        'direction': 'Lower platelets → higher mortality risk'
    },
    'creatinine__0_48h__max': {
        'domain': 'Renal',
        'sofa_alignment': 'SOFA renal component',
        'rationale': 'Peak creatinine is standard AKI marker. Complements BUN with different kinetics (less affected by catabolic state)',
        'direction': 'Higher creatinine → higher mortality risk (AKI)'
    },
    'lactate__0_48h__measured': {
        'domain': 'Measurement behavior',
        'sofa_alignment': 'Physician decision signal',
        'rationale': 'Whether lactate was ordered reflects physician concern. Measurement behavior captures severity beyond values (实验方案 §3, §8: "缺失即信息")',
        'direction': 'Lactate measured → physician suspected shock → higher risk'
    },
}

def main():
    print("=" * 80)
    print("L3 BASELINE: LOGISTIC REGRESSION (8 HAND-SELECTED FEATURES)")
    print("=" * 80)
    print()
    
    # Display feature selection rationale
    print("CLINICAL FEATURE SELECTION:")
    print(f"Total features: {len(SELECTED_FEATURES)}")
    print()
    
    # Count organ systems
    organ_systems = set(CLINICAL_RATIONALE[f]['domain'] for f in SELECTED_FEATURES)
    print(f"Organ systems covered: {len(organ_systems)}")
    for system in sorted(organ_systems):
        features = [f for f in SELECTED_FEATURES if CLINICAL_RATIONALE[f]['domain'] == system]
        print(f"  - {system}: {len(features)} feature(s)")
    print()
    
    print("FEATURE DETAILS:")
    for i, feat in enumerate(SELECTED_FEATURES, 1):
        info = CLINICAL_RATIONALE[feat]
        print(f"{i}. {feat}")
        print(f"   Domain: {info['domain']}")
        print(f"   SOFA: {info['sofa_alignment']}")
        print(f"   Rationale: {info['rationale']}")
        print(f"   Direction: {info['direction']}")
        print()
    
    # Load data
    print("Loading data...")
    load_data()
    X_train, y_train = get_split('train')
    X_val, y_val = get_split('validation')
    X_test, y_test = get_split('test')
    
    # Extract selected features
    X_train_sel = X_train[SELECTED_FEATURES].copy()
    X_val_sel = X_val[SELECTED_FEATURES].copy()
    X_test_sel = X_test[SELECTED_FEATURES].copy()
    
    print(f"Train: {X_train_sel.shape}, prevalence={y_train.mean():.4f}")
    print(f"Val:   {X_val_sel.shape}, prevalence={y_val.mean():.4f}")
    print(f"Test:  {X_test_sel.shape}, prevalence={y_test.mean():.4f}")
    print()
    
    # Check missingness
    print("Missingness check (before imputation):")
    for feat in SELECTED_FEATURES:
        missing_train = X_train_sel[feat].isna().sum()
        missing_pct = 100 * missing_train / len(X_train_sel)
        print(f"  {feat}: {missing_train} ({missing_pct:.2f}%)")
    print()
    
    # Median imputation from training set
    print("Applying median imputation...")
    imputation_values = {}
    for feat in SELECTED_FEATURES:
        if X_train_sel[feat].isna().any():
            median_val = X_train_sel[feat].median()
            imputation_values[feat] = median_val
            X_train_sel[feat].fillna(median_val, inplace=True)
            X_val_sel[feat].fillna(median_val, inplace=True)
            X_test_sel[feat].fillna(median_val, inplace=True)
            print(f"  {feat}: imputed with {median_val:.2f}")
    print()
    
    # Standardization (fit on train, apply to all)
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    print()
    
    # Train logistic regression with L2 regularization
    print("Training logistic regression (L2 penalty, C=1.0)...")
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train_scaled, y_train)
    print("Training complete.")
    print()
    
    # Display feature coefficients
    print("FEATURE COEFFICIENTS (standardized):")
    coefs = model.coef_[0]
    feature_importance = sorted(
        zip(SELECTED_FEATURES, coefs),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for feat, coef in feature_importance:
        direction = "↑ risk" if coef > 0 else "↓ risk"
        print(f"  {feat:30s}: {coef:+.4f}  {direction}")
    print()
    
    # Predict on validation set for threshold selection
    print("Validation set evaluation...")
    y_score_val = model.predict_proba(X_val_scaled)[:, 1]
    val_metrics = compute_metrics(y_val, y_score_val)
    print(f"  AUROC: {val_metrics['auroc']:.4f}")
    print(f"  AUPRC: {val_metrics['auprc']:.4f}")
    print()
    
    # Find threshold for recall ≈ 0.80
    print("Finding threshold for target recall=0.80...")
    threshold = find_threshold_for_recall(y_val, y_score_val, target_recall=0.80)
    print(f"  Selected threshold: {threshold:.4f}")
    
    # Apply threshold to validation set to verify
    val_threshold_metrics = compute_metrics(y_val, y_score_val, threshold=threshold)
    print(f"  At this threshold on validation:")
    print(f"    Sensitivity: {val_threshold_metrics['sensitivity']:.4f}")
    print(f"    PPV:         {val_threshold_metrics['ppv']:.4f}")
    print(f"    Specificity: {val_threshold_metrics['specificity']:.4f}")
    print()
    
    # Test set evaluation
    print("=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    y_score_test = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics without threshold
    print("\nMetrics (ranking performance):")
    test_metrics = compute_metrics(y_test, y_score_test)
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}")
    print()
    
    # Bootstrap confidence intervals
    print("Computing bootstrap 95% confidence intervals (1000 resamples)...")
    auroc_ci = bootstrap_ci(y_test, y_score_test, lambda yt, ys: roc_auc_score(yt, ys), n_bootstrap=1000)
    auprc_ci = bootstrap_ci(y_test, y_score_test, lambda yt, ys: average_precision_score(yt, ys), n_bootstrap=1000)
    print(f"  AUROC 95% CI: [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]")
    print(f"  AUPRC 95% CI: [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]")
    print()
    
    # Apply threshold (from validation) to test set
    print(f"Metrics at threshold={threshold:.4f} (selected on validation):")
    test_threshold_metrics = compute_metrics(y_test, y_score_test, threshold=threshold)
    print(f"  Sensitivity: {test_threshold_metrics['sensitivity']:.4f}")
    print(f"  PPV:         {test_threshold_metrics['ppv']:.4f}")
    print(f"  Specificity: {test_threshold_metrics['specificity']:.4f}")
    print()
    
    # PhysioNet 2012 challenge metric: min(Se, PPV)
    min_se_ppv = min(test_threshold_metrics['sensitivity'], test_threshold_metrics['ppv'])
    print(f"PhysioNet 2012 Challenge metric min(Se, PPV): {min_se_ppv:.4f}")
    print()
    
    # Save results
    results = {
        'model': 'L3_logistic_regression_8features',
        'description': 'Logistic regression with 8 hand-selected features (clinical rationale)',
        'n_features': len(SELECTED_FEATURES),
        'features': SELECTED_FEATURES,
        'clinical_rationale': CLINICAL_RATIONALE,
        'organ_systems_covered': sorted(list(organ_systems)),
        'imputation': {
            'method': 'median',
            'values': imputation_values
        },
        'regularization': {
            'penalty': 'l2',
            'C': 1.0,
            'class_weight': 'balanced'
        },
        'feature_coefficients': {
            feat: float(coef) for feat, coef in zip(SELECTED_FEATURES, coefs)
        },
        'threshold': {
            'value': float(threshold),
            'selection_method': 'target_recall_0.80_on_validation'
        },
        'test_samples': int(len(y_test)),
        'prevalence': float(y_test.mean()),
        'auroc': float(test_metrics['auroc']),
        'auroc_ci': [float(auroc_ci[0]), float(auroc_ci[1])],
        'auprc': float(test_metrics['auprc']),
        'auprc_ci': [float(auprc_ci[0]), float(auprc_ci[1])],
        'sensitivity': float(test_threshold_metrics['sensitivity']),
        'ppv': float(test_threshold_metrics['ppv']),
        'specificity': float(test_threshold_metrics['specificity']),
        'min_se_ppv': float(min_se_ppv)
    }
    
    output_path = Path(__file__).parent.parent / 'results' / 'L3_logistic_regression_8features.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print()
    print("=" * 80)
    print("L3 BASELINE COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
