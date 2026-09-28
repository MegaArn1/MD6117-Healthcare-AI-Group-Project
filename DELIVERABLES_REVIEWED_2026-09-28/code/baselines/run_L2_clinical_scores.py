#!/usr/bin/env python3
"""
L2 Clinical Scores Baseline Analysis

Analyzes SAPS-I and SOFA -1 sentinel issue and computes baseline performance
in three scenarios: keep -1 as numeric, exclude -1 rows, replace -1 with median.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_metrics(y_true, y_score):
    """Compute AUROC and AUPRC."""
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    return {"auroc": auroc, "auprc": auprc}

def analyze_clinical_score(df, score_col, available_col, label_col):
    """Analyze a clinical score across three scenarios."""
    
    # Filter to test set only
    test_df = df[df['split'] == 'test'].copy()
    n_total = len(test_df)
    
    # Identify -1 rows (missing/not computable)
    # In the CSV, -1 is stored as actual -1.0, and available flag is 0
    minus_one_mask = test_df[available_col] == 0
    n_minus_one = minus_one_mask.sum()
    
    # Available rows (where score is computable)
    available_mask = test_df[available_col] == 1
    n_available = available_mask.sum()
    
    # Mortality rates
    overall_mortality = test_df[label_col].mean()
    minus_one_mortality = test_df.loc[minus_one_mask, label_col].mean() if n_minus_one > 0 else None
    available_mortality = test_df.loc[available_mask, label_col].mean()
    
    results = {
        "score_name": score_col.upper().replace('_', '-'),
        "total_test_rows": int(n_total),
        "minus_one_count": int(n_minus_one),
        "minus_one_percentage": float(n_minus_one / n_total * 100),
        "available_count": int(n_available),
        "overall_mortality_rate": float(overall_mortality),
        "minus_one_mortality_rate": float(minus_one_mortality) if minus_one_mortality is not None else None,
        "available_mortality_rate": float(available_mortality),
    }
    
    # Scenario A: Keep -1 as numeric (inverted risk)
    # For scores where -1 exists, it's treated as lowest score
    y_true_full = test_df[label_col].values
    y_score_full = test_df[score_col].fillna(-1).values  # Handle any NaNs
    
    metrics_full = compute_metrics(y_true_full, y_score_full)
    results["full_test_metrics"] = {
        "scenario": "keep_-1_as_numeric",
        "n": int(n_total),
        "auroc": float(metrics_full["auroc"]),
        "auprc": float(metrics_full["auprc"])
    }
    
    # Scenario B: Exclude -1 rows (subset evaluation)
    available_df = test_df[available_mask].copy()
    y_true_available = available_df[label_col].values
    y_score_available = available_df[score_col].values
    
    metrics_available = compute_metrics(y_true_available, y_score_available)
    results["available_only_metrics"] = {
        "scenario": "exclude_-1_rows",
        "n": int(n_available),
        "auroc": float(metrics_available["auroc"]),
        "auprc": float(metrics_available["auprc"])
    }
    
    # Scenario C: Replace -1 with median of available scores
    median_score = available_df[score_col].median()
    test_df_median = test_df.copy()
    test_df_median.loc[minus_one_mask, score_col] = median_score
    
    y_true_median = test_df_median[label_col].values
    y_score_median = test_df_median[score_col].values
    
    metrics_median = compute_metrics(y_true_median, y_score_median)
    results["median_replacement_metrics"] = {
        "scenario": "replace_-1_with_median",
        "n": int(n_total),
        "median_value": float(median_score),
        "auroc": float(metrics_median["auroc"]),
        "auprc": float(metrics_median["auprc"])
    }
    
    # Impact quantification
    results["auroc_impact"] = float(metrics_available["auroc"] - metrics_full["auroc"])
    results["auprc_impact"] = float(metrics_available["auprc"] - metrics_full["auprc"])
    
    return results

def main():
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    labels_file = data_dir / "labels_and_baselines.csv"
    output_file = results_dir / "L2_clinical_scores_analysis.json"
    
    print(f"Loading data from {labels_file}...")
    df = pd.read_csv(labels_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSplit distribution:")
    print(df['split'].value_counts())
    
    # Analyze SAPS-I
    print("\n" + "="*60)
    print("Analyzing SAPS-I...")
    print("="*60)
    saps_results = analyze_clinical_score(df, 'saps_i', 'saps_i_available', 'In-hospital_death')
    
    print(f"\nSAPS-I Analysis:")
    print(f"  Total test rows: {saps_results['total_test_rows']}")
    print(f"  Rows with -1: {saps_results['minus_one_count']} ({saps_results['minus_one_percentage']:.2f}%)")
    print(f"  Available rows: {saps_results['available_count']}")
    print(f"  Overall mortality: {saps_results['overall_mortality_rate']:.4f}")
    print(f"  -1 rows mortality: {saps_results['minus_one_mortality_rate']:.4f}")
    print(f"  Available rows mortality: {saps_results['available_mortality_rate']:.4f}")
    print(f"\n  Scenario A (keep -1 as numeric):")
    print(f"    AUROC: {saps_results['full_test_metrics']['auroc']:.4f}")
    print(f"    AUPRC: {saps_results['full_test_metrics']['auprc']:.4f}")
    print(f"\n  Scenario B (exclude -1 rows):")
    print(f"    AUROC: {saps_results['available_only_metrics']['auroc']:.4f}")
    print(f"    AUPRC: {saps_results['available_only_metrics']['auprc']:.4f}")
    print(f"\n  Scenario C (replace -1 with median={saps_results['median_replacement_metrics']['median_value']:.1f}):")
    print(f"    AUROC: {saps_results['median_replacement_metrics']['auroc']:.4f}")
    print(f"    AUPRC: {saps_results['median_replacement_metrics']['auprc']:.4f}")
    print(f"\n  Impact (B - A):")
    print(f"    ΔAUROC: +{saps_results['auroc_impact']:.4f}")
    print(f"    ΔAUPRC: +{saps_results['auprc_impact']:.4f}")
    
    # Analyze SOFA
    print("\n" + "="*60)
    print("Analyzing SOFA...")
    print("="*60)
    sofa_results = analyze_clinical_score(df, 'sofa', 'sofa_available', 'In-hospital_death')
    
    print(f"\nSOFA Analysis:")
    print(f"  Total test rows: {sofa_results['total_test_rows']}")
    print(f"  Rows with -1: {sofa_results['minus_one_count']} ({sofa_results['minus_one_percentage']:.2f}%)")
    print(f"  Available rows: {sofa_results['available_count']}")
    print(f"  Overall mortality: {sofa_results['overall_mortality_rate']:.4f}")
    print(f"  -1 rows mortality: {sofa_results['minus_one_mortality_rate']:.4f}")
    print(f"  Available rows mortality: {sofa_results['available_mortality_rate']:.4f}")
    print(f"\n  Scenario A (keep -1 as numeric):")
    print(f"    AUROC: {sofa_results['full_test_metrics']['auroc']:.4f}")
    print(f"    AUPRC: {sofa_results['full_test_metrics']['auprc']:.4f}")
    print(f"\n  Scenario B (exclude -1 rows):")
    print(f"    AUROC: {sofa_results['available_only_metrics']['auroc']:.4f}")
    print(f"    AUPRC: {sofa_results['available_only_metrics']['auprc']:.4f}")
    print(f"\n  Scenario C (replace -1 with median={sofa_results['median_replacement_metrics']['median_value']:.1f}):")
    print(f"    AUROC: {sofa_results['median_replacement_metrics']['auroc']:.4f}")
    print(f"    AUPRC: {sofa_results['median_replacement_metrics']['auprc']:.4f}")
    print(f"\n  Impact (B - A):")
    print(f"    ΔAUROC: +{sofa_results['auroc_impact']:.4f}")
    print(f"    ΔAUPRC: +{sofa_results['auprc_impact']:.4f}")
    
    # Recommendation
    recommendation = {
        "summary": "Report baseline on score-available subset, evaluate model on same subset for head-to-head comparison",
        "rationale": [
            "The -1 sentinel (not computable) indicates missing data, not a score of -1",
            f"SAPS-I: {saps_results['minus_one_count']} rows ({saps_results['minus_one_percentage']:.2f}%) have -1, with {saps_results['minus_one_mortality_rate']:.1%} mortality vs {saps_results['available_mortality_rate']:.1%} for available",
            f"SOFA: {sofa_results['minus_one_count']} rows ({sofa_results['minus_one_percentage']:.2f}%) have -1, with {sofa_results['minus_one_mortality_rate']:.1%} mortality vs {sofa_results['available_mortality_rate']:.1%} for available",
            "Keeping -1 as numeric inverts risk (high-risk patients scored as lowest risk)",
            f"Excluding -1 rows recovers {saps_results['auroc_impact']:.4f} AUROC and {saps_results['auprc_impact']:.4f} AUPRC for SAPS-I",
            f"Excluding -1 rows recovers {sofa_results['auroc_impact']:.4f} AUROC and {sofa_results['auprc_impact']:.4f} AUPRC for SOFA",
            "Median replacement is better than keeping -1 but still biases comparison toward the model"
        ],
        "reporting_protocol": {
            "for_clinical_baselines": "Report AUROC/AUPRC on score-available subset only (SAPS-I n=1728, SOFA n=1733)",
            "for_ml_model_comparison": "Evaluate model on same score-available subset for fair head-to-head comparison",
            "for_ml_model_full_test": "Also report model performance on full test set (n=1775) with clear denominator",
            "always_specify": "Always name the denominator (n=1728 for SAPS-I subset, n=1733 for SOFA subset, n=1775 for full test)"
        }
    }
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print(f"\n{recommendation['summary']}\n")
    print("Rationale:")
    for i, point in enumerate(recommendation['rationale'], 1):
        print(f"  {i}. {point}")
    print("\nReporting Protocol:")
    for key, value in recommendation['reporting_protocol'].items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Save results
    output = {
        "analysis_date": "2026-09-23",
        "data_source": str(labels_file),
        "saps_i_results": saps_results,
        "sofa_results": sofa_results,
        "recommendation": recommendation
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
