"""Validation script for data loading and integrity checks.

Verifies:
- Row counts match expected (train=8283, val=1775, test=1775, total=11833)
- Prevalence is 0.1443 across splits
- Feature count is 1474 (after dropping RecordID)
- SAPS-I and SOFA -1 sentinel counts
"""
import sys
from pathlib import Path

# Add experiment directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_data, get_split, get_feature_names
import pandas as pd
import numpy as np


def main():
    print("=" * 70)
    print("DATA LOADING VALIDATION")
    print("=" * 70)
    
    # Load all data
    print("\n1. Loading data...")
    labels_df, X_unimputed, X_imputed_splits = load_data()
    print("   ✓ Data loaded successfully")
    
    # Verify row counts
    print("\n2. Verifying row counts...")
    expected_counts = {
        "train": 8283,
        "validation": 1775,
        "test": 1775
    }
    
    all_counts_match = True
    for split_name, expected_count in expected_counts.items():
        actual_count = len(X_imputed_splits[split_name])
        match = "✓" if actual_count == expected_count else "✗"
        print(f"   {match} {split_name}: {actual_count} (expected {expected_count})")
        if actual_count != expected_count:
            all_counts_match = False
    
    total_count = sum(len(X_imputed_splits[s]) for s in expected_counts.keys())
    expected_total = 11833
    match = "✓" if total_count == expected_total else "✗"
    print(f"   {match} total: {total_count} (expected {expected_total})")
    if total_count != expected_total:
        all_counts_match = False
    
    # Verify prevalence
    print("\n3. Verifying prevalence (0.1443 expected)...")
    expected_prevalence = 0.1443
    prevalence_matches = True
    
    for split_name in expected_counts.keys():
        split_labels = labels_df[labels_df["split"] == split_name]
        prevalence = split_labels["In-hospital_death"].mean()
        match = "✓" if abs(prevalence - expected_prevalence) < 0.001 else "✗"
        print(f"   {match} {split_name}: {prevalence:.4f}")
        if abs(prevalence - expected_prevalence) >= 0.001:
            prevalence_matches = False
    
    overall_prevalence = labels_df["In-hospital_death"].mean()
    match = "✓" if abs(overall_prevalence - expected_prevalence) < 0.001 else "✗"
    print(f"   {match} overall: {overall_prevalence:.4f}")
    if abs(overall_prevalence - expected_prevalence) >= 0.001:
        prevalence_matches = False
    
    # Verify feature count
    print("\n4. Verifying feature count (1474 expected)...")
    feature_names = get_feature_names()
    feature_count = len(feature_names)
    match = "✓" if feature_count == 1474 else "✗"
    print(f"   {match} Feature count: {feature_count}")
    
    # Verify feature consistency across splits
    print("\n5. Verifying feature consistency across splits...")
    features_consistent = True
    for split_name, X_split in X_imputed_splits.items():
        # Drop RecordID to get feature columns
        feature_cols = [c for c in X_split.columns if c != "RecordID"]
        if len(feature_cols) != 1474:
            print(f"   ✗ {split_name}: {len(feature_cols)} features (expected 1474)")
            features_consistent = False
        else:
            print(f"   ✓ {split_name}: {len(feature_cols)} features")
    
    # Check SAPS-I and SOFA -1 sentinel counts
    print("\n6. Checking SAPS-I and SOFA -1 sentinel counts...")
    
    # Check in unimputed data
    if "SAPS-I" in X_unimputed.columns:
        saps_minus_one_count = (X_unimputed["SAPS-I"] == -1).sum()
        print(f"   • SAPS-I = -1: {saps_minus_one_count} records ({saps_minus_one_count/len(X_unimputed)*100:.2f}%)")
    else:
        print("   • SAPS-I column not found in unimputed data")
    
    if "SOFA" in X_unimputed.columns:
        sofa_minus_one_count = (X_unimputed["SOFA"] == -1).sum()
        print(f"   • SOFA = -1: {sofa_minus_one_count} records ({sofa_minus_one_count/len(X_unimputed)*100:.2f}%)")
    else:
        print("   • SOFA column not found in unimputed data")
    
    # Summary statistics
    print("\n7. Summary statistics:")
    print(f"   • Total records: {len(labels_df)}")
    print(f"   • Total features (excluding RecordID): {feature_count}")
    print(f"   • Positive cases: {labels_df['In-hospital_death'].sum()}")
    print(f"   • Negative cases: {(labels_df['In-hospital_death'] == 0).sum()}")
    print(f"   • Overall prevalence: {overall_prevalence:.4f}")
    
    # Check baseline columns
    print("\n8. Baseline columns in labels_and_baselines.csv:")
    baseline_cols = [c for c in labels_df.columns if c not in ["RecordID", "In-hospital_death", "split"]]
    for col in baseline_cols:
        # Count -1 values (the sentinel for missing baselines)
        minus_one_count = (labels_df[col] == -1).sum()
        print(f"   • {col}: {minus_one_count} records with -1 ({minus_one_count/len(labels_df)*100:.2f}%)")
    
    # Final validation summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = all_counts_match and prevalence_matches and feature_count == 1474 and features_consistent
    
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
    else:
        print("✗ SOME VALIDATIONS FAILED")
        if not all_counts_match:
            print("  - Row counts do not match expected values")
        if not prevalence_matches:
            print("  - Prevalence does not match expected value (0.1443)")
        if feature_count != 1474:
            print("  - Feature count does not match expected value (1474)")
        if not features_consistent:
            print("  - Feature counts inconsistent across splits")
    
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
