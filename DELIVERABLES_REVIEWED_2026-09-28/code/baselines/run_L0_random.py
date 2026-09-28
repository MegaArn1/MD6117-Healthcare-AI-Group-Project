"""
L0 Random Baseline - No-skill floor for In-hospital Death prediction.

Generates random uniform [0,1] scores and computes AUPRC/AUROC with 95% bootstrap CI.
Establishes the no-skill floor: AUPRC should ≈ prevalence (0.1443), AUROC should ≈ 0.5.
"""
import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import get_split
from src.metrics import compute_metrics, bootstrap_ci


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load test split
    print("Loading test split...")
    X_test, y_test = get_split('test')
    n_samples = len(y_test)
    prevalence = y_test.mean()
    
    print(f"Test set: {n_samples} samples, prevalence = {prevalence:.4f}")
    
    # Generate random uniform [0,1] scores
    print("\nGenerating random uniform [0,1] scores...")
    y_score_random = np.random.uniform(0, 1, size=n_samples)
    
    # Compute point estimates
    print("\nComputing AUPRC and AUROC...")
    metrics = compute_metrics(y_test, y_score_random)
    auprc_point = metrics['auprc']
    auroc_point = metrics['auroc']
    
    # Compute bootstrap confidence intervals
    print("Computing 95% bootstrap confidence intervals (1000 iterations)...")
    
    auprc_estimate, auprc_lower, auprc_upper = bootstrap_ci(
        y_test, y_score_random,
        metric_fn=lambda y_t, y_s: compute_metrics(y_t, y_s)['auprc'],
        n_bootstrap=1000,
        ci=0.95
    )
    
    auroc_estimate, auroc_lower, auroc_upper = bootstrap_ci(
        y_test, y_score_random,
        metric_fn=lambda y_t, y_s: compute_metrics(y_t, y_s)['auroc'],
        n_bootstrap=1000,
        ci=0.95
    )
    
    # Package results
    results = {
        'model': 'L0_random_baseline',
        'description': 'Random uniform [0,1] scores - no-skill floor',
        'test_samples': int(n_samples),
        'prevalence': float(prevalence),
        'auprc': float(auprc_point),
        'auprc_ci': [float(auprc_lower), float(auprc_upper)],
        'auroc': float(auroc_point),
        'auroc_ci': [float(auroc_lower), float(auroc_upper)],
        'theoretical_auprc': float(prevalence),
        'theoretical_auroc': 0.5
    }
    
    # Save results
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "L0_random_baseline.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print formatted results
    print("\n" + "="*60)
    print("L0 RANDOM BASELINE RESULTS")
    print("="*60)
    print(f"Test samples: {n_samples:,}")
    print(f"Prevalence:   {prevalence:.4f}")
    print()
    print(f"AUPRC:        {auprc_point:.4f}  (95% CI: [{auprc_lower:.4f}, {auprc_upper:.4f}])")
    print(f"  Theoretical: {prevalence:.4f}  (prevalence)")
    print(f"  Difference:  {auprc_point - prevalence:+.4f}")
    print()
    print(f"AUROC:        {auroc_point:.4f}  (95% CI: [{auroc_lower:.4f}, {auroc_upper:.4f}])")
    print(f"  Theoretical: 0.5000  (no discrimination)")
    print(f"  Difference:  {auroc_point - 0.5:+.4f}")
    print("="*60)
    print("\nNo-skill floor established: Random guessing confirmed.")
    
    return results


if __name__ == '__main__':
    main()
