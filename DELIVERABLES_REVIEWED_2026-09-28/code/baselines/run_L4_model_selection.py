"""
L4 Model Selection

Compares available trained models (XGBoost, CatBoost, LightGBM) and selects best.
Loads whichever result files exist — missing models are skipped gracefully.
"""
import json
from pathlib import Path

print("="*60)
print("L4 Model Selection")
print("="*60)

RESULTS_DIR = Path(__file__).parent.parent / 'results'

# Load whichever result files exist
candidates = {}
for name, fname in [
    ('xgboost',  'L4_xgboost_training.json'),
    ('catboost', 'L4_catboost_training.json'),
    ('lightgbm', 'L4_lightgbm_training.json'),
]:
    path = RESULTS_DIR / fname
    if path.exists():
        with open(path) as f:
            candidates[name] = json.load(f)
        print(f"   Loaded {fname}")
    else:
        print(f"   Skipping {fname} (not found)")

if not candidates:
    raise FileNotFoundError("No training result files found in results/")

print("\n1. Validation Performance Comparison")
print("-" * 60)
for name, r in candidates.items():
    gpu_tag = "GPU" if r.get('gpu_used') else "CPU"
    print(f"{name.upper():10s}: AUROC {r['validation_auroc']:.4f}, "
          f"AUPRC {r['validation_auprc']:.4f}  "
          f"({gpu_tag}, {r['training_time_seconds']/60:.1f} min)")
print()

# Select by AUROC (primary), AUPRC (secondary), then speed
def score_key(item):
    name, r = item
    return (r['validation_auroc'], r['validation_auprc'],
            -r['training_time_seconds'])

ranked = sorted(candidates.items(), key=score_key, reverse=True)
selected, best = ranked[0]

if len(ranked) > 1:
    second_name, second = ranked[1]
    auroc_gap = best['validation_auroc'] - second['validation_auroc']
    if auroc_gap < 0.005:
        auprc_gap = best['validation_auprc'] - second['validation_auprc']
        if auprc_gap < 0.005:
            rationale = (f"AUROC and AUPRC essentially tied with {second_name.upper()}, "
                         f"{selected.upper()} faster")
        else:
            rationale = (f"AUROC tied with {second_name.upper()}, "
                         f"{selected.upper()} has higher AUPRC")
    else:
        rationale = f"{selected.upper()} has highest validation AUROC (primary metric)"
else:
    rationale = f"Only {selected.upper()} results available"

print("\n2. Model Selection")
print("-" * 60)
print(f"Selected:  {selected.upper()}")
print(f"Rationale: {rationale}")

# Build results dict
results = {
    'selected_model': selected,
    'selection_rationale': rationale,
}
for name, r in candidates.items():
    results[f'{name}_validation_auroc'] = r['validation_auroc']
    results[f'{name}_validation_auprc'] = r['validation_auprc']

output_file = RESULTS_DIR / 'L4_model_selection.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSelection saved to {output_file.name}")
print("="*60)
