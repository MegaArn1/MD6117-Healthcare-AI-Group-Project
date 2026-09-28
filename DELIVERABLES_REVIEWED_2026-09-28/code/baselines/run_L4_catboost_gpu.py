"""
L4 CatBoost Training with GPU Acceleration

- GPU via task_type='GPU' (CUDA, no OpenCL needed)
- Native NaN handling
- auto_class_weights='Balanced' for class imbalance
- Manual random-search loop with checkpoint/resume support:
    results/L4_catboost_search_checkpoint.json is written after every
    completed candidate; re-running the script skips already-done ones.
- Live log written to results/L4_catboost_training.log
- OOM safeguards: depth ≤ 8, border_count ≤ 128, max_ctr_complexity=2
"""
import sys
import json
import time
import logging
import threading
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import check_random_state
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_data
from metrics import compute_metrics

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path(__file__).parent.parent / 'results'
MODELS_DIR    = Path(__file__).parent.parent / 'models'
CHECKPOINT    = RESULTS_DIR / 'L4_catboost_search_checkpoint.json'
LOG_FILE      = RESULTS_DIR / 'L4_catboost_training.log'
MODELS_DIR.mkdir(exist_ok=True)

# ── Logging: both stdout and file ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='a'),
    ],
)
log = logging.getLogger(__name__)

log.info("=" * 60)
log.info("L4 CatBoost Training (GPU)")
log.info("=" * 60)

start_time = time.time()

# ── 1. Data ───────────────────────────────────────────────────────────────────
log.info("\n1. Loading unimputed data...")
labels_df, X_unimputed, _ = load_data()

train_df = labels_df[labels_df['split'] == 'train'].merge(X_unimputed, on='RecordID')
val_df   = labels_df[labels_df['split'] == 'validation'].merge(X_unimputed, on='RecordID')
test_df  = labels_df[labels_df['split'] == 'test'].merge(X_unimputed, on='RecordID')

feature_cols = [c for c in X_unimputed.columns if c != 'RecordID']
log.info(f"   Features: {len(feature_cols)}")

X_train = train_df[feature_cols].values;  y_train = train_df['In-hospital_death'].values
X_val   = val_df[feature_cols].values;    y_val   = val_df['In-hospital_death'].values
X_test  = test_df[feature_cols].values;   y_test  = test_df['In-hospital_death'].values

log.info(f"   Train: {len(y_train)} samples, {int(y_train.sum())} deaths")
log.info(f"   Val:   {len(y_val)} samples, {int(y_val.sum())} deaths")
log.info(f"   Test:  {len(y_test)} samples, {int(y_test.sum())} deaths")
log.info(f"   class imbalance (neg/pos): {float((y_train==0).sum()/y_train.sum()):.2f} — auto_class_weights='Balanced'")

# ── 2. GPU check ──────────────────────────────────────────────────────────────
log.info("\n2. Checking GPU availability...")
try:
    CatBoostClassifier(iterations=10, task_type='GPU', devices='0',
                       verbose=False, allow_writing_files=False
                       ).fit(X_train[:100], y_train[:100])
    log.info("   ✓ GPU available and working")
    use_gpu = True
except Exception as e:
    log.info(f"   ✗ GPU not available, falling back to CPU: {e}")
    use_gpu = False

# ── 3. Base params (fixed for all candidates) ─────────────────────────────────
base_params = dict(
    loss_function='Logloss',
    eval_metric='AUC',
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    # OOM safeguards for GPU with 1474 features on 24 GB VRAM:
    max_ctr_complexity=2,   # limits GPU memory for feature combinations
)
if use_gpu:
    base_params['task_type'] = 'GPU'
    base_params['devices']   = '0'

# ── 4. Search space (OOM-safe bounds) ─────────────────────────────────────────
PARAM_GRID = {
    'depth':               [4, 6, 8],          # was [4,6,8,10] — 10 OOMs
    'learning_rate':       [0.01, 0.03, 0.05, 0.1],
    'iterations':          [200, 300, 500],
    'l2_leaf_reg':         [1, 3, 5, 9],
    'border_count':        [32, 64, 128],       # was [32,64,128,254] — 254 OOMs
    'random_strength':     [0.5, 1, 3],
    'bagging_temperature': [0, 0.5, 1],
}
N_ITER   = 50
N_SPLITS = 5

# ── 5. Load checkpoint (resume support) ───────────────────────────────────────
if CHECKPOINT.exists():
    with open(CHECKPOINT) as f:
        ckpt = json.load(f)
    completed_results = ckpt.get('results', [])
    rng_state         = ckpt.get('rng_state', None)
    log.info(f"\n   Resuming from checkpoint: {len(completed_results)}/{N_ITER} iterations already done")
else:
    completed_results = []
    rng_state         = None
    log.info(f"\n3. Hyperparameter search: {N_ITER} iterations × {N_SPLITS} folds")

# Reconstruct the full candidate list with a fixed seed so resume is deterministic
rng = check_random_state(42)
all_candidates = []
for _ in range(N_ITER):
    candidate = {k: rng.choice(v).item() for k, v in PARAM_GRID.items()}
    all_candidates.append(candidate)

n_done = len(completed_results)
remaining = all_candidates[n_done:]

log.info(f"   {n_done} done, {len(remaining)} remaining")

# ── 6. Manual search loop ─────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Background heartbeat
_search_start = time.time()
_search_done  = threading.Event()

def _heartbeat():
    while not _search_done.wait(timeout=30):
        elapsed = time.time() - _search_start
        best_so_far = max((r['mean_cv_auroc'] for r in completed_results), default=float('nan'))
        log.info(f"   Still searching... elapsed {elapsed/60:.1f} min | "
                 f"best so far: {best_so_far:.4f} | "
                 f"done {len(completed_results)}/{N_ITER}")

threading.Thread(target=_heartbeat, daemon=True).start()

for i, candidate in enumerate(remaining, start=n_done + 1):
    iter_start = time.time()
    model = CatBoostClassifier(**base_params, **candidate)
    try:
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring='roc_auc', n_jobs=1)
        mean_auc = float(scores.mean())
        std_auc  = float(scores.std())
    except Exception as e:
        log.warning(f"   Iter {i}: FAILED ({e}) — skipping")
        mean_auc = float('nan')
        std_auc  = float('nan')

    elapsed_iter = time.time() - iter_start
    completed_results.append({
        'iteration': i,
        'params': candidate,
        'mean_cv_auroc': mean_auc,
        'std_cv_auroc': std_auc,
        'time_seconds': elapsed_iter,
    })

    log.info(f"   Iter {i:2d}/{N_ITER} | AUROC {mean_auc:.4f} ± {std_auc:.4f} | "
             f"{elapsed_iter:.0f}s | {candidate}")

    # Save checkpoint after every iteration
    with open(CHECKPOINT, 'w') as f:
        json.dump({'results': completed_results}, f, indent=2)

_search_done.set()

# ── 7. Best params ────────────────────────────────────────────────────────────
valid = [r for r in completed_results if not np.isnan(r['mean_cv_auroc'])]
if not valid:
    raise RuntimeError("All search iterations failed — check GPU memory or param bounds")

best_result = max(valid, key=lambda r: r['mean_cv_auroc'])
best_params = best_result['params']

log.info(f"\n   Best CV AUROC: {best_result['mean_cv_auroc']:.4f}")
log.info(f"   Best params: {best_params}")

# ── 8. Final model with early stopping on val set ─────────────────────────────
log.info("\n4. Training final model with best params + early stopping...")
train_pool = Pool(X_train, y_train)
val_pool   = Pool(X_val,   y_val)

final_params = {**base_params, **best_params, 'iterations': 2000}
best_model = CatBoostClassifier(**final_params)
best_model.fit(train_pool, eval_set=val_pool,
               early_stopping_rounds=50, verbose=False)

y_val_proba = best_model.predict_proba(X_val)[:, 1]
val_metrics = compute_metrics(y_val, y_val_proba)
log.info(f"   Val AUROC: {val_metrics['auroc']:.4f}")
log.info(f"   Val AUPRC: {val_metrics['auprc']:.4f}")

# ── 9. Save model and results ─────────────────────────────────────────────────
model_path = MODELS_DIR / 'L4_catboost_best.cbm'
best_model.save_model(str(model_path))
log.info(f"\n5. Model saved to {model_path.name}")

elapsed = time.time() - start_time
log.info(f"\nTotal training time: {elapsed/60:.1f} minutes")

results = {
    'model': 'catboost',
    'gpu_used': use_gpu,
    'n_features': len(feature_cols),
    'best_hyperparameters': best_params,
    'cv_best_auroc': best_result['mean_cv_auroc'],
    'cv_best_auroc_std': best_result['std_cv_auroc'],
    'validation_auroc': float(val_metrics['auroc']),
    'validation_auprc': float(val_metrics['auprc']),
    'training_time_seconds': float(elapsed),
    'auto_class_weights': 'Balanced',
    'train_samples': int(len(y_train)),
    'val_samples':   int(len(y_val)),
    'test_samples':  int(len(y_test)),
}

output_file = RESULTS_DIR / 'L4_catboost_training.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

log.info(f"\nResults saved to {output_file.name}")
log.info("=" * 60)
