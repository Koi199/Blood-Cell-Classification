"""
kfold_validator.py
──────────────────
Modular stratified k-fold validation wrapper.
Works with any classifier that follows the standard library pattern
(classifier_stage1, classifier_stage2, classifier_3class, classifier_rbc_count, etc.)

Runs k-fold alongside the existing train/val/test split — does not replace it.
Each fold trains a fresh model and reports per-fold and aggregate metrics.
Results are logged to MLflow as a separate k-fold experiment and saved to CSV.

Usage:
    from kfold_validator import kfold_validate
    from classifier_stage1 import train, DEFAULT_CONFIG, load_samples, COLLAPSE_MAP, LABEL_MAP
    import copy

    config = copy.deepcopy(DEFAULT_CONFIG)
    kfold_validate(
        train_fn       = train,
        config         = config,
        load_fn        = load_samples,
        label_key      = "binary",      # "binary" collapses via COLLAPSE_MAP, "raw" uses training label directly
        collapse_map   = COLLAPSE_MAP,  # only needed if label_key="binary"
        experiment_name= "Stage1_KFold",
        notes          = "resnet34 baseline kfold",
        n_splits       = 5,
    )
"""

import os
import copy
import numpy as np
import pandas as pd
import mlflow
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import sys
sys.path.append("C:/repos/Blood-Cell-Classification/Scripts/Logging")
from Logger import setup_mlflow, end_run

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def aggregate_results(fold_results):
    """
    Compute mean and std across all folds for each metric.
    Returns a flat dict: {metric_mean: val, metric_std: val, ...}
    """
    if not fold_results:
        return {}

    all_keys = [k for k in fold_results[0].keys() if k != "fold"]
    aggregated = {}
    for key in all_keys:
        values = [r[key] for r in fold_results if isinstance(r.get(key), (int, float))]
        if values:
            aggregated[f"{key}_mean"] = round(float(np.mean(values)), 4)
            aggregated[f"{key}_std"]  = round(float(np.std(values)),  4)
    return aggregated


def print_fold_summary(fold_results, aggregate):
    print(f"\n── K-Fold Summary ──")
    print(f"{'Fold':>6} {'test_acc':>10} {'macro_f1':>10} {'weighted_f1':>13}")
    print("-" * 45)
    for r in fold_results:
        print(f"  {r['fold']:>4}  {r['test_acc']:>10.4f}  {r['macro_f1']:>10.4f}  {r['weighted_f1']:>13.4f}")
    print("-" * 45)
    print(f"  Mean  {aggregate['test_acc_mean']:>10.4f}  {aggregate['macro_f1_mean']:>10.4f}  {aggregate['weighted_f1_mean']:>13.4f}")
    print(f"  Std   {aggregate['test_acc_std']:>10.4f}  {aggregate['macro_f1_std']:>10.4f}  {aggregate['weighted_f1_std']:>13.4f}")


# ─────────────────────────────────────────────
# CORE K-FOLD FUNCTION
# ─────────────────────────────────────────────
def kfold_validate(
    train_fn,
    config,
    load_fn,
    label_key="raw",
    collapse_map=None,
    experiment_name="KFold_Validation",
    notes="",
    n_splits=5,
    results_csv=None,
):
    """
    Run stratified k-fold validation on any classifier.
    Args:
        train_fn:         the train() function from any classifier module
        config:           config dict — checkpoint_path will be modified per fold
        load_fn:          load_samples() function from the same classifier module
                          must return list of (path, fine_label, training_label) tuples
        label_key:        "binary"  → stratify on COLLAPSE_MAP[training_label]
                          "raw"     → stratify on training_label directly
        collapse_map:     dict mapping training_label → binary label (only for label_key="binary")
        experiment_name:  MLflow experiment name for k-fold runs
        notes:            description logged to MLflow
        n_splits:         number of folds (default 5)
        results_csv:      path to save per-fold CSV results
                          (defaults to 
                          
                          _path dir / kfold_results.csv)

    Returns:
        aggregate: dict of {metric_mean, metric_std} across all folds
    """
    arch        = config.get("architecture", "resnet34")
    base_ckpt   = config["checkpoint_path"]
    ckpt_dir    = os.path.dirname(base_ckpt)
    ckpt_stem   = os.path.splitext(os.path.basename(base_ckpt))[0]

    if results_csv is None:
        results_csv = os.path.join(ckpt_dir, f"{ckpt_stem}_kfold_results.csv")

    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'#'*65}")
    print(f"  Stratified {n_splits}-Fold Validation")
    print(f"  Architecture: {arch}")
    print(f"  Experiment:   {experiment_name}")
    print(f"{'#'*65}\n")

    # ── Load all samples ──
    print("Loading all samples for k-fold split...")
    all_samples = load_fn(config["data_dir"])

    # ── Get stratification labels ──
    if label_key == "binary" and collapse_map is not None:
        strat_labels = [collapse_map[s[2]] for s in all_samples]
    else:
        strat_labels = [s[2] for s in all_samples]

    strat_labels = np.array(strat_labels)
    indices      = np.arange(len(all_samples))

    # ── K-Fold split ──
    skf          = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    setup_mlflow(experiment_name)

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, strat_labels)):
        fold_num = fold_idx + 1
        print(f"\n{'='*65}")
        print(f"  Fold {fold_num}/{n_splits}  —  "
              f"Train+Val: {len(train_val_idx)}  |  Test: {len(test_idx)}")
        print(f"{'='*65}")

        # Build fold config — unique checkpoint per fold
        fold_config = copy.deepcopy(config)
        fold_config["checkpoint_path"] = os.path.join(
            ckpt_dir, f"{ckpt_stem}_fold{fold_num}.pth"
        )

        # Inject fold split info so train_fn uses our predetermined split
        # We pass fold indices via config so the classifier can pick them up
        fold_config["_kfold_train_val_idx"] = train_val_idx.tolist()
        fold_config["_kfold_test_idx"]      = test_idx.tolist()
        fold_config["_kfold_all_samples"]   = all_samples
        fold_config["_kfold_active"]        = True

        try:
            results = train_fn(
                fold_config,
                notes=f"{notes} | fold {fold_num}/{n_splits}"
            )
            results["fold"] = fold_num
            fold_results.append(results)

            print(f"\n  Fold {fold_num} complete — test_acc: {results['test_acc']:.4f}  "
                  f"macro_f1: {results['macro_f1']:.4f}")

        except Exception as e:
            print(f"\n  ✗ Fold {fold_num} failed: {e}")
            # End any stuck MLflow run
            if mlflow.active_run():
                mlflow.end_run()
            fold_results.append({"fold": fold_num, "test_acc": None, "error": str(e)})
            continue

    # ── Aggregate ──
    valid_folds = [r for r in fold_results if r.get("test_acc") is not None]
    aggregate   = aggregate_results(valid_folds)

    print_fold_summary(valid_folds, aggregate)

    # ── Log aggregate to MLflow ──
    setup_mlflow(experiment_name)
    with mlflow.start_run(run_name=f"{arch}_kfold_aggregate"):
        mlflow.set_tag("notes", f"Aggregate of {len(valid_folds)}/{n_splits} folds — {notes}")
        mlflow.log_param("architecture",  arch)
        mlflow.log_param("n_splits",      n_splits)
        mlflow.log_param("folds_completed", len(valid_folds))
        mlflow.log_metrics({k: v for k, v in aggregate.items() if "_mean" in k})

    # ── Save per-fold CSV ──
    df = pd.DataFrame(fold_results)
    df.to_csv(results_csv, index=False)
    print(f"\n  ✓ Per-fold results saved to: {results_csv}")

    # ── Print aggregate ──
    print(f"\n── Aggregate Metrics (mean ± std across {len(valid_folds)} folds) ──")
    for key in ["test_acc", "macro_f1", "weighted_f1"]:
        mean = aggregate.get(f"{key}_mean", "N/A")
        std  = aggregate.get(f"{key}_std",  "N/A")
        print(f"  {key:20s}: {mean:.4f} ± {std:.4f}")

    return aggregate

if __name__ == "__main__":
    kfold_validate()

# ─────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────
"""
from kfold_validator import kfold_validate
from classifier_stage1 import train, DEFAULT_CONFIG, load_samples, COLLAPSE_MAP
import copy
 
if __name__ == "__main__":
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["architecture"]    = "resnet34"
    config["checkpoint_path"] = "checkpoints_stage1/resnet34_kfold.pth"
 
    aggregate = kfold_validate(
        train_fn        = train,
        config          = config,
        load_fn         = load_samples,
        label_key       = "binary",
        collapse_map    = COLLAPSE_MAP,
        experiment_name = "Stage1_KFold",
        notes           = "resnet34 baseline",
        n_splits        = 5,
    )
 
    print(f"Final: {aggregate['test_acc_mean']:.4f} ± {aggregate['test_acc_std']:.4f}")
 
 
# For Stage 2 / RBC count (no COLLAPSE_MAP), use label_key="raw":
    aggregate = kfold_validate(
        train_fn        = train,
        config          = config,
        load_fn         = load_samples,
        label_key       = "raw",         # uses training_label directly
        experiment_name = "Stage2_KFold",
        notes           = "resnet34 stage2",
        n_splits        = 5,
    )
"""