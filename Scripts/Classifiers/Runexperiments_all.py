"""
RunExperiments_All.py
─────────────────────
Unified grid search over all architectures and target sizes
for Stage 1, Stage 2, and 3-class classifiers.

Each classifier runs independently with its own:
  - Results CSV
  - Checkpoint folder
  - MLflow experiment

Configure which classifiers to run via RUN_STAGE1, RUN_STAGE2, RUN_3CLASS flags.
"""

import copy
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append("C:/repos/Blood-Cell-Classification/Scripts/Classifiers")
from Classifier_MCvsNonMC import train as train_stage1, DEFAULT_CONFIG as CONFIG_STAGE1
from Classifier_ClusteredvsNonClustered import train as train_stage2, DEFAULT_CONFIG as CONFIG_STAGE2
from Classifier_ClusteredvsNonClusteredvsNonmono import train as train_3class,  DEFAULT_CONFIG as CONFIG_3CLASS

# ─────────────────────────────────────────────
# TOGGLE — which classifiers to run
# ─────────────────────────────────────────────
RUN_STAGE1 = True
RUN_STAGE2 = True
RUN_3CLASS = True

# ─────────────────────────────────────────────
# GRID DEFINITION — shared across all classifiers
# ─────────────────────────────────────────────
ARCHITECTURES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b4",
    "convnext_tiny",
    "convnext_base",
    "vit_b16",
]

TARGET_SIZES = list(range(1500, 3001, 500))  # [1500, 2000, 2500, 3000]

# ─────────────────────────────────────────────
# RESULTS CSV PATHS
# ─────────────────────────────────────────────
RESULTS_STAGE1 = "C:/repos/Blood-Cell-Classification/grid_search_results_stage1.csv"
RESULTS_STAGE2 = "C:/repos/Blood-Cell-Classification/grid_search_results_stage2.csv"
RESULTS_3CLASS = "C:/repos/Blood-Cell-Classification/grid_search_results_3class.csv"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def save_result(results_csv, row):
    df_new = pd.DataFrame([row])
    if os.path.exists(results_csv):
        df_combined = pd.concat([pd.read_csv(results_csv), df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(results_csv, index=False)


def load_completed(results_csv):
    if not os.path.exists(results_csv):
        return set()
    df = pd.read_csv(results_csv)
    return {(row["architecture"], row["target_size"]) for _, row in df.iterrows()}


def print_summary(results_csv, metric_cols):
    if not os.path.exists(results_csv):
        return
    df       = pd.read_csv(results_csv)
    df_valid = df[df["test_acc"].notna()].copy()
    if df_valid.empty:
        return
    print(f"\n  Top 5 by test accuracy:")
    print(df_valid.nlargest(5, "test_acc")[
        ["architecture", "target_size", "test_acc"] + metric_cols
    ].to_string(index=False))


def build_stage1_targets(target_size):
    mono_per = target_size // 3
    return {
        "Unusable":   target_size,
        "Lymphocyte": 150,   # capped — 156 samples available
        "RBCalone":   400,   # capped — 404 samples available
        "MCwRBC":     mono_per,
        "MCwoRBC":    mono_per,
        "Clustered":  target_size - 2 * mono_per,
    }


def build_stage2_targets(target_size):
    return {
        "MCwRBC":    target_size // 2,
        "MCwoRBC":   target_size // 2,
        "Clustered": target_size,
    }


def build_3class_targets(target_size):
    mono_per = target_size // 2
    return {
        "Unusable":   target_size,
        "Lymphocyte": 150,   # capped
        "RBCalone":   400,   # capped
        "MCwRBC":     mono_per,
        "MCwoRBC":    mono_per,
        "Clustered":  target_size,
    }


def run_grid(classifier_name, train_fn, base_config, results_csv,
             build_targets_fn, checkpoint_prefix, metric_cols, total_runs):
    """
    Generic grid search loop — shared by all three classifiers.
    """
    print(f"\n{'#'*65}")
    print(f"  {classifier_name} Grid Search")
    print(f"  Total runs: {total_runs}")
    print(f"  Results:    {results_csv}")
    print(f"{'#'*65}")

    completed_keys = load_completed(results_csv)
    if completed_keys:
        print(f"  Resuming — {len(completed_keys)} runs already done, skipping.\n")

    completed  = len(completed_keys)
    failed     = []
    start_time = datetime.now()

    for arch in ARCHITECTURES:
        for target_size in TARGET_SIZES:

            if (arch, target_size) in completed_keys:
                continue

            run_label = f"{arch}_t{target_size}"
            print(f"\n{'='*65}")
            print(f"  [{classifier_name}] Run {completed+1}/{total_runs}: "
                  f"{arch}  |  target={target_size}")
            print(f"{'='*65}")

            config = copy.deepcopy(base_config)
            config["architecture"]    = arch
            config["checkpoint_path"] = f"{checkpoint_prefix}/{run_label}.pth"
            config["subclass_targets"] = build_targets_fn(target_size)

            try:
                results = train_fn(config, notes=f"{arch} target={target_size}")

                save_result(results_csv, {
                    "architecture": arch,
                    "target_size":  target_size,
                    "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    **results,
                })
                completed += 1

                df      = pd.read_csv(results_csv)
                best    = df.loc[df["test_acc"].idxmax()]
                elapsed = datetime.now() - start_time
                remaining = elapsed / completed * (total_runs - completed) if completed > 0 else "N/A"
                print(f"\n  ✓ test_acc: {results['test_acc']:.4f}")
                print(f"  Best so far: {best['architecture']} t={best['target_size']} "
                      f"→ {best['test_acc']:.4f}")
                print(f"  Elapsed: {str(elapsed).split('.')[0]}  |  "
                      f"Remaining: ~{str(remaining).split('.')[0]}")

            except Exception as e:
                print(f"\n  ✗ Failed: {run_label} — {e}")
                failed.append({"architecture": arch, "target_size": target_size, "error": str(e)})
                save_result(results_csv, {
                    "architecture": arch,
                    "target_size":  target_size,
                    "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "test_acc":     None,
                    "error":        str(e),
                })
                continue

    print(f"\n  {classifier_name} complete — "
          f"Completed: {completed}  |  Failed: {len(failed)}")
    print_summary(results_csv, metric_cols)

    if failed:
        print(f"\n  Failed runs:")
        for f in failed:
            print(f"    {f['architecture']} t={f['target_size']}: {f['error'][:80]}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":

    total_runs = len(ARCHITECTURES) * len(TARGET_SIZES)

    overall_start = datetime.now()

    if RUN_STAGE1:
        run_grid(
            classifier_name   = "Stage 1 — NonMono vs Monocyte",
            train_fn          = train_stage1,
            base_config       = CONFIG_STAGE1,
            results_csv       = RESULTS_STAGE1,
            build_targets_fn  = build_stage1_targets,
            checkpoint_prefix = "checkpoints_stage1",
            metric_cols       = ["mono_recall", "non_mono_recall", "macro_f1"],
            total_runs        = total_runs,
        )

    if RUN_STAGE2:
        run_grid(
            classifier_name   = "Stage 2 — Clustered vs Unclustered",
            train_fn          = train_stage2,
            base_config       = CONFIG_STAGE2,
            results_csv       = RESULTS_STAGE2,
            build_targets_fn  = build_stage2_targets,
            checkpoint_prefix = "checkpoints_stage2",
            metric_cols       = ["clust_recall", "unclust_recall", "macro_f1"],
            total_runs        = total_runs,
        )

    if RUN_3CLASS:
        run_grid(
            classifier_name   = "3-Class — Unclustered / Clustered / NonMono",
            train_fn          = train_3class,
            base_config       = CONFIG_3CLASS,
            results_csv       = RESULTS_3CLASS,
            build_targets_fn  = build_3class_targets,
            checkpoint_prefix = "checkpoints_3class",
            metric_cols       = ["clust_recall", "unclust_recall", "nonmono_recall"],
            total_runs        = total_runs,
        )

    elapsed = datetime.now() - overall_start
    print(f"\n{'#'*65}")
    print(f"  All grid searches complete")
    print(f"  Total elapsed: {str(elapsed).split('.')[0]}")
    print(f"  Stage 1 results: {RESULTS_STAGE1}")
    print(f"  Stage 2 results: {RESULTS_STAGE2}")
    print(f"  3-Class results: {RESULTS_3CLASS}")
    print(f"{'#'*65}")