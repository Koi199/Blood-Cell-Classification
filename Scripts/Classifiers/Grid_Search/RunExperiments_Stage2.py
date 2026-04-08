"""
RunExperiments_Stage2.py
────────────────────────
Grid search over all architectures and subclass target sizes for Stage 2.

Architectures: resnet18, resnet34, resnet50, resnet101, resnet152,
               efficientnet_b0, efficientnet_b1, efficientnet_b4,
               convnext_tiny, convnext_base, vit_b16

Target sizes: 1500, 2000, 2500, 3000
              Applied equally to Unclustered and Clustered classes

Total runs: 11 architectures × 4 target sizes = 44 runs
"""

import copy
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append("C:/repos/Blood-Cell-Classification/Scripts/Classifiers")
from Classifier_ClusteredvsNonClustered import train, DEFAULT_CONFIG

# ─────────────────────────────────────────────
# GRID DEFINITION
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

RESULTS_CSV = "C:/repos/Blood-Cell-Classification/grid_search_results_stage2.csv"


def save_result(row):
    df_new = pd.DataFrame([row])
    if os.path.exists(RESULTS_CSV):
        df_combined = pd.concat([pd.read_csv(RESULTS_CSV), df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(RESULTS_CSV, index=False)


def build_subclass_targets(target_size):
    """
    Equal balance — Unclustered total = Clustered total = target_size.
    MCwRBC and MCwoRBC split evenly within Unclustered.
    """
    return {
        "MCwRBC":    target_size // 2,
        "MCwoRBC":   target_size // 2,
        "Clustered": target_size,
    }


# ─────────────────────────────────────────────
# MAIN GRID SEARCH
# ─────────────────────────────────────────────
if __name__ == "__main__":

    total_runs = len(ARCHITECTURES) * len(TARGET_SIZES)
    completed  = 0
    failed     = []

    print(f"── Stage 2 Grid Search ──")
    print(f"  Architectures: {len(ARCHITECTURES)}")
    print(f"  Target sizes:  {TARGET_SIZES}")
    print(f"  Total runs:    {total_runs}")
    print(f"  Results CSV:   {RESULTS_CSV}")
    print()

    # Resume support
    completed_keys = set()
    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        for _, row in df_existing.iterrows():
            completed_keys.add((row["architecture"], row["target_size"]))
        print(f"  Resuming — {len(completed_keys)} runs already completed, skipping.\n")

    start_time = datetime.now()

    for arch in ARCHITECTURES:
        for target_size in TARGET_SIZES:

            if (arch, target_size) in completed_keys:
                completed += 1
                continue

            run_label = f"{arch}_t{target_size}"
            print(f"\n{'='*65}")
            print(f"  Run {completed+1}/{total_runs}: {arch}  |  target={target_size}")
            print(f"{'='*65}")

            config = copy.deepcopy(DEFAULT_CONFIG)
            config["architecture"]    = arch
            config["checkpoint_path"] = f"checkpoints_stage2/{run_label}.pth"
            config["subclass_targets"] = build_subclass_targets(target_size)

            try:
                results = train(config, notes=f"{arch} target={target_size}")

                row = {
                    "architecture": arch,
                    "target_size":  target_size,
                    "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    **results,
                }
                save_result(row)
                completed += 1

                df      = pd.read_csv(RESULTS_CSV)
                best    = df.loc[df["test_acc"].idxmax()]
                elapsed = datetime.now() - start_time
                remaining = elapsed / completed * (total_runs - completed)
                print(f"\n  ✓ Run complete — test_acc: {results['test_acc']:.4f}")
                print(f"  Current best: {best['architecture']} t={best['target_size']} "
                      f"→ {best['test_acc']:.4f}")
                print(f"  Elapsed: {str(elapsed).split('.')[0]}  |  "
                      f"Remaining: ~{str(remaining).split('.')[0]}")

            except Exception as e:
                print(f"\n  ✗ Run failed: {run_label}")
                print(f"  Error: {e}")
                failed.append({"architecture": arch, "target_size": target_size, "error": str(e)})
                save_result({
                    "architecture": arch,
                    "target_size":  target_size,
                    "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "test_acc":     None,
                    "error":        str(e),
                })
                continue

    # ── Final summary ──
    print(f"\n{'='*65}")
    print(f"  Grid search complete")
    print(f"  Completed: {completed}  |  Failed: {len(failed)}")
    print(f"  Results:   {RESULTS_CSV}")

    if os.path.exists(RESULTS_CSV):
        df       = pd.read_csv(RESULTS_CSV)
        df_valid = df[df["test_acc"].notna()].copy()
        if not df_valid.empty:
            print(f"\n  Top 5 runs by test accuracy:")
            top5 = df_valid.nlargest(5, "test_acc")[
                ["architecture", "target_size", "test_acc",
                 "clust_recall", "unclust_recall", "macro_f1"]
            ]
            print(top5.to_string(index=False))

    if failed:
        print(f"\n  Failed runs:")
        for f in failed:
            print(f"    {f['architecture']} t={f['target_size']}: {f['error'][:80]}")