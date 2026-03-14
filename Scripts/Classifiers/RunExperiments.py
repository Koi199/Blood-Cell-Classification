"""
RunExperiments.py
─────────────────
Grid search over all architectures and subclass target sizes.

Architectures: resnet18, resnet34, resnet50, resnet101, resnet152,
               efficientnet_b0, efficientnet_b1, efficientnet_b4,
               convnext_tiny, convnext_base, vit_b16

Target sizes: 1500 → 3000 in steps of 500
              Applied to all NonMono subclasses and Monocyte subtypes equally

Total runs: 11 architectures × 4 target steps = 44 runs
"""

import copy
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append("C:/repos/Blood-Cell-Classification/Scripts/Classifiers")
from Classifier_MCvsNonMC import train, DEFAULT_CONFIG

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

# Target sizes from 1500 to 3000 in steps of 500
TARGET_SIZES = list(range(1500, 3001, 500))  # [1500, 2000, 2500, 3000]

# ─────────────────────────────────────────────
# RESULTS TRACKER
# Saved incrementally so you don't lose results if a run crashes
# ─────────────────────────────────────────────
RESULTS_CSV = "C:/repos/Blood-Cell-Classification/grid_search_results.csv"


def save_result(row):
    """Append a single result row to the CSV immediately after each run."""
    df_new = pd.DataFrame([row])
    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(RESULTS_CSV, index=False)


def build_subclass_targets(target_size):
    """
    Build subclass targets for a given target size.

    Lymphocyte: capped at 150 (only 156 clean samples, collapsed into Unusable)
    RBCalone:   capped at 400 (only 404 clean samples)
    Unusable:   scales with target_size
    Monocyte subtypes: equal split, each = target_size // 3
    """
    mono_per_subtype = target_size // 3
    return {
        "Unusable":   target_size,
        "Lymphocyte": 150,                                  # capped — 156 samples available
        "RBCalone":   400,                                  # capped — 404 samples available
        "MCwRBC":     mono_per_subtype,
        "MCwoRBC":    mono_per_subtype,
        "Clustered":  target_size - 2 * mono_per_subtype,  # absorbs rounding remainder
    }


# ─────────────────────────────────────────────
# MAIN GRID SEARCH
# ─────────────────────────────────────────────
if __name__ == "__main__":

    total_runs   = len(ARCHITECTURES) * len(TARGET_SIZES)
    completed    = 0
    failed       = []

    print(f"── Grid Search ──")
    print(f"  Architectures: {len(ARCHITECTURES)}")
    print(f"  Target sizes:  {len(TARGET_SIZES)}  ({TARGET_SIZES[0]} → {TARGET_SIZES[-1]})")
    print(f"  Total runs:    {total_runs}")
    print(f"  Results CSV:   {RESULTS_CSV}")
    print()

    # Resume support — skip already completed runs
    completed_keys = set()
    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        for _, row in df_existing.iterrows():
            completed_keys.add((row["architecture"], row["target_size"]))
        print(f"  Resuming — {len(completed_keys)} runs already completed, skipping.\n")

    start_time = datetime.now()

    for arch in ARCHITECTURES:
        for target_size in TARGET_SIZES:

            # Skip if already done
            if (arch, target_size) in completed_keys:
                completed += 1
                continue

            run_label = f"{arch}_t{target_size}"
            print(f"\n{'='*65}")
            print(f"  Run {completed+1}/{total_runs}: {arch}  |  target={target_size}")
            print(f"{'='*65}")

            config = copy.deepcopy(DEFAULT_CONFIG)
            config["architecture"]    = arch
            config["checkpoint_path"] = f"checkpoints_stage1/{run_label}.pth"
            config["subclass_targets"] = build_subclass_targets(target_size)

            try:
                results = train(config, notes=f"{arch} target={target_size}")

                # Record result
                row = {
                    "architecture":    arch,
                    "target_size":     target_size,
                    "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M"),
                    **results,          # unpacks all metrics directly
                }
                save_result(row)
                completed += 1

                # Print running best
                df = pd.read_csv(RESULTS_CSV)
                best = df.loc[df["test_acc"].idxmax()]
                elapsed = datetime.now() - start_time
                print(f"\n  ✓ Run complete — test_acc: {results['test_acc']:.4f}")
                print(f"  Current best: {best['architecture']} t={best['target_size']} "
                      f"→ {best['test_acc']:.4f}")
                print(f"  Elapsed: {str(elapsed).split('.')[0]}  |  "
                      f"Remaining: ~{(elapsed / completed * (total_runs - completed))}")

            except Exception as e:
                print(f"\n  ✗ Run failed: {run_label}")
                print(f"  Error: {e}")
                failed.append({"architecture": arch, "target_size": target_size, "error": str(e)})
                # Save failure to CSV so we know what to investigate
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
    print(f"  Total runs:  {total_runs}")
    print(f"  Completed:   {completed}")
    print(f"  Failed:      {len(failed)}")
    print(f"  Results:     {RESULTS_CSV}")

    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        df_valid = df[df["test_acc"].notna()].copy()
        if not df_valid.empty:
            print(f"\n  Top 5 runs by test accuracy:")
            top5 = df_valid.nlargest(5, "test_acc")[
                ["architecture", "target_size", "test_acc",
                 "mono_recall", "non_mono_recall", "macro_f1"]
            ]
            print(top5.to_string(index=False))

    if failed:
        print(f"\n  Failed runs:")
        for f in failed:
            print(f"    {f['architecture']} t={f['target_size']}: {f['error'][:80]}")