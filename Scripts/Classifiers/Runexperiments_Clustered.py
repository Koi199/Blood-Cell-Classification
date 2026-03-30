"""
RunExperiments_RBCCount_Clustered.py
─────────────────────────────────────
Grid search over all architectures for clustered RBC count classification.
5 classes: RBC_0, RBC_1, RBC_2, RBC_3, RBC_alone

Total runs: 11 architectures
"""

import copy
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append("C:/repos/Blood-Cell-Classification/Scripts/Classifiers")
from Classifier_ClusteredRBCCount import train, DEFAULT_CONFIG

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

RESULTS_CSV = "C:/repos/Blood-Cell-Classification/grid_search_results_rbc_clustered.csv"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def save_result(row):
    df_new = pd.DataFrame([row])
    if os.path.exists(RESULTS_CSV):
        df_combined = pd.concat([pd.read_csv(RESULTS_CSV), df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(RESULTS_CSV, index=False)


# ─────────────────────────────────────────────
# MAIN GRID SEARCH
# ─────────────────────────────────────────────
if __name__ == "__main__":

    total_runs = len(ARCHITECTURES)
    completed  = 0
    failed     = []

    print(f"── Clustered RBC Count Grid Search ──")
    print(f"  Architectures: {total_runs}")
    print(f"  Classes:       RBC_0, RBC_1, RBC_2, RBC_3, RBC_alone")
    print(f"  Results CSV:   {RESULTS_CSV}")
    print()

    # Resume support
    completed_keys = set()
    if os.path.exists(RESULTS_CSV):
        df_existing    = pd.read_csv(RESULTS_CSV)
        completed_keys = set(df_existing["architecture"].tolist())
        print(f"  Resuming — {len(completed_keys)} runs already completed, skipping.\n")

    start_time = datetime.now()

    for arch in ARCHITECTURES:

        if arch in completed_keys:
            completed += 1
            continue

        print(f"\n{'='*65}")
        print(f"  Run {completed+1}/{total_runs}: {arch}")
        print(f"{'='*65}")

        config = copy.deepcopy(DEFAULT_CONFIG)
        config["architecture"]    = arch
        config["checkpoint_path"] = f"checkpoints_rbc_clustered/{arch}.pth"

        try:
            results = train(config, notes=arch)

            save_result({
                "architecture": arch,
                "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
                **results,
            })
            completed += 1

            df        = pd.read_csv(RESULTS_CSV)
            best      = df.loc[df["test_acc"].idxmax()]
            elapsed   = datetime.now() - start_time
            remaining = elapsed / completed * (total_runs - completed)
            print(f"\n  ✓ Run complete — test_acc: {results['test_acc']:.4f}")
            print(f"  Current best: {best['architecture']} → {best['test_acc']:.4f}")
            print(f"  Elapsed: {str(elapsed).split('.')[0]}  |  "
                  f"Remaining: ~{str(remaining).split('.')[0]}")

        except Exception as e:
            print(f"\n  ✗ Run failed: {arch}")
            print(f"  Error: {e}")
            failed.append({"architecture": arch, "error": str(e)})
            save_result({
                "architecture": arch,
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
            print(df_valid.nlargest(5, "test_acc")[
                ["architecture", "test_acc",
                 "rbc0_recall", "rbc1_recall", "rbc2_recall",
                 "rbc3_recall", "rbcalone_recall", "macro_f1"]
            ].to_string(index=False))

    if failed:
        print(f"\n  Failed runs:")
        for f in failed:
            print(f"    {f['architecture']}: {f['error'][:80]}")