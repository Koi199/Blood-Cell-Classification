"""
kfold_trainer_capped_oversampling.py
────────────────────────────────────────────────────────────────────────────
Re-runs donor-grouped k-fold training with ONE change from kfold_trainer.py:

    OLD: subclass_targets are FIXED per config (e.g. RBC_alone -> 400 for
         every fold), regardless of how many native examples that fold's
         training set actually contains. A fold with only ~60 native
         RBC_alone cells gets oversampled ~6-7x; a fold with ~270 native
         cells gets oversampled ~1.5x. This creates very different effective
         training diversity per fold even though the *target count* looks
         identical in the config.

    NEW: subclass_targets are computed PER FOLD, capped at
         `max_oversample_ratio` x each fold's own native count for that
         class. This keeps the desired class balance while preventing any
         single fold from relying on extreme repetition of a small number
         of unique images.

Everything else (model, training loop, evaluation, plotting) is imported
directly from kfold_trainer.py so the two runs are apples-to-apples except
for this one change.

USAGE:
    python kfold_trainer_capped_oversampling.py --config clustered_binary --max_ratio 3.0
"""

import os
import copy
import argparse
import traceback
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Reuse everything shareable from the original trainer rather than duplicating it
from K_foldValidation import (
    CONFIGS, resolve_config, get_img_size, make_transforms, build_model,
    BloodCellDataset, load_samples, train_one_epoch, evaluate,
    save_fold_plots, save_summary_plot, log_kfold_excel,
)

import sys
sys.path.append("C:/repos/Blood-Cell-Classification/Scripts/Logging")
from Logger import (
    setup_mlflow, start_run, log_params, log_epoch,
    log_results, log_artifacts, log_confusion_matrix, end_run
)


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────

NTFY_URL = "https://ntfy.sh/kyle_pipeline_done"


def notify(message: str) -> None:
    """Best-effort push notification. Never lets a notification failure
    (e.g. no network) crash or mask the actual training result."""
    try:
        requests.post(NTFY_URL, data=message.encode("utf-8"), timeout=10)
    except Exception as e:
        print(f"  [WARN] Notification failed to send: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CAPPED, PER-FOLD SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def compute_capped_targets(train_samples: list, base_targets: dict,
                            max_oversample_ratio: float) -> dict:
    """
    Compute this fold's actual sampling targets given its own native class
    counts, capped so no class is oversampled beyond max_oversample_ratio x
    its native count in THIS fold's training set.

    base_targets keys may be EITHER a raw folder name or a collapsed display
    name (see kfold_trainer.load_samples / make_weighted_sampler) — checked
    in that order, matching the fixed-target sampler's behavior.

    Returns (capped_targets, ratio_report) where ratio_report has per-key
    native/base/capped/ratio info for logging.
    """
    raw_counts     = {}
    display_counts = {}
    for _, raw_name, display_name, _, _ in train_samples:
        raw_counts[raw_name]         = raw_counts.get(raw_name, 0) + 1
        display_counts[display_name] = display_counts.get(display_name, 0) + 1

    capped_targets = {}
    ratio_report   = {}
    for name, base_target in base_targets.items():
        native = raw_counts.get(name) if name in raw_counts else display_counts.get(name, 0)

        if native == 0:
            capped_targets[name] = 0
            ratio_report[name]   = {"native": 0, "base_target": base_target,
                                     "capped_target": 0, "ratio": 0.0}
            continue

        max_allowed     = int(native * max_oversample_ratio)
        capped_target   = min(base_target, max_allowed)
        effective_ratio = capped_target / native

        capped_targets[name] = capped_target
        ratio_report[name] = {
            "native": native, "base_target": base_target,
            "capped_target": capped_target, "ratio": round(effective_ratio, 2),
        }

    return capped_targets, ratio_report


def make_capped_weighted_sampler(train_samples: list, capped_targets: dict):
    """Same mechanics as kfold_trainer.make_weighted_sampler, but takes
    pre-computed per-fold capped targets instead of a fixed config dict.
    Matches on raw folder name first, then collapsed display name."""
    raw_counts     = {}
    display_counts = {}
    for _, raw_name, display_name, _, _ in train_samples:
        raw_counts[raw_name]         = raw_counts.get(raw_name, 0) + 1
        display_counts[display_name] = display_counts.get(display_name, 0) + 1

    weights = np.zeros(len(train_samples), dtype=np.float32)
    for idx, (_, raw_name, display_name, _, _) in enumerate(train_samples):
        if raw_name in capped_targets:
            natural = raw_counts[raw_name]
            target  = capped_targets[raw_name]
        elif display_name in capped_targets:
            natural = display_counts[display_name]
            target  = capped_targets[display_name]
        else:
            natural = display_counts[display_name]
            target  = natural

        weights[idx] = target / natural if natural else 0.0

    total_samples = sum(capped_targets.values())
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights),
        num_samples=total_samples,
        replacement=True,
    )
    return sampler, total_samples


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAIN LOOP (mirrors kfold_trainer.train_kfold, capped-sampler variant)
# ─────────────────────────────────────────────────────────────────────────────

def train_kfold_capped(user_config: dict, max_oversample_ratio: float = 3.0,
                        notes: str = ""):
    cfg          = resolve_config(user_config)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch         = cfg["architecture"]
    img_size     = get_img_size(arch, cfg["img_size"])
    n_splits     = cfg["n_splits"]
    class_names  = cfg["class_names"]
    num_classes  = len(class_names)
    name         = cfg["name"]
    base_targets = cfg.get("subclass_targets", None)

    if not base_targets:
        raise ValueError(
            f"[{name}] This script only makes sense for configs with "
            f"subclass_targets defined — nothing to cap otherwise."
        )

    train_tf, val_tf = make_transforms(img_size)

    print(f"\n{'='*60}")
    print(f"  Classifier          : {name}")
    print(f"  Classes             : {class_names}")
    print(f"  Device              : {device}")
    print(f"  Arch                : {arch}  |  img_size: {img_size}")
    print(f"  Folds               : {n_splits}")
    print(f"  Max oversample ratio: {max_oversample_ratio}x  (capped, per-fold)")
    print(f"{'='*60}")

    print("\nLoading samples:")
    all_samples = load_samples(cfg["data_dir"], cfg["folder_map"], class_names)
    print(f"  Total images loaded: {len(all_samples)}")

    # Tuple layout: (path, raw_folder_name, display_name, label_int, donor_id)
    X      = np.array([s[0] for s in all_samples])
    y      = np.array([s[3] for s in all_samples])
    groups = np.array([s[4] for s in all_samples])

    sgkf               = StratifiedGroupKFold(n_splits=n_splits)
    fold_results       = []
    all_ratio_reports  = {}   # fold -> ratio_report, for the experiment log diff

    setup_mlflow(f"{name}_KFold_CappedOversampling")

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold}/{n_splits}")
        print(f"  Train donors : {np.unique(groups[train_idx]).tolist()}")
        print(f"  Test donors  : {np.unique(groups[test_idx]).tolist()}")

        train_samples = [all_samples[i] for i in train_idx]
        test_samples  = [all_samples[i] for i in test_idx]

        test_y = y[test_idx]
        for i, cname in enumerate(class_names):
            print(f"    Test — {cname}: {(test_y == i).sum()} images")

        # ── KEY CHANGE: per-fold capped targets instead of fixed config targets ──
        capped_targets, ratio_report = compute_capped_targets(
            train_samples, base_targets, max_oversample_ratio
        )
        all_ratio_reports[fold] = ratio_report

        print("\n  Capped per-fold sampling (native -> target, effective ratio):")
        for cname, info in ratio_report.items():
            flag = "  <-- capped below base target" if info["capped_target"] < info["base_target"] else ""
            print(f"    {cname:12s}: native={info['native']:4d}  "
                  f"base_target={info['base_target']:4d}  "
                  f"capped_target={info['capped_target']:4d}  "
                  f"ratio={info['ratio']:.2f}x{flag}")

        sampler, total_samples = make_capped_weighted_sampler(train_samples, capped_targets)
        print(f"    Effective samples/epoch: {total_samples}")

        train_loader = DataLoader(
            BloodCellDataset([(s[0], s[3]) for s in train_samples], transform=train_tf),
            batch_size=cfg["batch_size"], sampler=sampler,
            num_workers=cfg["num_workers"], pin_memory=True,
        )
        test_loader = DataLoader(
            BloodCellDataset([(s[0], s[3]) for s in test_samples], transform=val_tf),
            batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True,
        )

        model     = build_model(arch, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["num_epochs"]
        )

        # Distinct checkpoint suffix so this run never overwrites the fixed-target run
        checkpoint_path = os.path.join(
            cfg["checkpoint_dir"], f"{name}_capped{max_oversample_ratio}x_fold{fold}_v2.pth"
        )
        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

        run_notes = f"{notes} | capped_oversampling max={max_oversample_ratio}x" if notes \
            else f"capped_oversampling max={max_oversample_ratio}x"
        start_run(run_name=f"{name}_capped_fold{fold}", notes=run_notes)
        log_params({**cfg, "max_oversample_ratio": max_oversample_ratio,
                    "fold_capped_targets": capped_targets})

        train_losses, val_losses = [], []
        train_accs,   val_accs   = [], []
        best_acc                 = 0.0
        epochs_no_improve        = 0

        for epoch in range(1, cfg["num_epochs"] + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(test_loss)
            train_accs.append(train_acc)
            val_accs.append(test_acc)

            print(f"    Epoch {epoch:02d}/{cfg['num_epochs']} | "
                  f"Train {train_loss:.4f}/{train_acc:.4f} | "
                  f"Test  {test_loss:.4f}/{test_acc:.4f}")

            log_epoch(epoch, train_loss, train_acc, test_loss, test_acc)

            if test_acc > best_acc:
                best_acc          = test_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"      ✓ Best model saved (acc={test_acc:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg["early_stopping_patience"]:
                    print(f"      Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(torch.load(checkpoint_path))
        _, test_acc, preds, true_labels = evaluate(model, test_loader, criterion, device)

        print(f"\n── Fold {fold} Results ──")
        print(classification_report(true_labels, preds,
                                    target_names=class_names, zero_division=0))

        n_labels = list(range(num_classes))
        prec, rec, f1, _ = precision_recall_fscore_support(
            true_labels, preds, labels=n_labels, zero_division=0
        )
        _, _, f1m, _ = precision_recall_fscore_support(
            true_labels, preds, average="macro", zero_division=0
        )
        _, _, f1w, _ = precision_recall_fscore_support(
            true_labels, preds, average="weighted", zero_division=0
        )

        fold_result = {
            "fold":           fold,
            "test_donors":    np.unique(groups[test_idx]).tolist(),
            "test_acc":       round(test_acc, 4),
            "macro_f1":       round(float(f1m), 4),
            "weighted_f1":    round(float(f1w), 4),
            "capped_targets": capped_targets,
        }
        for i, cname in enumerate(class_names):
            fold_result[f"{cname}_prec"]   = round(float(prec[i]), 4)
            fold_result[f"{cname}_recall"] = round(float(rec[i]),  4)
            fold_result[f"{cname}_f1"]     = round(float(f1[i]),   4)

        fold_results.append(fold_result)

        artifact_paths = save_fold_plots(
            train_losses, val_losses, train_accs, val_accs,
            true_labels, preds, class_names, cfg["checkpoint_dir"], fold
        )
        log_results(fold_result)
        log_confusion_matrix(true_labels, preds, class_names, artifact_paths[2])
        log_artifacts(artifact_paths + [checkpoint_path])
        end_run()

    # ── Aggregate summary ──
    print(f"\n{'='*60}")
    print(f"  K-FOLD SUMMARY (capped oversampling) — {name}")
    print(f"{'='*60}")

    metric_keys = ["test_acc", "macro_f1"] + [f"{c}_f1" for c in class_names]
    summary     = {}
    for metric in metric_keys:
        values          = [r[metric] for r in fold_results]
        mean, std       = float(np.mean(values)), float(np.std(values))
        summary[metric] = (mean, std)
        print(f"  {metric:30s}: {mean:.4f} ± {std:.4f}")

    print("\n  Per-fold breakdown:")
    for r in fold_results:
        class_f1s = "  ".join(f"{c}={r[f'{c}_f1']:.3f}" for c in class_names)
        print(f"    Fold {r['fold']} {r['test_donors']}: "
              f"acc={r['test_acc']:.4f}  macro_f1={r['macro_f1']:.4f}  |  {class_f1s}")

    save_summary_plot(
        fold_results, ["test_acc", "macro_f1"] + [f"{c}_f1" for c in class_names],
        cfg["checkpoint_dir"], name
    )

    # ── Build the "key differences vs previous run" note for the experiment log ──
    diff_note   = build_diff_note(base_targets, all_ratio_reports, max_oversample_ratio)
    full_notes  = f"{notes}\n{diff_note}" if notes else diff_note

    log_kfold_excel(cfg, fold_results, summary, notes=full_notes)

    print("\n  Key differences vs previous (fixed-target) run:")
    print(diff_note)

    return fold_results, summary


def build_diff_note(base_targets: dict, all_ratio_reports: dict,
                     max_oversample_ratio: float) -> str:
    """
    Summarize, per class, how the capped-per-fold targets differed from the
    old fixed target across folds -- this is what gets written into the
    experiment log's notes column so the run is self-documenting.
    """
    lines = [f"CHANGE: capped per-fold oversampling (max {max_oversample_ratio}x native count) "
             f"replacing fixed subclass_targets used in previous runs."]

    for cname, base_target in base_targets.items():
        capped_vals = [all_ratio_reports[f][cname]["capped_target"] for f in all_ratio_reports]
        native_vals = [all_ratio_reports[f][cname]["native"] for f in all_ratio_reports]
        ratios      = [all_ratio_reports[f][cname]["ratio"] for f in all_ratio_reports]

        was_capped = any(c < base_target for c in capped_vals)
        lines.append(
            f"  {cname}: old fixed target={base_target} for every fold | "
            f"new native range=[{min(native_vals)}, {max(native_vals)}], "
            f"new target range=[{min(capped_vals)}, {max(capped_vals)}], "
            f"effective ratio range=[{min(ratios):.2f}x, {max(ratios):.2f}x]"
            + (" -- previously oversampled beyond cap in at least one fold" if was_capped else "")
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL MULTI-CONFIG RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all_configs(config_names: list, max_oversample_ratio: float,
                     notes: str = "", n_splits_override: int = None) -> dict:
    """
    Run train_kfold_capped for each config in config_names sequentially.
    Configs without subclass_targets are skipped (nothing to cap) with a
    warning rather than aborting the whole pipeline.

    Sends one ntfy notification when the full pipeline finishes (success
    or failure), plus a per-layer notification as each config completes,
    so you get progress updates on a multi-hour run instead of a single
    notification at the very end.
    """
    all_results  = {}
    skipped      = []
    failed       = {}

    print(f"\n{'#'*60}")
    print(f"  UNIVERSAL K-FOLD RERUN — {len(config_names)} config(s): {config_names}")
    print(f"  Max oversample ratio: {max_oversample_ratio}x")
    print(f"{'#'*60}")

    for config_name in config_names:
        cfg = copy.deepcopy(CONFIGS[config_name])
        if n_splits_override is not None:
            cfg["n_splits"] = n_splits_override

        if not cfg.get("subclass_targets"):
            print(f"\n[SKIP] '{config_name}' has no subclass_targets defined — "
                  f"nothing to cap, skipping.")
            skipped.append(config_name)
            continue

        try:
            print(f"\n{'='*60}\n  STARTING LAYER: {config_name}\n{'='*60}")
            results, summary = train_kfold_capped(
                cfg, max_oversample_ratio=max_oversample_ratio, notes=notes
            )
            all_results[config_name] = (results, summary)

            mean_acc = summary["test_acc"][0]
            notify(f"[{config_name}] layer complete — mean acc={mean_acc:.4f}")

        except Exception as e:
            error_str = f"{e}\n{traceback.format_exc()}"
            print(f"\n[ERROR] Layer '{config_name}' failed:\n{error_str}")
            failed[config_name] = str(e)
            notify(f"[{config_name}] layer FAILED — {e}")
            # Continue to the next config rather than aborting the whole run
            continue

    # ── Final summary notification ──
    completed = list(all_results.keys())
    summary_lines = [
        f"K-fold pipeline finished — {len(completed)}/{len(config_names)} layer(s) completed.",
    ]
    if completed:
        summary_lines.append(f"Completed: {completed}")
    if skipped:
        summary_lines.append(f"Skipped (no subclass_targets): {skipped}")
    if failed:
        summary_lines.append(f"Failed: {list(failed.keys())}")

    final_message = "\n".join(summary_lines)
    print(f"\n{'#'*60}\n{final_message}\n{'#'*60}")
    notify(final_message)

    return {"completed": all_results, "skipped": skipped, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-run k-fold training with fold-aware capped oversampling, "
                     "across one, several, or all cascade layers."
    )
    parser.add_argument(
        "--config", type=str, nargs="+", default=None,
        choices=list(CONFIGS.keys()),
        help="One or more config names to run. Omit (or pass --all) to run every "
             "config in CONFIGS that has subclass_targets defined."
    )
    parser.add_argument("--all", action="store_true",
                         help="Run every config in CONFIGS (equivalent to omitting --config).")
    parser.add_argument("--max_ratio", type=float, default=3.0,
                         help="Max oversampling ratio relative to each fold's own native count")
    parser.add_argument("--notes", type=str, default="", help="Additional run notes")
    parser.add_argument("--n_splits", type=int, default=None, help="Override number of folds")
    args = parser.parse_args()

    if args.all or args.config is None:
        config_names = list(CONFIGS.keys())
    else:
        config_names = args.config

    run_all_configs(
        config_names,
        max_oversample_ratio=args.max_ratio,
        notes=args.notes,
        n_splits_override=args.n_splits,
    )