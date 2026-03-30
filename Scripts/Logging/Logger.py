import os
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────
# TRACKING URI
# MLflow stores all runs under mlruns/ at this path.
# Change to wherever your project lives.
# ─────────────────────────────────────────────
TRACKING_URI = "C:/repos/Blood-Cell-Classification/mlruns"


def setup_mlflow(experiment_name):
    """
    Call once at the start of each training script.
    Sets the tracking URI and experiment name.

    Usage:
        from logger import setup_mlflow, log_params, log_epoch, log_results, log_artifacts, end_run
        setup_mlflow("Stage1_MCvsNonMC")
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    print(f"  ✓ MLflow tracking: {TRACKING_URI}")
    print(f"  ✓ Experiment: {experiment_name}")


def start_run(run_name=None, notes=""):
    """
    Start a new MLflow run. Call before training loop.
    Automatically ends any lingering active run first so grid searches
    don't get blocked by a previous failed run.

    Usage:
        start_run(run_name="baseline", notes="3-class subclass split")
    """
    # Safety cleanup — end any run left open by a previous crash
    if mlflow.active_run() is not None:
        print(f"  ⚠ Lingering MLflow run detected — ending it before starting new run")
        mlflow.end_run()

    run = mlflow.start_run(run_name=run_name)
    if notes:
        mlflow.set_tag("notes", notes)
    print(f"  ✓ MLflow run started: {run.info.run_id[:8]}...")
    return run


def log_params(config):
    """
    Log all relevant CONFIG fields as MLflow params.
    Handles both subclass_targets dict and simple values.

    Usage:
        log_params(CONFIG)
    """
    params = {
        "lr":                    config.get("lr"),
        "weight_decay":          config.get("weight_decay"),
        "batch_size":            config.get("batch_size"),
        "num_epochs":            config.get("num_epochs"),
        "early_stopping":        config.get("early_stopping_patience"),
        "threshold":             config.get("classification_threshold"),
        "img_size":              config.get("img_size"),
        "class_weights":         str(config.get("class_weights", "none")),
        "subclass_targets":      str(config.get("subclass_targets", "none")),
        "binary_targets":        str(config.get("binary_targets", "none")),
        "optimise_metric":       config.get("optimise_metric", "val_acc"),
    }
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    mlflow.log_params(params)


def log_epoch(epoch, train_loss, train_acc, val_loss, val_acc):
    """
    Log per-epoch metrics. Call inside the training loop.

    Usage:
        log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
    """
    mlflow.log_metrics({
        "train_loss": round(train_loss, 4),
        "train_acc":  round(train_acc,  4),
        "val_loss":   round(val_loss,   4),
        "val_acc":    round(val_acc,    4),
    }, step=epoch)


def log_results(results):
    """
    Log final test set metrics after training completes.
    Accepts the same results dict used by log_experiment().

    Usage:
        log_results(results)
    """
    # Log whatever keys exist — works for Stage 1, Stage 2, and 3-class
    metric_keys = [
        "test_acc", "val_acc",
        "non_mono_prec", "non_mono_recall", "non_mono_f1",
        "mono_prec",     "mono_recall",     "mono_f1",
        "unclust_prec",  "unclust_recall",  "unclust_f1",
        "clust_prec",    "clust_recall",    "clust_f1",
        "nonmono_prec",  "nonmono_recall",  "nonmono_f1",
        "macro_f1",      "weighted_f1",
        "tp", "tn", "fp", "fn",
    ]
    metrics = {k: results[k] for k in metric_keys if k in results}
    mlflow.log_metrics(metrics)


def log_artifacts(artifact_paths):
    """
    Log a list of file paths as MLflow artifacts (plots, checkpoints, etc.)
    Skips any paths that don't exist yet.

    Usage:
        log_artifacts([
            "checkpoints/loss_curve.png",
            "checkpoints/confusion_matrix.png",
            "checkpoints/resnet34_binary.pth",
        ])
    """
    for path in artifact_paths:
        if os.path.exists(path):
            mlflow.log_artifact(path)
            print(f"  ✓ Artifact logged: {path}")
        else:
            print(f"  ✗ Artifact not found, skipping: {path}")


def log_confusion_matrix(true_labels, preds, class_names, output_path):
    """
    Save confusion matrix as raw counts (not %), log as MLflow artifact.

    Usage:
        log_confusion_matrix(
            true_labels, preds,
            class_names=["Non-Monocyte", "Monocyte"],
            output_path="checkpoints/confusion_matrix.png"
        )
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",   # fmt="d" = raw integers
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix (counts)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path}")

    mlflow.log_artifact(output_path)


def end_run():
    """
    End the active MLflow run. Call after all logging is done.

    Usage:
        end_run()
    """
    mlflow.end_run()
    print("  ✓ MLflow run ended")