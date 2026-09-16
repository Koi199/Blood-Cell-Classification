from pathlib import Path
import csv
import json
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ─────────────────────────────────────────────
# COUNT CELLS
# Binary RBC classification:
#   Unclustered_RBCCount: 0 = No_RBC, 1 = Has_RBC
#   Cluster_RBCCount:     0 = No_RBC, 1 = Has_RBC, 2 = RBC_alone (excluded)
# ─────────────────────────────────────────────
def count_cells(result: list) -> dict:
    """
    Count cells from cascade classification results.

    Returns:
        dict with keys:
            Nonmonocyte_count
            Unclustered_monocyte        (No_RBC)
            Unclustered_monocyte_hasRBC (Has_RBC)
            Clustered_monocyte          (No_RBC)
            Clustered_monocyte_hasRBC   (Has_RBC)
            Clustered_RBC_alone         (RBC_alone, excluded from index)
    """
    Nonmonocyte_count           = 0
    Unclustered_monocyte        = 0
    Unclustered_monocyte_hasRBC = 0
    Clustered_monocyte          = 0
    Clustered_monocyte_hasRBC   = 0
    Clustered_RBC_alone         = 0

    for items in result:
        path_len = len(items['path'])

        if path_len == 1:
            Nonmonocyte_count += 1

        elif path_len == 3:
            last       = items['path'][2]
            model_name = last['model']
            pred       = last['pred']

            if model_name == 'Unclustered_RBCCount':
                if pred == 0:
                    Unclustered_monocyte += 1
                elif pred == 1:
                    Unclustered_monocyte_hasRBC += 1

            elif model_name == 'Cluster_RBCCount':
                if pred == 0:
                    Clustered_monocyte += 1
                elif pred == 1:
                    Clustered_monocyte_hasRBC += 1
                elif pred == 2:
                    Clustered_RBC_alone += 1  # excluded from phagocytic index

    return {
        "Nonmonocyte_count":           Nonmonocyte_count,
        "Unclustered_monocyte":        Unclustered_monocyte,
        "Unclustered_monocyte_hasRBC": Unclustered_monocyte_hasRBC,
        "Clustered_monocyte":          Clustered_monocyte,
        "Clustered_monocyte_hasRBC":   Clustered_monocyte_hasRBC,
        "Clustered_RBC_alone":         Clustered_RBC_alone,
    }


# ─────────────────────────────────────────────
# SAVE RESULTS TO CSV
# ─────────────────────────────────────────────
def save_foldercounts_to_csv(counts: dict, csv_path, folder_name: str = None,
                        sample_name: str = None, append: bool = True) -> None:
    """
    Save the dict returned by count_cells() to CSV.

    Args:
        counts: dict returned by count_cells(result)
        csv_path: destination CSV path (str or Path)
        folder_name: top-level folder / donor label (optional)
        sample_name: subfolder / sample label (optional)
        append: if True, adds a row to an existing CSV (creating it with a
                header if it doesn't exist yet). If False, overwrites with
                a single-row CSV.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {}
    if folder_name is not None:
        row["folder"] = folder_name
    if sample_name is not None:
        row["sample"] = sample_name
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    row.update(counts)

    file_exists = csv_path.exists()
    mode = "a" if (append and file_exists) else "w"
    write_header = not (append and file_exists)

    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────
# RAM VERSION OF SAVING LIST TO CSV
# ─────────────────────────────────────────────
def save_results_list_to_csv_ram(results, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for entry in results:
        flat = {
            "parent":      entry.get("parent", ""),
            "index":       entry.get("index", -1),
            "final_pred":  entry["final_pred"],
            "final_score": entry["final_score"],
            "rbc_count":    entry.get("rbc_count", 0),
        }

        # Combined cascade confidence
        scores = [step["score"] for step in entry["path"]]
        flat["combined_score"] = float(np.prod(scores))

        # Determine final outcome label
        path_len = len(entry["path"])

        if path_len == 1:
            outcome = "NONmonocyte"

        elif path_len == 3:
            last       = entry["path"][2]
            model_name = last["model"]
            pred       = last["pred"]

            if model_name == "Unclustered_RBCCount":
                outcome = (
                    "UNclustered Monocyte" if pred == 0 else
                    "UNclustered Monocyte RBC" if pred == 1 else
                    "UNKNOWN"
                )

            elif model_name == "Cluster_RBCCount":
                outcome = (
                    "Clustered Monocyte" if pred == 0 else
                    "Clustered Monocyte RBC" if pred == 1 else
                    "RBC alone" if pred == 2 else
                    "UNKNOWN"
                )
            else:
                outcome = "UNKNOWN"

        else:
            outcome = "UNKNOWN"

        flat["class"] = outcome

        # Flatten cascade steps
        for idx, step in enumerate(entry["path"], start=1):
            flat[f"model{idx}_name"]  = step["model"]
            flat[f"model{idx}_pred"]  = step["pred"]
            flat[f"model{idx}_score"] = step["score"]
            flat[f"model{idx}_probs"] = json.dumps(step["probs"])

        rows.append(flat)

    # Build CSV header dynamically from all keys present
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ─────────────────────────────────────────────
# PHAGOCYTIC INDEX
# ─────────────────────────────────────────────
def calculate_phagocytic_index(result: dict) -> dict:
    """
    Calculate phagocytic index from count_cells() output.
    """
    Unclustered_monocytes   = result['Unclustered_monocyte'] + result['Unclustered_monocyte_hasRBC']
    Clustered_monocytes     = result['Clustered_monocyte']   + result['Clustered_monocyte_hasRBC']
    Total_monocytes         = Unclustered_monocytes + Clustered_monocytes

    Unclustered_phagocytosed = result['Unclustered_monocyte_hasRBC']
    Clustered_phagocytosed   = result['Clustered_monocyte_hasRBC']
    Total_phagocytosed       = Unclustered_phagocytosed + Clustered_phagocytosed

    phagocytic_index_Unclustered = Unclustered_phagocytosed / Unclustered_monocytes if Unclustered_monocytes > 0 else 0
    phagocytic_index_Clustered   = Clustered_phagocytosed   / Clustered_monocytes   if Clustered_monocytes   > 0 else 0
    Total_phagocytic_index       = Total_phagocytosed        / Total_monocytes       if Total_monocytes       > 0 else 0

    return {
        "Total Phagocytic Index":        round(Total_phagocytic_index,       3),
        "Unclustered Phagocytic Index":  round(phagocytic_index_Unclustered, 3),
        "Clustered Phagocytic Index":    round(phagocytic_index_Clustered,   3),
        "Total Monocytes":               Total_monocytes,
        "Total Phagocytosed Monocytes":  Total_phagocytosed,
        "Total Nonmonocytes":            result['Nonmonocyte_count'],
        "RBC Alone (excluded)":          result['Clustered_RBC_alone'],
    }


# ─────────────────────────────────────────────
# RBC COUNT FROM SEGMENTATION
# ─────────────────────────────────────────────
def count_rbcs_from_segmentation(npy_dir: str, log_fn=print) -> dict:
    npy_dir   = Path(npy_dir)
    npy_files = sorted(npy_dir.glob("*_rbc_seg.npy"))

    if not npy_files:
        log_fn(f"  No *_rbc_seg.npy files found in {npy_dir}")
        return {"total_rbcs": 0, "total_cells": 0, "per_cell": [], "rbc_count_dist": {}}

    per_cell       = []
    total_rbcs     = 0
    rbc_count_dist = {}

    for npy_path in npy_files:
        try:
            data      = np.load(npy_path, allow_pickle=True).item()
            rbc_count = int(data.get("rbc_count", 0))
            filename  = data.get("filename", str(npy_path))
            per_cell.append({"filename": filename, "rbc_count": rbc_count})
            total_rbcs += rbc_count
            rbc_count_dist[rbc_count] = rbc_count_dist.get(rbc_count, 0) + 1
        except Exception as e:
            log_fn(f"  ❌ Failed to load {npy_path.name}: {e}")
            continue

    return {
        "total_rbcs":     total_rbcs,
        "total_cells":    len(per_cell),
        "per_cell":       per_cell,
        "rbc_count_dist": rbc_count_dist,
    }


# ─────────────────────────────────────────────
# MERGE INTO RESULTS (for CSV export)
# ─────────────────────────────────────────────
def merge_rbc_counts_into_results(results: list[dict], rbc_pipeline_output: dict) -> list[dict]:
    """
    Attach a per-cell 'rbc_count' field onto each row in `results`.
    Cells not RBC-segmented get rbc_count = 0.
    """
    rbc_lookup = {}
    for c in rbc_pipeline_output.get("clustered_counts", []):
        rbc_lookup[(c["parent"], c["index"])] = c["rbc_count"]
    for c in rbc_pipeline_output.get("unclustered_counts", []):
        rbc_lookup[(c["parent"], c["index"])] = c["rbc_count"]

    for r in results:
        key = (r["parent"], r["index"])
        r["rbc_count"] = rbc_lookup.get(key, 0)

    return results


# ─────────────────────────────────────────────
# PI VARIANCE FROM CALIBRATED SOFTMAX SCORES
# ─────────────────────────────────────────────
# Class index mapping within each stage's softmax vector.
# Matches this project's CONFIGS class_names order:
#   model1 (stage1_usability):  [Unusable=0, Usable=1]
#   model2 (stage2_clustered):  [Unclustered=0, Clustered=1]
#   model3 (rbc_binary):        [No_RBC=0, Has_RBC=1]
# If CONFIGS class_names ever change order, update these.
_USABLE_IDX   = 1   # P(Usable) is index 1 in model1_probs
_HAS_RBC_IDX  = 1   # P(Has_RBC) is index 1 in model3_probs


def _parse_prob(probs_json: str, index: int) -> float:
    """Extract one probability from a JSON-encoded softmax vector string."""
    if not probs_json or (isinstance(probs_json, float) and np.isnan(probs_json)):
        return np.nan
    return json.loads(probs_json)[index]


def compute_pi_variance_from_softmax(
    predictions_csv,
    rbc_count_mean: float | None = None,
    rbc_count_var: float | None = None,
    seg_miss_rate_var: float | None = None,
    mean_detection_rate: float | None = None,
    log_fn=print,
) -> dict:
    """
    Compute the point estimate and analytical 95% CI for the Phagocytic
    Index, using calibrated per-cell softmax probabilities from predictions.csv
    as the source of classification uncertainty (Poisson-binomial variance).

    This replaces Monte Carlo subsampling with a closed-form delta-method
    approach: each cell contributes its own p_i(1-p_i) uncertainty term,
    and these are aggregated across all cells into Var(PI) directly.

    Sources of uncertainty combined here:
        1. Stage 1 (usability): is this cell truly a monocyte?   → p1_i
        2. Stage 3 (RBC-binary): does this monocyte truly have   → p3_i
           an engulfed RBC?
        Stage 2 (clustering) does NOT introduce its own variance
        term here because the RBC-binary decision routes to different
        models but the Has_RBC probability p3_i already comes from
        whichever model actually ran -- no extra uncertainty layer needed.

        3. RBC count magnitude (given Has_RBC=1): how many RBCs?
           Provided as empirical moments (rbc_count_mean, rbc_count_var)
           from either the AI's own rbc_count column (approximate) or
           a manual audit (more accurate). If not provided, defaults to
           using the AI's own rbc_count column mean/variance directly.

    Args:
        predictions_csv: path to a sample's predictions.csv
        rbc_count_mean:  mean RBC count per Has_RBC cell. If None,
                         computed from AI's rbc_count column (monocytes
                         with rbc_count > 0 only -- cells classified as
                         Has_RBC). Pass an audited value for more accuracy.
        rbc_count_var:   variance of RBC count per Has_RBC cell.
                         Same default logic as rbc_count_mean.
        log_fn:          logging callable

    Returns:
        dict with:
            point_estimate_pi: PI from the full pool using hard predictions
            N_hat: expected monocyte count (soft, from p1_i probabilities)
            X_hat: expected RBC-positive monocyte count (soft)
            var_N, var_X, cov_XN: intermediate variance components
            variance_pi, sd_pi: Var(PI) and its square root
            ci_lower, ci_upper: 95% confidence interval
    """
    csv_path = Path(predictions_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"predictions.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"model1_probs", "class", "rbc_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"predictions.csv missing required column(s): {missing}. "
            f"Columns present: {list(df.columns)}"
        )

    n_cells = len(df)
    log_fn(f"Loaded {n_cells} cells from {csv_path.name}")

    # ── p1_i: P(Usable / monocyte) for every cell ──────────────────────────
    # model1_probs is always present since stage1_usability ran on all cells.
    p1 = df["model1_probs"].apply(lambda x: _parse_prob(x, _USABLE_IDX)).to_numpy()

    # ── p3_i: P(Has_RBC) for cells that reached an RBC-binary node ─────────
    # model3_probs is only populated for cells that passed stages 1 and 2.
    # Non-monocyte cells (path_len == 1) never get a p3 -- they contribute to
    # Var(N) through p1_i but contribute 0 to X (can never be counted as
    # phagocytosed). We represent this as p3_i = 0 for those cells.
    p3 = np.zeros(n_cells)
    if "model3_probs" in df.columns:
        has_model3 = df["model3_probs"].notna()
        p3[has_model3] = df.loc[has_model3, "model3_probs"].apply(
            lambda x: _parse_prob(x, _HAS_RBC_IDX)
        ).to_numpy()

    n_with_rbc_node = int((p3 > 0).sum())
    log_fn(f"  {n_with_rbc_node} cells reached the RBC-binary classification node.")

    # ── RBC count moments: mu_c and sigma^2_c ──────────────────────────────
    # These describe the distribution of ACTUAL rbc_count per Has_RBC cell.
    # If not provided empirically (from a manual audit), fall back to the
    # AI's own rbc_count column for cells classified Has_RBC (i.e. where
    # rbc_count > 0). This is an approximation -- audited values are better.
    has_rbc_classes = {"UNclustered Monocyte RBC", "Clustered Monocyte RBC"}
    has_rbc_mask = df["class"].isin(has_rbc_classes)
    ai_rbc_counts = df.loc[has_rbc_mask, "rbc_count"].to_numpy()

    if rbc_count_mean is None:
        if len(ai_rbc_counts) == 0:
            log_fn("⚠️  No cells with rbc_count > 0 found -- using mu_c = 1.0 as fallback.")
            rbc_count_mean = 1.0
        else:
            rbc_count_mean = float(ai_rbc_counts.mean())
            log_fn(f"  rbc_count_mean not provided -- using AI column mean: {rbc_count_mean:.4f}")

    if rbc_count_var is None:
        if len(ai_rbc_counts) < 2:
            log_fn("⚠️  Fewer than 2 Has_RBC cells -- using rbc_count_var = 0.0 as fallback.")
            rbc_count_var = 0.0
        else:
            rbc_count_var = float(ai_rbc_counts.var(ddof=1))
            log_fn(f"  rbc_count_var not provided -- using AI column variance: {rbc_count_var:.4f}")

    mu_c    = rbc_count_mean
    sigma2_c = rbc_count_var

    # ── Point estimate (hard predictions, matching worker.py's MonocyteIdx) ─
    monocyte_classes = {
        "UNclustered Monocyte", "UNclustered Monocyte RBC",
        "Clustered Monocyte",   "Clustered Monocyte RBC",
    }
    monocyte_mask = df["class"].isin(monocyte_classes)
    N_hard = monocyte_mask.sum()
    X_hard = df.loc[monocyte_mask, "rbc_count"].sum()
    point_estimate_pi = float((X_hard / N_hard) * 100) if N_hard > 0 else 0.0

    # ── Soft expected counts (using probabilities) ──────────────────────────
    # E[M_i] = p1_i, so E[N] = sum(p1_i)
    N_hat = float(np.sum(p1))

    # E[X_i] = p1_i * p3_i * mu_c
    # (three independent factors: usable, has-RBC, and the count itself)
    E_Xi = p1 * p3 * mu_c
    X_hat = float(np.sum(E_Xi))

    # ── Var(N): Poisson-binomial variance of the monocyte count ─────────────
    # N = sum(M_i), M_i ~ Bernoulli(p1_i)
    # Var(N) = sum(p1_i * (1 - p1_i))
    var_N = float(np.sum(p1 * (1 - p1)))

    # ── Add segmentation miss-rate variance (empirical, from Cellpose audit) ─
    # Cellpose's variable detection rate adds uncertainty to N independently
    # of classification uncertainty. Var_seg(N) = N_hat² × σ²(miss_rate).
    # seg_miss_rate_var is the sample variance of (missed / total) across your
    # audit images. Pass None to omit this term (classification uncertainty only).
    if seg_miss_rate_var is not None:
        # r̄ = mean detection rate = 1 - mean_miss_rate
        # σ²_r = variance of detection rate from audit = variance of (1 - miss_rate) = variance of miss_rate
        # CV²_r = σ²_r / r̄²

        cv2_seg = seg_miss_rate_var / (mean_detection_rate ** 2)
        var_N_seg = (N_hat ** 2) * cv2_seg
        log_fn(f"  Var(N) from classification:  {var_N:.4f}")
        log_fn(f"  Var(N) from segmentation:    {var_N_seg:.4f}  "
               f"(seg_miss_rate_var={seg_miss_rate_var:.6f})")
        var_N = var_N + var_N_seg
        log_fn(f"  Var(N) combined:             {var_N:.4f}")

    # ── Var(X): variance of total weighted RBC count ─────────────────────────
    # X_i = M_i * R_i * C_i, M_i and R_i independent Bernoullis, C_i ~ empirical
    # Var(X_i) = E[X_i^2] - E[X_i]^2
    # E[X_i^2] = p1_i * p3_i * (sigma2_c + mu_c^2)   (using E[C_i^2] = Var + Mean^2)
    p_i = p1 * p3   # joint probability P(usable AND has_RBC)
    E_Xi_sq = p_i * (sigma2_c + mu_c ** 2)
    var_X = float(np.sum(E_Xi_sq - E_Xi ** 2))

    # ── Cov(X, N): covariance between numerator and denominator ─────────────
    # Cov(X_i, M_i) = p1_i * p3_i * mu_c * (1 - p1_i)
    # (M_i^2 = M_i for a Bernoulli; cross-term simplifies to p1*(1-p1)*p3*mu_c)
    cov_XN = float(np.sum(p1 * p3 * mu_c * (1 - p1)))

    # ── Delta-method variance of PI = 100 * X / N ───────────────────────────
    # Var(PI) ≈ 100^2 * [ Var(X)/N^2 - 2*(X/N^3)*Cov(X,N) + (X^2/N^4)*Var(N) ]
    if N_hat == 0:
        raise ValueError("N_hat is 0 -- no cells classified as monocytes with nonzero probability.")

    var_pi = (100 ** 2) * (
        var_X / N_hat ** 2
        - 2 * (X_hat / N_hat ** 3) * cov_XN
        + (X_hat ** 2 / N_hat ** 4) * var_N
    )
    var_pi = max(float(var_pi), 0.0)  # guard against tiny floating-point negatives
    sd_pi = float(np.sqrt(var_pi))

    ci_lower = point_estimate_pi - 1.96 * sd_pi
    ci_upper = point_estimate_pi + 1.96 * sd_pi

    log_fn(f"\n── PI Variance (delta-method / Poisson-binomial) ──")
    log_fn(f"  Point estimate PI (hard decisions): {point_estimate_pi:.3f}%")
    log_fn(f"  N_hat (soft monocyte count):        {N_hat:.1f}  (hard: {N_hard})")
    log_fn(f"  X_hat (soft RBC-positive count):    {X_hat:.2f}  (hard: {X_hard})")
    log_fn(f"  Var(N) = {var_N:.4f}")
    log_fn(f"  Var(X) = {var_X:.4f}")
    log_fn(f"  Cov(X,N) = {cov_XN:.4f}")
    log_fn(f"  Var(PI)  = {var_pi:.6f}   SD(PI) = {sd_pi:.4f}")
    log_fn(f"  95% CI:  [{ci_lower:.3f}%, {ci_upper:.3f}%]")

    return {
        "point_estimate_pi": point_estimate_pi,
        "N_hat":             N_hat,
        "X_hat":             X_hat,
        "var_N":             var_N,
        "var_X":             var_X,
        "cov_XN":            cov_XN,
        "variance_pi":       var_pi,
        "sd_pi":             sd_pi,
        "ci_lower":          ci_lower,
        "ci_upper":          ci_upper,
        "rbc_count_mean":    mu_c,
        "rbc_count_var":     sigma2_c,
    }


# ─────────────────────────────────────────────
# PLOT PI WITH CI
# ─────────────────────────────────────────────
def plot_pi_with_ci(
    result: dict,
    sample_name: str = "",
    save_path=None,
    dpi: int = 300,
    figsize: tuple = (6, 4),
):
    """
    Simple bar/errorbar plot of the PI point estimate with its 95% CI,
    suitable for a results figure or a panel in a multi-sample comparison.

    Args:
        result:      dict returned by compute_pi_variance_from_softmax()
        sample_name: label for the x-axis / title
        save_path:   if given, saves the figure (.png/.pdf/.svg)
        dpi:         resolution for saved raster formats
        figsize:     figure size in inches

    Returns:
        (fig, ax)
    """
    pi    = result["point_estimate_pi"]
    lower = result["ci_lower"]
    upper = result["ci_upper"]
    sd    = result["sd_pi"]

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar([sample_name or "Sample"], [pi], color="#4C72B0", alpha=0.75, width=0.4)
    ax.errorbar(
        [sample_name or "Sample"], [pi],
        yerr=[[pi - lower], [upper - pi]],
        fmt="none", color="black", capsize=8, linewidth=2,
    )

    ax.set_ylabel("Phagocytic Index (%)", fontsize=11)
    title = "PI with 95% CI (Poisson-binomial / delta-method)"
    if sample_name:
        title += f"\n{sample_name}"
    ax.set_title(title, fontsize=11, fontweight="bold")

    footnote = (
        f"PI = {pi:.2f}%  |  95% CI: [{lower:.2f}%, {upper:.2f}%]  |  SD = {sd:.3f}"
    )
    ax.text(0.5, -0.18, footnote, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color="#555555")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_pi_distributions_grid(
    results: dict[str, dict],
    save_path=None,
    dpi: int = 300,
    ncols: int = 3,
):
    """
    Plot multiple samples' PI point estimates with 95% CI error bars as a
    grid -- replaces the old Monte Carlo histogram grid.

    Args:
        results:   dict mapping sample_name -> result dict (each from
                   compute_pi_variance_from_softmax())
        save_path: if given, saves the combined grid figure
        dpi:       resolution for saved raster formats
        ncols:     number of columns in the grid

    Returns:
        (fig, axes)
    """
    names = list(results.keys())
    n = len(names)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        r = results[name]
        pi    = r["point_estimate_pi"]
        lower = r["ci_lower"]
        upper = r["ci_upper"]

        ax.bar([name], [pi], color="#4C72B0", alpha=0.75, width=0.4)
        ax.errorbar(
            [name], [pi],
            yerr=[[pi - lower], [upper - pi]],
            fmt="none", color="black", capsize=6, linewidth=1.5,
        )
        ax.set_title(f"{name}\nPI={pi:.2f}% [{lower:.2f}, {upper:.2f}]", fontsize=8)
        ax.set_ylabel("PI (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes