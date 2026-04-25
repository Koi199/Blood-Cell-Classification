from pathlib import Path
import csv
import json
import numpy as np


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
def save_results_list_to_csv(results, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for entry in results:
        flat = {
            "file":        entry["file"],
            "final_pred":  entry["final_pred"],
            "final_score": entry["final_score"],
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
                match pred:
                    case 0: outcome = "UNclustered Monocyte"
                    case 1: outcome = "UNclustered Monocyte RBC"
                    case _: outcome = "UNKNOWN"

            elif model_name == "Cluster_RBCCount":
                match pred:
                    case 0: outcome = "Clustered Monocyte"
                    case 1: outcome = "Clustered Monocyte RBC"
                    case 2: outcome = "RBC alone"
                    case _: outcome = "UNKNOWN"

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

    # Build CSV header
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
# Based on binary Has_RBC classification.
# Actual RBC counts come from segmentation (count_rbcs_from_segmentation).
# ─────────────────────────────────────────────
def calculate_phagocytic_index(result: dict) -> dict:
    """
    Calculate phagocytic index from count_cells() output.

    Phagocytic index = monocytes with RBC / total monocytes
    RBC counts per cell come separately from count_rbcs_from_segmentation().

    Args:
        result: dict returned by count_cells()

    Returns:
        dict with phagocytic indices and totals.
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
# Compiles RBC counts from .npy files produced
# by rbc_segmentation_pipeline.py
# ─────────────────────────────────────────────
def count_rbcs_from_segmentation(
    npy_dir: str,
    log_fn=print,
) -> dict:
    """
    Compile RBC counts from .npy segmentation files produced by
    rbc_segmentation_pipeline.py.

    Args:
        npy_dir:  Directory containing *_rbc_seg.npy files.
        log_fn:   Callable for logging.

    Returns:
        dict with keys:
            "total_rbcs"      : total RBCs detected across all cells
            "total_cells"     : number of cells segmented
            "per_cell"        : list of dicts {filename, rbc_count} per cell
            "rbc_count_dist"  : dict mapping rbc_count → number of cells with that count
    """
    npy_dir   = Path(npy_dir)
    npy_files = sorted(npy_dir.glob("*_rbc_seg.npy"))

    if not npy_files:
        log_fn(f"  No *_rbc_seg.npy files found in {npy_dir}")
        return {
            "total_rbcs":     0,
            "total_cells":    0,
            "per_cell":       [],
            "rbc_count_dist": {},
        }

    per_cell       = []
    total_rbcs     = 0
    rbc_count_dist = {}

    for npy_path in npy_files:
        try:
            data      = np.load(npy_path, allow_pickle=True).item()
            rbc_count = int(data.get("rbc_count", 0))
            filename  = data.get("filename", str(npy_path))

            per_cell.append({
                "filename":  filename,
                "rbc_count": rbc_count,
            })

            total_rbcs += rbc_count
            rbc_count_dist[rbc_count] = rbc_count_dist.get(rbc_count, 0) + 1

        except Exception as e:
            log_fn(f"  ❌ Failed to load {npy_path.name}: {e}")
            continue

    log_fn(f"\n── RBC Count Summary ({npy_dir.name}) ──")
    log_fn(f"  Cells processed : {len(per_cell)}")
    log_fn(f"  Total RBCs      : {total_rbcs}")
    log_fn(f"  Distribution    :")
    for count, n_cells in sorted(rbc_count_dist.items()):
        log_fn(f"    {count} RBC(s): {n_cells} cells")

    return {
        "total_rbcs":     total_rbcs,
        "total_cells":    len(per_cell),
        "per_cell":       per_cell,
        "rbc_count_dist": rbc_count_dist,
    }