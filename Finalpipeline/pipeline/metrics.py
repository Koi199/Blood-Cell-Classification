from pathlib import Path
import csv
import json
import numpy as np


# Reports the count of each class
def count_cells(result: list):
    Nonmonocyte_count = 0
    Clustered_monocyte = 0
    Unclustered_monocyte = 0
    Clustered_RBC = 0
    Clustered_monocyte_oneRBC = 0
    Clustered_monocyte_twoRBCs = 0
    Clustered_monocyte_threeRBCs = 0
    Unclustered_monocyte_oneRBC = 0
    Unclustered_monocyte_twoRBCs = 0
    Unclustered_monocyte_threeRBCs = 0

    for items in result:
        if len(items['path']) == 1:
            Nonmonocyte_count += 1
        elif len(items['path']) == 3:
            if items['path'][2]['model'] == 'Unclustered_RBCCount':
                count = items['path'][2]['pred']
                match count:
                    case 0:
                        Unclustered_monocyte += 1
                    case 1:
                        Unclustered_monocyte_oneRBC += 1
                    case 2:
                        Unclustered_monocyte_twoRBCs += 1
                    case 3:
                        Unclustered_monocyte_threeRBCs += 1

            if items['path'][2]['model'] == 'Cluster_RBCCount':
                count = items['path'][2]['pred']
                match count:
                    case 0:
                        Clustered_monocyte += 1
                    case 1:
                        Clustered_monocyte_oneRBC += 1
                    case 2:
                        Clustered_monocyte_twoRBCs += 1
                    case 3:
                        Clustered_monocyte_threeRBCs += 1
                    case 4:
                        Clustered_RBC += 1

    return {
        "Nonmonocyte_count": Nonmonocyte_count,
        "Unclustered_monocyte": Unclustered_monocyte,
        "Unclustered_monocyte_oneRBC": Unclustered_monocyte_oneRBC,
        "Unclustered_monocyte_twoRBCs": Unclustered_monocyte_twoRBCs,
        "Unclustered_monocyte_threeRBCs": Unclustered_monocyte_threeRBCs,
        "Clustered_monocyte": Clustered_monocyte,
        "Clustered_monocyte_oneRBC": Clustered_monocyte_oneRBC,
        "Clustered_monocyte_twoRBCs": Clustered_monocyte_twoRBCs,
        "Clustered_monocyte_threeRBCs": Clustered_monocyte_threeRBCs,
        "Clustered_RBC": Clustered_RBC
    }


# Save a detailed record of the results to a given path
def save_results_list_to_csv(results, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for entry in results:
        flat = {
            "file": entry["file"],
            "final_pred": entry["final_pred"],
            "final_score": entry["final_score"],
        }

        # Combined cascade confidence
        scores = [step["score"] for step in entry["path"]]
        flat["combined_score"] = float(np.prod(scores))

        # ----------------------------------------------------
        # Determine final outcome label
        # ----------------------------------------------------
        path_len = len(entry["path"])

        if path_len == 1:
            outcome = "NONmonocyte"

        elif path_len == 3:
            last = entry["path"][2]
            model_name = last["model"]
            count = last["pred"]

            if model_name == "Unclustered_RBCCount":
                match count:
                    case 0: outcome = "UNclustered Monocyte"
                    case 1: outcome = "UNclustered Monocyte oneRBC"
                    case 2: outcome = "UNclustered Monocyte twoRBCs"
                    case 3: outcome = "UNclustered Monocyte threeRBCs"
                    case _: outcome = "UNKNOWN"

            elif model_name == "Cluster_RBCCount":
                match count:
                    case 0: outcome = "Clustered Monocyte"
                    case 1: outcome = "Clustered Monocyte oneRBC"
                    case 2: outcome = "Clustered Monocyte twoRBCs"
                    case 3: outcome = "Clustered Monocyte threeRBCs"
                    case 4: outcome = "Clustered RBC"
                    case _: outcome = "UNKNOWN"

            else:
                outcome = "UNKNOWN"

        else:
            outcome = "UNKNOWN"

        flat["class"] = outcome

        # ----------------------------------------------------
        # Flatten cascade steps
        # ----------------------------------------------------
        for idx, step in enumerate(entry["path"], start=1):
            flat[f"model{idx}_name"] = step["model"]
            flat[f"model{idx}_pred"] = step["pred"]
            flat[f"model{idx}_score"] = step["score"]
            flat[f"model{idx}_probs"] = json.dumps(step["probs"])  # fixed: was overwriting same key

        rows.append(flat)

    # Build CSV header
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    # Write CSV
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# Calculate phagocytic index
def calculate_phagocytic_index(result: list):
    Unclustered_monocytes = (
        result['Unclustered_monocyte'] +
        result['Unclustered_monocyte_oneRBC'] +
        result['Unclustered_monocyte_twoRBCs'] +
        result['Unclustered_monocyte_threeRBCs']
    )
    Clustered_monocytes = (
        result['Clustered_monocyte'] +
        result['Clustered_monocyte_oneRBC'] +
        result['Clustered_monocyte_twoRBCs'] +
        result['Clustered_monocyte_threeRBCs']
    )
    Total_monocytes = Unclustered_monocytes + Clustered_monocytes
    Total_nonmonocytes = result['Nonmonocyte_count'] + result['Clustered_RBC']
    Unclustered_phagocytosed = (
        result['Unclustered_monocyte_oneRBC'] +
        result['Unclustered_monocyte_twoRBCs'] +
        result['Unclustered_monocyte_threeRBCs']
    )
    Clustered_phagocytosed = (
        result['Clustered_monocyte_oneRBC'] +
        result['Clustered_monocyte_twoRBCs'] +
        result['Clustered_monocyte_threeRBCs']
    )

    # Fixed: guard against division by zero
    phagocytic_index_Unclustered = Unclustered_phagocytosed / Unclustered_monocytes if Unclustered_monocytes > 0 else 0
    phagocytic_index_Clustered = Clustered_phagocytosed / Clustered_monocytes if Clustered_monocytes > 0 else 0
    Total_phagocytic_index = (Unclustered_phagocytosed + Clustered_phagocytosed) / Total_monocytes if Total_monocytes > 0 else 0

    return {
        "Total Phagocytic Index": round(Total_phagocytic_index, 3),
        "Unclustered phagocytic Index": round(phagocytic_index_Unclustered, 3),
        "Clustered phagocytic Index": round(phagocytic_index_Clustered, 3),
        "Total Nonmonocyte": Total_nonmonocytes
    }