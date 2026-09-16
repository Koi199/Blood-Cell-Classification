from PySide6.QtCore import QObject, Signal
from pathlib import Path
import requests
from pipeline.segmentation import run_segmentation
from pipeline.npyprocessing import extract_single_cells
from pipeline.prediction import build_cascade_tree, run_classification_ram
from pipeline.metrics import count_cells, save_results_list_to_csv_ram, save_foldercounts_to_csv, merge_rbc_counts_into_results, compute_pi_variance_from_softmax, plot_pi_distributions_grid, plot_pi_with_ci
from pipeline.segmentationcount import run_full_rbc_segmentation_pipeline_ram


class PipelineWorker(QObject):
    log = Signal(str)
    label_PI = Signal(str)
    label_UnclusteredPI = Signal(str)
    label_ClusteredPI = Signal(str)
    label_ClusteredMonocyteCount = Signal(str)
    label_UnclusteredMonocyteCount = Signal(str)
    label_ClusteredRBCCount = Signal(str)
    label_UnclusteredRBCCount = Signal(str)
    label_UnclusteredPhagocyteCount = Signal(str)
    label_ClusteredPhagocyteCount = Signal(str)
    finished = Signal()
    finished_with_data = Signal(object, object, float)  # results, cell_image_lookup

    def __init__(self, image_paths, input_dir, output_dir):
        super().__init__()
        self.image_paths = image_paths
        self.input_dir = Path(input_dir) if input_dir is not None else None
        self.output_dir = Path(output_dir)

        self.npy_dir     = self.output_dir / "segmentednpy"
        self.overlay_dir = self.output_dir / "Overlay"
        self.output_cells_dir = self.output_dir / "SingleCells"
        self.model_path  = "C:/repos/Blood-Cell-Classification/Finalpipeline/model/MMAv7"


    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log.emit("Pipeline started.")
        self.log.emit(f"using {self.model_path}")

        # ── Segmentation ──
        try:
            seg_output = run_segmentation(
                image_paths      = self.image_paths,
                save_base_dir    = self.npy_dir,
                overlay_base_dir = self.overlay_dir,
                model_path       = self.model_path,
                cellprob_threshold = -1.0,
                diameter         = 40, # changed from 60
                log_fn           = self.log.emit,
            )
        except Exception as e:
            self.log.emit(f"❌ Segmentation error: {e}")
            self.finished.emit()
            return

        if seg_output["total_cells"] <= 0:
            self.log.emit("Not enough cells for counting.")
            self.log.emit("Pipeline finished.")
            self.finished.emit()
            return

        # ── Single cell extraction ──
        single_cell_BUFFER = []
        try:
            seg_files = list(self.npy_dir.glob("**/*_seg.npy"))
            if not seg_files:
                self.log.emit("No segmentation files found. Check your directory path.")
                self.finished.emit()
                return

            for idx, seg_file in enumerate(seg_files, start=1):
                self.log.emit(f"\n[{idx}/{len(seg_files)}] Processing: {seg_file.name}")
                self.log.emit(f"Path: {seg_file}\n")
                try:
                    cells = extract_single_cells(
                        seg_file=seg_file,
                        output_dir=None,
                        log_fn=self.log.emit,
                    )
                    single_cell_BUFFER.extend(cells)
                except Exception as e:
                    self.log.emit(f"ERROR processing {seg_file.name}: {str(e)}\n")

            self.log.emit(f"Buffered {len(single_cell_BUFFER)} cells into RAM")

        except Exception as e:
            self.log.emit(f"❌ Single Cell Extraction error: {e}")
            self.finished.emit()
            return

        if not single_cell_BUFFER:
            self.log.emit("No cells extracted — aborting classification.")
            self.finished.emit()
            return

        # ── Classification ──
        self.log.emit("\nBuilding classifier (loading weights)... ")
        tree = build_cascade_tree(device="cuda")

        try:
            self.log.emit(f"Classifying {len(single_cell_BUFFER)} cells from RAM")
            results = run_classification_ram(
                cells=single_cell_BUFFER,
                tree=tree,
                log_fn=self.log.emit,
            )
            self.classified_results = results
            self.cell_image_lookup = {
                (c["parent"], c["index"]): c["image"]
                for c in single_cell_BUFFER
            }
        except Exception as e:
            self.log.emit(f"ERROR during classification: {str(e)}")
            self.finished.emit()
            return

        # ── RBC segmentation (RAM, single call) ──
        clustered_total_rbcs = 0
        unclustered_total_rbcs = 0
        try:
            # Run RBC segmentation on all cells in RAM, all files and folders at once, and merge the counts into the results
            rbc_results = run_full_rbc_segmentation_pipeline_ram(
                results=results,
                cells=single_cell_BUFFER,
                flow_threshold=0.6,
                cellprob_threshold=-1.5,
                model_path="C:/repos/Blood-Cell-Classification/Finalpipeline/model/RBCCountSegmentationv2",
                log_fn=self.log.emit,
            )

            clustered_total_rbcs   = sum(c["rbc_count"] for c in rbc_results["clustered_counts"])
            unclustered_total_rbcs = sum(c["rbc_count"] for c in rbc_results["unclustered_counts"])
            total_rbcs             = rbc_results["total_rbcs"]

            self.log.emit(f"Clustered RBCs: {clustered_total_rbcs}")
            self.log.emit(f"Unclustered RBCs: {unclustered_total_rbcs}")
            self.log.emit(f"Total RBCs: {total_rbcs}")

            results = merge_rbc_counts_into_results(results, rbc_results)

        except Exception as e:
            self.log.emit(f"❌ Counting ERROR: {e}")
            clustered_total_rbcs = 0
            unclustered_total_rbcs = 0
            total_rbcs = 0
            # ensure every row still has the column even if segmentation failed,
            # so predictions.csv doesn't end up with a missing column
            for r in results:
                r.setdefault("rbc_count", 0)

        # ── Phagocytic index ──
        try:
            confident_results = [r for r in results if not r.get("low_confidence")]
            counts = count_cells(confident_results)
            counts["Clustered_total_RBCs"]   = clustered_total_rbcs
            counts["Unclustered_total_RBCs"] = unclustered_total_rbcs
            save_foldercounts_to_csv(
                counts,
                csv_path=self.output_dir.parent.parent / "batch_cell_counts.csv",  # one shared file above all donor folders
                folder_name=getattr(self, "_donor_name", self.output_dir.parent.name),
                sample_name=self.output_dir.name, append=True
            )
            self.log.emit(f"counts: {counts}")

            u_mono   = counts.get("Unclustered_monocyte", 0)
            u_mono_r = counts.get("Unclustered_monocyte_hasRBC", 0)
            c_mono   = counts.get("Clustered_monocyte", 0)
            c_mono_r = counts.get("Clustered_monocyte_hasRBC", 0)

            total_mono = u_mono + u_mono_r + c_mono + c_mono_r
            if total_mono == 0:
                raise ValueError("No monocytes detected — cannot compute PI")

            MonocyteIdx = (total_rbcs / total_mono) * 100

            denom_u = u_mono + u_mono_r
            denom_c = c_mono + c_mono_r

            Unclustered_PI = (unclustered_total_rbcs * 100 / denom_u) if denom_u else 0
            Clustered_PI   = (clustered_total_rbcs   * 100 / denom_c) if denom_c else 0

            self.label_PI.emit(f"{float(MonocyteIdx):.4f}")
            self.label_UnclusteredPI.emit(f"{float(Unclustered_PI):.4f}")
            self.label_ClusteredPI.emit(f"{float(Clustered_PI):.4f}")

            self.label_ClusteredMonocyteCount.emit(f"{c_mono + c_mono_r}")
            self.label_UnclusteredMonocyteCount.emit(f"{u_mono + u_mono_r}")

            self.label_ClusteredRBCCount.emit(f"{clustered_total_rbcs}")
            self.label_UnclusteredRBCCount.emit(f"{unclustered_total_rbcs}")

            self.label_ClusteredPhagocyteCount.emit(f"{c_mono_r}")
            self.label_UnclusteredPhagocyteCount.emit(f"{u_mono_r}")

        except Exception as e:
            self.log.emit(f"ERROR Calculating Index: {e}")
            counts = {}
            MonocyteIdx = 0

        # ── Cell count summary ──
        try:
            text = "\n".join(f"{key}: {value}" for key, value in counts.items())
            self.log.emit(text)
        except Exception:
            self.log.emit("ERROR counting cells.\n")

        # ── Save CSV ──
        try:
            
            csv_path = self.output_dir / "predictions.csv"
            save_results_list_to_csv_ram(results, csv_path=csv_path)
        except Exception as e:
            self.log.emit(f"ERROR saving results: {e}")

        # == Compute PI subsampling distribution ==
        try:
            variance_result = compute_pi_variance_from_softmax(
            predictions_csv=self.output_dir / "predictions.csv",
            log_fn=self.log.emit,
            seg_miss_rate_var=0.0067,  # example value, adjust as needed
            mean_detection_rate=0.997  # example value, adjust as needed
            )
            self.log.emit(
                f"PI = {variance_result['point_estimate_pi']:.2f}%  "
                f"95% CI: [{variance_result['ci_lower']:.2f}%, {variance_result['ci_upper']:.2f}%]"

            )
            plot_pi_with_ci(
                result=variance_result,
                save_path=self.output_dir / "pi_distribution.png",
            )
            plot_pi_distributions_grid(
                results=variance_result,
                save_path=self.note / "pi_distribution_grid.png",
            )

        except Exception as e:
            self.log.emit(f"ERROR computing PI subsampling distribution: {e}")

        # emit data for GUI post-processing
        self.finished_with_data.emit(self.classified_results, self.cell_image_lookup, MonocyteIdx)
        self.log.emit("Emitted results and cell images to GUI.")

        self.log.emit("Pipeline finished.")

        requests.post(
            "https://ntfy.sh/kyle_pipeline_done",
            data="Pipeline run completed successfully.".encode("utf-8")
        )

        self.finished.emit()
