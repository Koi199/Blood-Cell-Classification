from PySide6.QtCore import QObject, Signal
from pipeline.segmentation import run_segmentation
from pipeline.npyprocessing import extract_single_cells
from pipeline.prediction import build_cascade_tree, run_classification
from pipeline.metrics import count_cells, save_results_list_to_csv, calculate_phagocytic_index
from pipeline.segmentationcount import run_full_rbc_segmentation_pipeline
from pathlib import Path

class PipelineWorker(QObject):
    log = Signal(str)
    label_PI = Signal(str)
    label_UnclusteredI = Signal(str)
    label_ClusteredI = Signal(str)
    finished = Signal()

    def __init__(self, image_paths, root_dir):
        super().__init__()
        self.image_paths = image_paths
        # Always store root_dir as a Path
        self.root_dir = Path(root_dir)

        # Build all subpaths here
        self.npy_dir     = self.root_dir / "segmentednpy"
        self.overlay_dir = self.root_dir / "Overlay"
        self.output_dir  = self.root_dir / "SingleCells"
        self.model_path  = "C:/repos/Blood-Cell-Classification/Finalpipeline/model/MMATrainv5"

    def run(self):
        self.log.emit("Pipeline started.")

        # Cellpose Segmentation Loop
        self.log.emit(f"using {self.model_path}")
        try:
            seg_output = run_segmentation(
                image_paths      = self.image_paths,
                save_base_dir    = self.npy_dir,
                overlay_base_dir = self.overlay_dir,
                model_path       = self.model_path,
                diameter         = 70, # pixels
                log_fn           = self.log.emit,   # ← routes all logs to the UI
            )

        except Exception as e:
            self.log.emit(f"❌ Segmentation error: {e}")

        if seg_output['total_cells'] > 100: # need at least 100 cells for inference

            # Single Cell npy processing
            self.npy_dir = Path(self.npy_dir)
            self.output_dir = Path(self.output_dir)
            try:
                seg_files = list(self.npy_dir.glob("**/*_seg.npy"))
                if len(seg_files) == 0:
                    self.log.emit("No segmentation files found. Check your directory path.")
                else:

                    # Process each segmentation file
                    for idx, seg_file in enumerate(seg_files, start=1):
                        self.log.emit(f"\n[{idx}/{len(seg_files)}] Processing: {seg_file.name}")
                        self.log.emit(f"Path: {seg_file}\n")
                        
                        try:                        
                            # Create output directory with slide grouping
                            output_dir = self.output_dir
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            extract_single_cells(
                                seg_file=seg_file,
                                output_dir=output_dir, 
                                log_fn = self.log.emit
                            )
                        except Exception as e:
                            self.log.emit(f"ERROR processing {seg_file.name}: {str(e)}\n")
                
                self.log.emit("Single cells saved!")
            except Exception as e:
                self.log.emit(f"❌ Single Cell Extraction error: {e}")

            # Prediction
            self.log.emit("\nBuilding classifier (loading weights)... ")
            tree = build_cascade_tree(device="cuda")

            try:
                cell_files = list(self.output_dir.glob("*"))
                self.log.emit(f"Classifying {len(cell_files)} crops")
                if len(cell_files) == 0:
                    self.log.emit("No cell files found. Check your directory path.")
                else:
                    try:
                        results = run_classification(
                            image_paths = cell_files,
                            tree        = tree,
                            log_fn      = self.log.emit
                        )
                    except Exception as e:
                        self.log.emit(f"ERROR processing files: {str(e)}\n")

            except Exception as e:
                self.log.emit(f"❌ Prediction error: {e}")

            # Count segmentation
            try:
                count_results = run_full_rbc_segmentation_pipeline(
                    results = results, 
                    clustered_save_dir = self.root_dir / "clustered", 
                    unclustered_save_dir = self.root_dir / "unclustered",
                    model_path = "C:/repos/Blood-Cell-Classification/Finalpipeline/model/RBCCountSegmentation",
                    log_fn      = self.log.emit
                )
                self.log.emit(f"Clustered RBCs: {count_results['clustered_counts']['total_rbcs']}")
                self.log.emit(f"Unclustered RBCs: {count_results['unclustered_counts']['total_rbcs']}")
                self.log.emit(f"Total RBCs: {count_results['total_rbcs']}")

            except Exception as e:
                self.log.emit(f"❌ Counting ERROR: {e}")

            # Calculate phagocytic index
            try:
                confident_results = [r for r in results if not r.get("low_confidence")]
                counts = count_cells(confident_results)
                self.log.emit(f"counts: {counts}")

                u_mono   = counts.get('Unclustered_monocyte', 0)
                u_mono_r = counts.get('Unclustered_monocyte_hasRBC', 0)
                c_mono   = counts.get('Clustered_monocyte', 0)
                c_mono_r = counts.get('Clustered_monocyte_hasRBC', 0)

                total_mono = u_mono + u_mono_r + c_mono + c_mono_r

                if total_mono == 0:
                    raise ValueError("No monocytes detected — cannot compute PI")

                MonocyteIdx = count_results['total_rbcs'] / total_mono

                denom_u = u_mono + u_mono_r
                denom_c = c_mono + c_mono_r

                Unclustered_PI = count_results['unclustered_counts']['total_rbcs'] / denom_u if denom_u else 0
                Clustered_PI   = count_results['clustered_counts']['total_rbcs'] / denom_c if denom_c else 0

                self.label_PI.emit(f"{MonocyteIdx:.4f}")
                self.label_UnclusteredI.emit(f"{Unclustered_PI:.4f}")
                self.label_ClusteredI.emit(f"{Clustered_PI:.4f}")

            except Exception as e:
                self.log.emit(f"ERROR Calculating Index: {e}")



            # Display cell count
            try:
                text = "\n".join(f"{key}: {value}" for key, value in counts.items())
                self.log.emit(text)
                
            except Exception as e:
                self.log.emit(f"ERROR counting cells.\n")

            # Save results in csv format
            try:
                csv_path = self.root_dir / "predictions.csv"
                save_results_list_to_csv(results,csv_path=csv_path )
            except Exception as e:
                self.log.emit(f"ERROR saving results.\n")

        else:
            self.log.emit("Not enough cells for counting.")

        self.log.emit("Pipeline finished.")
        self.finished.emit()
