from PySide6.QtCore import QObject, Signal
from pipeline.segmentation import run_segmentation
from pipeline.npyprocessing import extract_single_cells
from pipeline.prediction import build_cascade_tree, run_classification, cellcount
from pathlib import Path

class PipelineWorker(QObject):
    log = Signal(str)
    finished = Signal()

    def __init__(self, image_paths, npy_dir, overlay_dir, output_dir, model_path):
        super().__init__()
        self.image_paths = image_paths
        self.npy_dir    = npy_dir
        self.overlay_dir = overlay_dir
        self.output_dir = output_dir
        self.model_path  = model_path

    def run(self):
        self.log.emit("Pipeline started.")

        # Cellpose Segmentation Loop
        try:
            run_segmentation(
                image_paths      = self.image_paths,
                save_base_dir    = self.npy_dir,
                overlay_base_dir = self.overlay_dir,
                model_path       = self.model_path,
                log_fn           = self.log.emit,   # ← routes all logs to the UI
            )
        except Exception as e:
            self.log.emit(f"❌ Segmentation error: {e}")

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

        counts = cellcount(results)
        text = "\n".join(f"{key}: {value}" for key, value in counts.items())
        self.log.emit(text)
        self.log.emit("Pipeline finished.")
        self.finished.emit()
