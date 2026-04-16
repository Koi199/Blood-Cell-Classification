import os
from PySide6.QtWidgets import QMainWindow, QFileDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QThread
from pipeline.worker import PipelineWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ui_path = os.path.join(base_dir, "main_window.ui")

        loader = QUiLoader()
        ui_file = QFile(ui_path)

        if not ui_file.exists():
            raise FileNotFoundError(f"UI file not found at: {ui_path}")

        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file)  # no parent
        ui_file.close()

        if self.ui is None:
            raise RuntimeError(f"Failed to load UI: {loader.errorString()}")

        self.setCentralWidget(self.ui)
        self.ui.show()
        self.adjustSize()
        self.setMinimumSize(800, 600)  # locks in the calculated size as the minimum

        self.ui.Button_Upload.clicked.connect(self.upload_images)
        self.ui.Button_Start.clicked.connect(self.start_pipeline)
        self.ui.Button_Clear.clicked.connect(self.clear_images)
        self.image_paths = []

    def upload_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not files:
            return

        # ✅ Extend instead of replace, and deduplicate
        new_files = [f for f in files if f not in self.image_paths]
        self.image_paths.extend(new_files)

        # Refresh the list widget to show all images
        self.ui.listWidget_imageList.clear()
        for f in self.image_paths:
            self.ui.listWidget_imageList.addItem(f)

        self.ui.TextEdit_Log.append(f"Added {len(new_files)} image ({len(self.image_paths)} total)")


    def clear_images(self):
        self.image_paths = []
        self.ui.listWidget_imageList.clear()
        self.ui.TextEdit_Log.append("Image list cleared.")

    def start_pipeline(self):
        if not self.image_paths:
            self.ui.TextEdit_Log.append("No images selected.")
            return

        # Hardcoded for now — could be moved to UI input fields later
        npy_dir    = "D:/MMA_PipelineTest/segmentednpy"
        overlay_dir = "D:/MMA_PipelineTest/Overlay"
        model_path  = "D:/MMA_batch1/TrainedCellpose/models/MMA_trainv3"
        output_dir = "D:/MMA_PipelineTest/SingleCells"

        self.ui.Button_Start.setEnabled(False)
        self.ui.TextEdit_Log.append("Starting pipeline...")

        self.thread = QThread()
        self.worker = PipelineWorker(
            image_paths  = self.image_paths,
            npy_dir     = npy_dir,
            overlay_dir  = overlay_dir,
            output_dir = output_dir,
            model_path   = model_path,
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.ui.TextEdit_Log.append)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(lambda: self.ui.Button_Start.setEnabled(True))
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()