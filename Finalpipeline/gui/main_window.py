import os
from PySide6.QtWidgets import QMainWindow, QFileDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QThread
from pipeline.worker import PipelineWorker
from pathlib import Path
import pandas as pd
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

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

    def display_image_in_label(self, element, filepath):
        pixmap = QPixmap(filepath)
        pixmap = pixmap.scaled(
            element.width(),
            element.height(),
            Qt.KeepAspectRatio
        )
        element.setPixmap(pixmap)

    def after_pipeline(self):
        self.ui.TextEdit_Log.append("Pipeline complete. Running post-processing...")
        root_dir = Path(self.ui.TextInput_folderpath.toPlainText().strip())
        try:
            csv_path = root_dir / 'predictions.csv'

            # Load CSV
            df = pd.read_csv(csv_path)

            # Dictionary: class → top file
            top1_by_class = {}

            # Group by class
            for cls, group in df.groupby("class"):
                # Sort by combined score (descending)
                group_sorted = group.sort_values(by="combined_score", ascending=False)

                # Take the top 1 file
                top_file = group_sorted.iloc[0]["file"]

                top1_by_class[cls] = top_file

            # Example: pick one class to display
            # (you can choose which class you want to show)
            some_class = list(top1_by_class.keys())[0]
            best_image = top1_by_class[some_class]

            self.ui.TextEdit_Log.append(f"Top image for {some_class}: {best_image}")

            # Display in QLabel
            self.display_image_in_label(self.ui.Image_1, top1_by_class['NONmonocyte'])
            self.display_image_in_label(self.ui.Image_2, top1_by_class['UNclustered Monocyte oneRBC'])
            self.display_image_in_label(self.ui.Image_3, top1_by_class['UNclustered Monocyte'])
            self.display_image_in_label(self.ui.Image_4, top1_by_class['Clustered Monocyte oneRBC'])
            self.display_image_in_label(self.ui.Image_5, top1_by_class['Clustered Monocyte'])

        except Exception as e:
            self.ui.TextEdit_Log.append(f"Error in post-processing: {e}")

    def start_pipeline(self):
        if not self.image_paths:
            self.ui.TextEdit_Log.append("No images selected.")
            return

        root_dir = Path(self.ui.TextInput_folderpath.toPlainText().strip())
 
        if not root_dir.exists():
            self.ui.TextEdit_Log.append(f"❌ Folder does not exist: {root_dir}")
            return

        self.ui.Button_Start.setEnabled(False)
        self.ui.TextEdit_Log.append("Starting pipeline...")

        self.thread = QThread()
        self.worker = PipelineWorker(
            image_paths  = self.image_paths,
            root_dir= root_dir
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.ui.TextEdit_Log.append)
        # When worker finishes
        self.worker.finished.connect(self.after_pipeline) #<-- figure out the images to display
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(lambda: self.ui.Button_Start.setEnabled(True))
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()