import os
from PySide6.QtWidgets import QMainWindow, QFileDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QThread
from pipeline.worker import PipelineWorker
from pathlib import Path
import pandas as pd
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import numpy as np

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
        self.setMinimumSize(800,600)  # locks in the calculated size as the minimum

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

    def display_image_array_in_label(self, label, np_img):
        # Ensure RGB
        if np_img.ndim == 2:
            np_img = np.stack([np_img]*3, axis=-1)

        h, w, ch = np_img.shape
        bytes_per_line = ch * w

        qimg = QImage(np_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio)

        label.setPixmap(pixmap)


    def handle_after_pipeline(self, results, cell_images, pi_value):
        self.results = results
        self.cell_images = cell_images
        self.pi_value = pi_value

        self.ui.TextEdit_Log.append("Pipeline complete. Running post-processing...")

        try:
            root_dir = Path(self.ui.TextInput_folderpath.toPlainText().strip())
            csv_path = root_dir / 'predictions.csv'

            df = pd.read_csv(csv_path)

            top1_by_class = {}

            for cls, group in df.groupby("class"):
                group_sorted = group.sort_values(by="combined_score", ascending=False)

                parent = group_sorted.iloc[0]["parent"]
                index  = group_sorted.iloc[0]["index"]

                top1_by_class[cls] = (parent, index)

            def show(label_widget, key):
                parent, index = top1_by_class[key]
                np_img = self.cell_images[(parent, index)]
                self.display_image_array_in_label(label_widget, np_img)

            show(self.ui.Image_Nonmonocyte,               'NONmonocyte')
            show(self.ui.Image_MonocytewithRBC,           'UNclustered Monocyte RBC')
            show(self.ui.Image_emptymonocyte,             'UNclustered Monocyte')
            show(self.ui.Image_ClusteredmonocytewithRBC,  'Clustered Monocyte RBC')
            show(self.ui.Image_ClusteredemptyMonocyte,    'Clustered Monocyte')

        except Exception as e:
            self.ui.TextEdit_Log.append(f"Error in post-processing: {e}")
            return

        if self.pi_value > 0.2:
            self.ui.label_Accepted.setStyleSheet('background-color:red;')
            self.ui.label_Accepted.setText('No')
        else:
            self.ui.label_Accepted.setStyleSheet('background-color:green;')
            self.ui.label_Accepted.setText('Yes')


    def start_pipeline(self):
        if not self.image_paths:
            self.ui.TextEdit_Log.append("No images selected.")
            return

        root_dir = Path(self.ui.TextInput_folderpath.toPlainText().strip())
 
        if not root_dir.exists():
            self.ui.TextEdit_Log.append(f"❌ Folder does not exist: {root_dir}")
            try:
                root_dir.mkdir(parents=True, exist_ok=True)
                self.ui.TextEdit_Log.append(f"📁 Created folder: {root_dir}")
            except Exception as e:
                self.ui.TextEdit_Log.append(f"Error: {e}")

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
        self.worker.label_PI.connect(self.ui.label_PI.setText)

        self.worker.label_UnclusteredPI.connect(self.ui.label_UnclusteredPI.setText)
        #self.worker.label_UnclusteredPIError.connect(self.ui.label_UnclusteredPIError.setText)

        self.worker.label_ClusteredPI.connect(self.ui.label_ClusteredPI.setText)
        #self.worker.label_ClusteredPIError.connect(self.ui.label_ClusteredPIError.setText)

        self.worker.label_PI.connect(self.ui.label_PI.setText)
        #self.worker.label_PIError.connect(self.ui.label_PIError.setText)

        self.worker.label_ClusteredMonocyteCount.connect(self.ui.label_ClusteredMonocyteCount.setText)
        #self.worker.label_ClusteredMonocyteCountError.connect(self.ui.label_ClusteredMonocyteCountError.setText)

        self.worker.label_UnclusteredMonocyteCount.connect(self.ui.label_UnclusteredMonocyteCount.setText)
        #self.worker.label_UnclusteredMonocyteCountError.connect(self.ui.label_UnclusteredMonocyteCountError.setText)

        self.worker.label_UnclusteredPhagocyteCount.connect(self.ui.label_UnclusteredPhagocyteCount.setText)
        #self.worker.label_UnclusteredPhagocyteCountError.connect(self.ui.label_UnclusteredPhagocyteCountError.setText)

        self.worker.label_ClusteredPhagocyteCount.connect(self.ui.label_ClusteredPhagocyteCount.setText)
        #self.worker.label_ClusteredPhagocyteCountError.connect(self.ui.label_ClusteredPhagocyteCountError.setText)

        self.worker.label_ClusteredRBCCount.connect(self.ui.label_ClusteredRBCCount.setText)
        #self.worker.label_ClusteredRBCCountError.connect(self.ui.label_ClusteredRBCCountError.setText)

        self.worker.label_UnclusteredRBCCount.connect(self.ui.label_UnclusteredRBCCount.setText)
        #self.worker.label_UnclusteredRBCCountError.connect(self.ui.label_UnclusteredRBCCountError.setText)

        # When worker finishes
        self.worker.finished_with_data.connect(self.handle_after_pipeline)
        self.worker.finished_with_data.connect(self.worker.deleteLater)   # <-- moved here

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(lambda: self.ui.Button_Start.setEnabled(True))
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()