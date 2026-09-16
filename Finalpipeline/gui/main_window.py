import os
from PySide6.QtWidgets import QMainWindow, QFileDialog, QListView, QTreeView, QAbstractItemView
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

        self.ui.Button_Upload.clicked.connect(self.upload_images)
        self.ui.Button_Start.clicked.connect(self.start_pipeline)
        self.ui.Button_Clear.clicked.connect(self.clear_images)
        self.image_paths = []
        self.folder_paths = []
        self._folder_queue = []
        self._current_folder_idx = 0
        self._total_folders = 0
        self.retired = []

    def upload_images(self):
        if self.ui.checkBox_BatchMode.isChecked():
            self._upload_folders()
        else:
            self._upload_files()

    def _upload_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not files:
            return

        new_files = [f for f in files if f not in self.image_paths]
        self.image_paths.extend(new_files)

        self.ui.listWidget_imageList.clear()
        for f in self.image_paths:
            self.ui.listWidget_imageList.addItem(f)

        self.ui.TextEdit_Log.append(f"Added {len(new_files)} image(s) ({len(self.image_paths)} total)")

    def _upload_folders(self):
        dialog = QFileDialog(self, "Select Folders")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

        for view_cls in (QListView, QTreeView):
            view = dialog.findChild(view_cls)
            if view:
                view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        if not dialog.exec():
            return

        folders = dialog.selectedFiles()
        if not folders:
            return

        new_folders = [f for f in folders if f not in self.folder_paths]
        self.folder_paths.extend(new_folders)

        self.ui.listWidget_imageList.clear()   # reusing the same list widget, now showing folder paths
        for f in self.folder_paths:
            self.ui.listWidget_imageList.addItem(f)

        self.ui.TextEdit_Log.append(f"Added {len(new_folders)} folder(s) ({len(self.folder_paths)} total)")

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


    def handle_after_pipeline(self, results, cell_images, pi_value): # need to add the statistics processing also need to implement individual file processing
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
        if self.ui.checkBox_BatchMode.isChecked():
            self.start_batch_pipeline()
        else:
            self.start_single_pipeline()

    def start_single_pipeline(self):
        if not self.image_paths:
            self.ui.TextEdit_Log.append("No images selected.")
            return

        output_dir = Path(self.ui.TextInput_folderpath.toPlainText().strip())
        if not str(output_dir):
            self.ui.TextEdit_Log.append("❌ Please set an output folder.")
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        self.ui.Button_Start.setEnabled(False)
        self.ui.TextEdit_Log.append(f"Starting pipeline on {len(self.image_paths)} image(s)...")
        thread = QThread()
        worker = PipelineWorker(
            image_paths=self.image_paths,
            input_dir=None,
            output_dir=output_dir,
        )   
        worker.moveToThread(thread)

        self.thread = thread
        self.worker = worker

        thread.started.connect(worker.run)
        worker.log.connect(self.ui.TextEdit_Log.append)
        worker.label_PI.connect(self.ui.label_PI.setText)
        worker.label_UnclusteredPI.connect(self.ui.label_UnclusteredPI.setText)
        worker.label_ClusteredPI.connect(self.ui.label_ClusteredPI.setText)
        worker.label_ClusteredMonocyteCount.connect(self.ui.label_ClusteredMonocyteCount.setText)
        worker.label_UnclusteredMonocyteCount.connect(self.ui.label_UnclusteredMonocyteCount.setText)
        worker.label_UnclusteredPhagocyteCount.connect(self.ui.label_UnclusteredPhagocyteCount.setText)
        worker.label_ClusteredPhagocyteCount.connect(self.ui.label_ClusteredPhagocyteCount.setText)
        worker.label_ClusteredRBCCount.connect(self.ui.label_ClusteredRBCCount.setText)
        worker.label_UnclusteredRBCCount.connect(self.ui.label_UnclusteredRBCCount.setText)

        worker.finished_with_data.connect(self.handle_after_pipeline)

        # Proper shutdown order:
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self.ui.Button_Start.setEnabled(True))
        # retire old refs after this pair is fully torn down, so they aren't GC'd early
        thread.finished.connect(lambda: self.retired.append((thread, worker)))

        thread.start()

    def start_batch_pipeline(self):
        if not self.folder_paths:
            self.ui.TextEdit_Log.append("No folders selected.")
            return

        output_root_text = Path(self.ui.TextInput_folderpath.toPlainText().strip())
        if not output_root_text:
            self.ui.TextEdit_Log.append("❌ Please set an output folder.")
            return

        self.output_root = Path(output_root_text)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self._folder_queue = []
        for folder in self.folder_paths:
            root = Path(folder)
            subfolders = sorted(p for p in root.iterdir() if p.is_dir())
            if not subfolders:
                self.ui.TextEdit_Log.append(f"[{root.name}] no subfolders found, skipping.")
                continue
            for sub in subfolders:
                self._folder_queue.append((root.name, sub))

        if not self._folder_queue:
            self.ui.TextEdit_Log.append("No valid samples found across selected folders.")
            return

        self._total_folders = len(self._folder_queue)
        self._current_folder_idx = 0

        self.ui.Button_Start.setEnabled(False)
        self.ui.TextEdit_Log.append(f"Starting batch over {self._total_folders} sample(s)...")
        self._run_next_folder()

    def _run_next_folder(self):
        if not self._folder_queue:
            self.ui.Button_Start.setEnabled(True)
            self.ui.TextEdit_Log.append("✅ All samples complete.")
            return

        donor_name, sample_dir = self._folder_queue.pop(0)
        self._current_folder_idx += 1

        valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_files = sorted(
            str(sample_dir / f) for f in os.listdir(sample_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        )

        if not image_files:
            self.ui.TextEdit_Log.append(f"[{donor_name}/{sample_dir.name}] no images found, skipping.")
            self._run_next_folder()
            return

        sample_output_dir = self.output_root / donor_name / sample_dir.name

        self.ui.TextEdit_Log.append(
            f"[{self._current_folder_idx}/{self._total_folders}] "
            f"{donor_name}/{sample_dir.name}: processing {len(image_files)} images "
            f"→ {sample_output_dir}"
        )

        thread = QThread()
        worker = PipelineWorker(
            image_paths=image_files,
            input_dir=sample_dir,
            output_dir=sample_output_dir,
        )
        worker.moveToThread(thread)
        worker._donor_name = donor_name

        # keep references alive on self so nothing gets GC'd mid-run
        self.thread = thread
        self.worker = worker

        thread.started.connect(worker.run)
        worker.log.connect(self.ui.TextEdit_Log.append)
        worker.label_PI.connect(self.ui.label_PI.setText)
        worker.label_UnclusteredPI.connect(self.ui.label_UnclusteredPI.setText)
        worker.label_ClusteredPI.connect(self.ui.label_ClusteredPI.setText)
        worker.label_ClusteredMonocyteCount.connect(self.ui.label_ClusteredMonocyteCount.setText)
        worker.label_UnclusteredMonocyteCount.connect(self.ui.label_UnclusteredMonocyteCount.setText)
        worker.label_UnclusteredPhagocyteCount.connect(self.ui.label_UnclusteredPhagocyteCount.setText)
        worker.label_ClusteredPhagocyteCount.connect(self.ui.label_ClusteredPhagocyteCount.setText)
        worker.label_ClusteredRBCCount.connect(self.ui.label_ClusteredRBCCount.setText)
        worker.label_UnclusteredRBCCount.connect(self.ui.label_UnclusteredRBCCount.setText)

        worker.finished_with_data.connect(self.handle_after_pipeline)

        # Proper shutdown order:
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._run_next_folder)   # <-- only advance once thread has truly stopped

        # retire old refs after this pair is fully torn down, so they aren't GC'd early
        thread.finished.connect(lambda: self.retired.append((thread, worker)))

        thread.start()