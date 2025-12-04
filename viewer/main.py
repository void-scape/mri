from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QCheckBox, QSlider, QGroupBox, QFormLayout, QFrame, QFileDialog
)

# ---- local import (shared inference stub) ----
# keep repo root as CWD; don't make viewer a package
from shared.infer import run_inference

class Divider(QFrame):
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine if orientation==Qt.Horizontal else QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)

def make_placeholder(title: str) -> QGroupBox:
    box = QGroupBox(title)
    box.setStyleSheet("QGroupBox{font-weight:600;}")
    layout = QVBoxLayout(box)
    area = QLabel(f"{title} View")
    area.setMinimumSize(320, 320)
    area.setAlignment(Qt.AlignCenter)
    area.setStyleSheet("border:1px dashed #999; color:#666;")
    layout.addWidget(area)
    return box

# ---- Worker to run inference in background ----
class InferWorker(QtCore.QObject):
    finished = QtCore.Signal(str)       # output mask path
    progress = QtCore.Signal(int)       # 0..100

    def __init__(self, image_path: Path, out_path: Path):
        super().__init__()
        self.image_path = Path(image_path)
        self.out_path = Path(out_path)

    @QtCore.Slot()
    def run(self):
        def cb(pct: int):
            self.progress.emit(int(pct))
        out = run_inference(self.image_path, self.out_path, progress_callback=cb)
        self.finished.emit(str(out))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cartilage Viewer (V-Net) —  Initial Design")
        self.resize(1200, 800)

        self._image_path: Path | None = None
        self._mask_path: Path | None  = None
        self._mask_visible = False

        # ===== Toolbar (only one action) =====
        tb = self.addToolBar("Main")
        self.act_open = QtGui.QAction("Open Input Image…", self)
        tb.addAction(self.act_open)

        # ===== Left: very small control panel =====
        left_box = QGroupBox("Overlay")
        left_layout = QFormLayout(left_box)

        self.btn_toggle_mask = QPushButton("Show Cartilage Mask")
        self.btn_toggle_mask.setEnabled(False)  # enabled after inference finishes

        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(60)
        self.opacity.setToolTip("Overlay opacity")

        left_layout.addRow(self.btn_toggle_mask)
        left_layout.addRow("Opacity", self.opacity)

        # ===== Center: 3 viewer panes + sliders =====
        center = QWidget(); center_layout = QVBoxLayout(center)

        views_row = QWidget(); views_grid = QGridLayout(views_row)
        views_grid.setContentsMargins(8, 8, 8, 8)
        views_grid.setHorizontalSpacing(16); views_grid.setVerticalSpacing(8)

        self.sag_box = make_placeholder("Sagittal")
        self.cor_box = make_placeholder("Coronal")
        self.ax_box  = make_placeholder("Axial")

        views_grid.addWidget(self.sag_box, 0, 0)
        views_grid.addWidget(self.cor_box, 0, 1)
        views_grid.addWidget(self.ax_box,  0, 2)
        views_grid.setColumnStretch(0, 1)
        views_grid.setColumnStretch(1, 2)
        views_grid.setColumnStretch(2, 2)
        center_layout.addWidget(views_row, stretch=10)

        sliders = QWidget(); g = QGridLayout(sliders)
        self.sag_slider = QSlider(Qt.Horizontal); self.sag_slider.setRange(0, 100)
        self.cor_slider = QSlider(Qt.Horizontal); self.cor_slider.setRange(0, 100)
        self.ax_slider  = QSlider(Qt.Horizontal); self.ax_slider.setRange(0, 100)
        self.cb_link    = QCheckBox("Link views")

        g.addWidget(QLabel("Sagittal"), 0, 0); g.addWidget(self.sag_slider, 0, 1)
        g.addWidget(QLabel("Coronal"),  1, 0); g.addWidget(self.cor_slider,  1, 1)
        g.addWidget(QLabel("Axial"),    2, 0); g.addWidget(self.ax_slider,   2, 1)
        g.addWidget(self.cb_link, 3, 1, alignment=Qt.AlignRight)
        center_layout.addWidget(sliders, stretch=0)

        # ===== Main layout =====
        main = QWidget(); h = QHBoxLayout(main)
        h.addWidget(left_box, 2)
        h.addWidget(center, 9)
        self.setCentralWidget(main)

        # ===== Status/progress =====
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate until we get real %; we’ll switch to 0..100 when progress arrives
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Ready. Load an input image to begin.")

        # ===== Wiring =====
        self.act_open.triggered.connect(self.on_open_image)
        self.btn_toggle_mask.clicked.connect(self.on_toggle_mask)

        # (placeholders for future)
        for s in (self.sag_slider, self.cor_slider, self.ax_slider):
            s.valueChanged.connect(lambda *_: None)
        self.opacity.valueChanged.connect(lambda *_: None)
        self.cb_link.stateChanged.connect(lambda *_: None)

    # ---------- Slots ----------
    def on_open_image(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Open Input Image", "", "Images (*.mhd *.nii *.nii.gz *.nrrd *.mha)")
        if not fp:
            return
        self._image_path = Path(fp)
        self._mask_path  = None
        self._mask_visible = False
        self.btn_toggle_mask.setEnabled(False)
        self.btn_toggle_mask.setText("Show Cartilage Mask")
        self.statusBar().showMessage(f"Loaded image: {self._image_path.name}. Running V-Net in background…")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)   # indeterminate at start

        # choose an output path next to the image for now
        out = self._image_path.with_suffix("").with_name(self._image_path.stem + "_mask.nii.gz")

        # spin up a worker thread
        self.thread = QtCore.QThread(self)
        self.worker = InferWorker(self._image_path, out)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_infer_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @QtCore.Slot(int)
    def on_progress(self, pct: int):
        if self.progress.minimum() == 0 and self.progress.maximum() == 0:
            self.progress.setRange(0, 100)  # switch from indeterminate to % as soon as we get numbers
        self.progress.setValue(max(0, min(100, pct)))

    @QtCore.Slot(str)
    def on_infer_finished(self, mask_path: str):
        self._mask_path = Path(mask_path)
        self.progress.setVisible(False)
        self.btn_toggle_mask.setEnabled(True)
        self.statusBar().showMessage(f"V-Net done. Mask saved to {self._mask_path.name}. Click 'Show Cartilage Mask'.")

    def on_toggle_mask(self):
        if not self._mask_path:
            return
        self._mask_visible = not self._mask_visible
        self.btn_toggle_mask.setText("Hide Cartilage Mask" if self._mask_visible else "Show Cartilage Mask")
        # Placeholder: reflect state in labels
        for box in (self.sag_box, self.cor_box, self.ax_box):
            lbl: QLabel = box.findChild(QLabel)
            base = box.title().split()[0]  # Sagittal / Coronal / Axial
            lbl.setText(f"{base} View — MASK {'ON' if self._mask_visible else 'OFF'}")
        self.statusBar().showMessage(f"Mask {'shown' if self._mask_visible else 'hidden'} (fill mode).")

def main():
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
