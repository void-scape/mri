from __future__ import annotations

import sys
import pathlib
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QFormLayout, QFrame, QFileDialog, QMessageBox
)

# ---- ensure repo root is importable (assumes this file lives in repo_root/viewer/main.py) ----
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- local import (shared inference stub) ----
from shared.infer import run_inference


# ---------- helpers ----------
def make_output_mask_path(image_path: Path) -> Path:
    """
    Create <basename>_mask.nii.gz next to the image, handling double suffixes like .nii.gz.

    Examples:
      foo.nii.gz -> foo_mask.nii.gz
      foo.nii    -> foo_mask.nii.gz
      foo.mhd    -> foo_mask.nii.gz
    """
    name = image_path.name
    if name.endswith(".nii.gz"):
        base = name[:-len(".nii.gz")]
    else:
        # strip only the last suffix (e.g., .nii, .mhd, .nrrd, .mha)
        base = image_path.stem
    return image_path.with_name(f"{base}_mask.nii.gz")


def find_groupbox_label(box: QGroupBox) -> QLabel:
    lbl = box.findChild(QLabel)
    if lbl is None:
        raise RuntimeError(f"No QLabel found inside groupbox '{box.title()}'")
    return lbl


class Divider(QFrame):
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine if orientation == Qt.Horizontal else QFrame.VLine)
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
    finished = QtCore.Signal(str)   # output mask path
    progress = QtCore.Signal(int)   # 0..100
    error = QtCore.Signal(str)      # error message

    def __init__(self, image_path: Path, out_path: Path):
        super().__init__()
        self.image_path = Path(image_path)
        self.out_path = Path(out_path)

    @QtCore.Slot()
    def run(self):
        try:
            def cb(pct: int):
                self.progress.emit(int(pct))
            out = run_inference(self.image_path, self.out_path, progress_callback=cb)
            self.finished.emit(str(out))
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cartilage Viewer (V-Net) — Initial Design")
        self.resize(1200, 800)

        self._image_path: Path | None = None
        self._mask_path: Path | None = None
        self._mask_visible = False
        self._is_running = False

        # ===== Toolbar =====
        tb = self.addToolBar("Main")
        self.act_open = QtGui.QAction("Open Input Image…", self)
        tb.addAction(self.act_open)

        # ===== Left controls =====
        left_box = QGroupBox("Overlay")
        left_layout = QFormLayout(left_box)

        self.btn_toggle_mask = QPushButton("Show Cartilage Mask")
        self.btn_toggle_mask.setEnabled(False)

        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(60)
        self.opacity.setToolTip("Overlay opacity")

        left_layout.addRow(self.btn_toggle_mask)
        left_layout.addRow("Opacity", self.opacity)

        # ===== Center: 3 panes + sliders =====
        center = QWidget()
        center_layout = QVBoxLayout(center)

        views_row = QWidget()
        views_grid = QGridLayout(views_row)
        views_grid.setContentsMargins(8, 8, 8, 8)
        views_grid.setHorizontalSpacing(16)
        views_grid.setVerticalSpacing(8)

        self.sag_box = make_placeholder("Sagittal")
        self.cor_box = make_placeholder("Coronal")
        self.ax_box = make_placeholder("Axial")

        views_grid.addWidget(self.sag_box, 0, 0)
        views_grid.addWidget(self.cor_box, 0, 1)
        views_grid.addWidget(self.ax_box, 0, 2)
        views_grid.setColumnStretch(0, 1)
        views_grid.setColumnStretch(1, 2)
        views_grid.setColumnStretch(2, 2)

        center_layout.addWidget(views_row, stretch=10)

        sliders = QWidget()
        g = QGridLayout(sliders)
        g.setColumnStretch(1, 1)     # make the slider column actually expand

        self.sag_slider = QSlider(Qt.Horizontal)
        self.cor_slider = QSlider(Qt.Horizontal)
        self.ax_slider = QSlider(Qt.Horizontal)

        for s in (self.sag_slider, self.cor_slider, self.ax_slider):
            s.setRange(0, 100)       # so the groove draws on macOS
            s.setValue(0)
            s.setEnabled(False)      # still disabled until image load
            s.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)


        g.addWidget(QLabel("Sagittal"), 0, 0)
        g.addWidget(self.sag_slider, 0, 1)
        g.addWidget(QLabel("Coronal"), 1, 0)
        g.addWidget(self.cor_slider, 1, 1)
        g.addWidget(QLabel("Axial"), 2, 0)
        g.addWidget(self.ax_slider, 2, 1)

        center_layout.addWidget(sliders, stretch=0)

        # ===== Main layout =====
        main = QWidget()
        h = QHBoxLayout(main)
        h.addWidget(left_box, 2)
        h.addWidget(center, 9)
        self.setCentralWidget(main)

        # ===== Status / progress =====
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)   # indeterminate until real % arrives
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Ready. Load an input image to begin.")

        # ===== Wiring =====
        self.act_open.triggered.connect(self.on_open_image)
        self.btn_toggle_mask.clicked.connect(self.on_toggle_mask)

        self.sag_slider.valueChanged.connect(lambda v: self._on_slice_changed("Sagittal", self.sag_box, v))
        self.cor_slider.valueChanged.connect(lambda v: self._on_slice_changed("Coronal", self.cor_box, v))
        self.ax_slider.valueChanged.connect(lambda v: self._on_slice_changed("Axial", self.ax_box, v))

        # keep these for later; MVP will use them
        self.opacity.valueChanged.connect(lambda *_: None)

        # init placeholders
        self._refresh_view_labels()

    # ---------- UI helpers ----------
    def _set_running(self, running: bool):
        self._is_running = running
        self.act_open.setEnabled(not running)
        self.progress.setVisible(running)

    def _refresh_view_labels(self):
        # show slice index + whether mask is toggled (placeholder behavior for now)
        sag_lbl = find_groupbox_label(self.sag_box)
        cor_lbl = find_groupbox_label(self.cor_box)
        ax_lbl = find_groupbox_label(self.ax_box)

        sag_lbl.setText(self._label_text("Sagittal", self.sag_slider.value()))
        cor_lbl.setText(self._label_text("Coronal", self.cor_slider.value()))
        ax_lbl.setText(self._label_text("Axial", self.ax_slider.value()))

    def _label_text(self, plane: str, idx: int) -> str:
        mask = "MASK ON" if self._mask_visible else "MASK OFF"
        return f"{plane} View\nSlice: {idx}\n{mask}"

    def _enable_sliders_placeholder(self):
        # Until we load real volume dims (MVP step), give a “reasonable” range to test interaction.
        for s in (self.sag_slider, self.cor_slider, self.ax_slider):
            s.setRange(0, 100)
            s.setValue(50)
            s.setEnabled(True)

    def _on_slice_changed(self, plane: str, box: QGroupBox, v: int):
        lbl = find_groupbox_label(box)
        lbl.setText(self._label_text(plane, v))

    def _show_error(self, title: str, msg: str):
        QMessageBox.critical(self, title, msg)

    # ---------- Slots ----------
    def on_open_image(self):
        if self._is_running:
            return

        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Open Input Image",
            "",
            "Images (*.mhd *.nii *.nii.gz *.nrrd *.mha)"
        )
        if not fp:
            return

        self._image_path = Path(fp)
        self._mask_path = None
        self._mask_visible = False
        self.btn_toggle_mask.setEnabled(False)
        self.btn_toggle_mask.setText("Show Cartilage Mask")

        self._enable_sliders_placeholder()
        self._refresh_view_labels()

        self.statusBar().showMessage(f"Loaded image: {self._image_path.name}. Running V-Net in background…")

        out = make_output_mask_path(self._image_path)

        # spin up a worker thread
        self.thread = QtCore.QThread(self)
        self.worker = InferWorker(self._image_path, out)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_infer_finished)
        self.worker.error.connect(self.on_infer_error)

        # Cleanup
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self._set_running(True)
        self.progress.setRange(0, 0)   # indeterminate at start
        self.thread.start()

    @QtCore.Slot(int)
    def on_progress(self, pct: int):
        if self.progress.minimum() == 0 and self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
        self.progress.setValue(max(0, min(100, pct)))

    @QtCore.Slot(str)
    def on_infer_finished(self, mask_path: str):
        self._mask_path = Path(mask_path)
        self._set_running(False)

        self.btn_toggle_mask.setEnabled(True)
        self.statusBar().showMessage(
            f"V-Net done. Mask saved to {self._mask_path.name}. Click 'Show Cartilage Mask'."
        )

    @QtCore.Slot(str)
    def on_infer_error(self, err: str):
        self._set_running(False)
        self.statusBar().showMessage("Inference failed.")
        self._show_error("Inference Error", err)

    def on_toggle_mask(self):
        if not self._mask_path:
            return
        self._mask_visible = not self._mask_visible
        self.btn_toggle_mask.setText("Hide Cartilage Mask" if self._mask_visible else "Show Cartilage Mask")
        self._refresh_view_labels()
        self.statusBar().showMessage(f"Mask {'shown' if self._mask_visible else 'hidden'} (placeholder overlay).")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
