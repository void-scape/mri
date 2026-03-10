from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np

from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QFormLayout, QFrame, QFileDialog,
    QMessageBox, QComboBox, QProgressBar
)

# Optional deps (only needed for specific formats)
try:
    import h5py
except Exception:
    h5py = None

try:
    import SimpleITK as sitk
except Exception:
    sitk = None


# ----------------- UI helpers -----------------
class Divider(QFrame):
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine if orientation == Qt.Horizontal else QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


def make_view_label() -> QLabel:
    lbl = QLabel("No image loaded")
    lbl.setMinimumSize(320, 320)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet("border:1px dashed #999; color:#666;")
    lbl.setScaledContents(False)
    return lbl


def make_info_label() -> QLabel:
    lbl = QLabel("Slice: - / -   •   Zoom: 1.00×   •   Vol: -×-×-")
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet("color:#aaa; font-size:11px;")
    return lbl


def msg_error(parent, title: str, text: str):
    QMessageBox.critical(parent, title, text)


# ----------------- IO: load image + mask -----------------
def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def is_hdf5(path: Path) -> bool:
    # Your ".im" files are HDF5 containers; treat .im as HDF5 too
    return path.suffix.lower() in {".h5", ".hdf5", ".im", ".seg"}


def load_hdf5_volume_xyz(path: Path, key: str = "data") -> np.ndarray:
    if h5py is None:
        raise RuntimeError("h5py is not installed. Install with: python -m pip install h5py")
    with h5py.File(str(path), "r") as f:
        if key not in f:
            raise KeyError(f"Dataset key '{key}' not found. Keys: {list(f.keys())}")
        arr = np.array(f[key])
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume in '{key}', got shape={arr.shape}")
    return arr  # dataset-native ordering (X,Y,Z), matches your .npy masks


def load_nifti_volume_xyz(path: Path) -> np.ndarray:
    """
    Reads NIfTI and returns volume as (X,Y,Z).
    NOTE: NIfTI orientation conventions vary; this is best-effort for interoperability.
    Primary supported workflow is HDF5(.im/.hdf5) + .npy.
    """
    if sitk is None:
        raise RuntimeError("SimpleITK is not installed. Install with: python -m pip install SimpleITK")
    img = sitk.ReadImage(str(path))
    vol_zyx = sitk.GetArrayFromImage(img)  # (Z,Y,X)
    if vol_zyx.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape={vol_zyx.shape}")
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0))  # (X,Y,Z)
    return vol_xyz


def load_volume_xyz(path: Path) -> np.ndarray:
    if is_hdf5(path):
        return load_hdf5_volume_xyz(path, key="data")
    if is_nifti(path):
        return load_nifti_volume_xyz(path)
    raise ValueError(f"Unsupported image format: {path.name}")


def onehot_to_labels(mask: np.ndarray) -> np.ndarray:
    """
    Converts one-hot (X,Y,Z,C) -> labels (X,Y,Z) with values 0..C
    Background=0 where sum over channels == 0.
    """
    if mask.ndim == 3:
        return mask.astype(np.uint8)

    if mask.ndim != 4:
        raise ValueError(f"Mask must be 3D labels or 4D one-hot. Got shape={mask.shape}")

    sums = mask.sum(axis=-1)
    labels = np.argmax(mask, axis=-1).astype(np.uint8) + 1  # 1..C
    labels[sums == 0] = 0
    return labels



def load_mask_xyz(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        m = np.load(str(path))
        return onehot_to_labels(m)

    if is_nifti(path):
        return load_nifti_volume_xyz(path).astype(np.uint8)

    if is_hdf5(path):
        m = load_hdf5_volume_xyz(path, key="data")
        return m.astype(np.uint8)

    raise ValueError(f"Unsupported mask format: {path.name}")


# ----------------- Dice evaluation -----------------
def dice_for_label(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    p = (pred == label)
    g = (gt == label)
    ps = int(p.sum())
    gs = int(g.sum())
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    inter = int(np.logical_and(p, g).sum())
    return (2.0 * inter) / (ps + gs)


def dice_foreground(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0
    g = gt > 0
    ps = int(p.sum())
    gs = int(g.sum())
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    inter = int(np.logical_and(p, g).sum())
    return (2.0 * inter) / (ps + gs)


def dice_report(pred: np.ndarray, gt: np.ndarray) -> Tuple[str, float, float]:
    """
    Returns (multi-line report, fg_dice, mean_label_dice_over_gt_labels)
    Computes mean dice over labels present in GT (excluding 0).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    pred_u = sorted(np.unique(pred).astype(int).tolist())
    gt_u = sorted(np.unique(gt).astype(int).tolist())

    labels_gt = [l for l in gt_u if l != 0]
    fg = dice_foreground(pred, gt)

    per = []
    if labels_gt:
        for l in labels_gt:
            per.append((l, dice_for_label(pred, gt, l)))
        mean_d = float(np.mean([d for _, d in per]))
    else:
        mean_d = float("nan")

    extra_pred = [l for l in pred_u if l != 0 and l not in gt_u]

    lines = []
    lines.append(f"Foreground Dice (pred>0 vs gt>0): {fg:.4f}")
    if labels_gt:
        lines.append(f"Mean Dice over GT labels {labels_gt}: {mean_d:.4f}")
        lines.append("Per-label Dice (GT labels):")
        for l, d in per:
            lines.append(f"  Label {l}: {d:.4f}")
    else:
        lines.append("No non-zero labels found in GT.")

    if extra_pred:
        counts = {l: int(np.sum(pred == l)) for l in extra_pred}
        lines.append(f"Note: pred has labels not in GT: {extra_pred} (voxel counts: {counts})")

    return "\n".join(lines), fg, mean_d


# ----------------- rendering helpers -----------------
def robust_window(arr: np.ndarray) -> Tuple[float, float]:
    a = arr.astype(np.float32)
    lo = float(np.percentile(a, 1.0))
    hi = float(np.percentile(a, 99.0))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def to_uint8(slice2d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = slice2d.astype(np.float32)
    x = (x - lo) / (hi - lo + 1e-8)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def label_colors_rgba() -> np.ndarray:
    return np.array([
        [0,   0,   0,   0],    # 0 background
        [255, 0,   0, 255],    # 1 red
        [0, 255,   0, 255],    # 2 green
        [255, 255, 0, 255],    # 3 yellow
        [0, 255, 255, 255],    # 4 cyan
        [255, 0, 255, 255],    # 5 magenta
        [255, 128, 0, 255],    # 6 orange
    ], dtype=np.uint8)


def blend_overlay(gray8: np.ndarray, labels2d: Optional[np.ndarray], opacity_0_100: int) -> QtGui.QImage:
    h, w = gray8.shape
    rgb = np.repeat(gray8[:, :, None], 3, axis=2).astype(np.float32)

    if labels2d is not None:
        lut = label_colors_rgba()
        lab = labels2d.astype(np.int32)
        lab = np.clip(lab, 0, lut.shape[0] - 1)
        rgba = lut[lab]  # (H,W,4)

        alpha = (rgba[:, :, 3].astype(np.float32) / 255.0) * (opacity_0_100 / 100.0)
        a = alpha[:, :, None]
        over_rgb = rgba[:, :, :3].astype(np.float32)

        rgb = rgb * (1.0 - a) + over_rgb * a

    out = np.clip(rgb, 0, 255).astype(np.uint8)
    qimg = QtGui.QImage(out.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    return qimg


# ----------------- Inference worker -----------------
class InferWorker(QtCore.QObject):
    finished = QtCore.Signal(str)   # output path
    failed = QtCore.Signal(str)     # error log

    def __init__(self, cmd: list[str], cwd: Path, out_path: Path):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.out_path = out_path

    @QtCore.Slot()
    def run(self):
        try:
            p = subprocess.run(self.cmd, cwd=str(self.cwd), capture_output=True, text=True)
            if p.returncode != 0:
                log = (p.stdout or "") + "\n" + (p.stderr or "")
                self.failed.emit(log.strip() or f"Process failed with return code {p.returncode}")
                return
            self.finished.emit(str(self.out_path))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# ----------------- main window -----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cartilage Viewer")
        self.resize(1200, 820)

        self.image_path: Optional[Path] = None
        self.mask_path: Optional[Path] = None

        self.vol_xyz: Optional[np.ndarray] = None
        self.mask_xyz: Optional[np.ndarray] = None
        self.mask_visible = False

        # Ground truth
        self.gt_path: Optional[Path] = None
        self.gt_xyz: Optional[np.ndarray] = None

        self.win_lo = 0.0
        self.win_hi = 1.0

        # 3D cursor (x,y,z) in voxel coords
        self.cursor_xyz = [0, 0, 0]

        # For click/drag mapping and crosshair drawing
        self._label_xform: Dict[QLabel, Dict[str, float]] = {}
        self._slice_hw: Dict[QLabel, Tuple[int, int]] = {}
        self._dragging_label: Optional[QLabel] = None

        # Remember last directory for file dialogs
        self.last_dir: Path = Path.cwd()

        # Inference paths
        self.ckpt_path: Optional[Path] = (Path("checkpoints") / "vnet_model_best.pth.tar")
        if not self.ckpt_path.exists():
            self.ckpt_path = None
        self.infer_script = Path("inference") / "infer_knee.py"

        # ===== Toolbar =====
        tb = self.addToolBar("Main")
        self.act_open_img = QtGui.QAction("Open Image…", self)
        self.act_open_mask = QtGui.QAction("Open Mask…", self)
        self.act_open_gt = QtGui.QAction("Open Ground Truth…", self)
        self.act_clear_gt = QtGui.QAction("Clear Ground Truth", self)
        self.act_clear_mask = QtGui.QAction("Clear Mask", self)
        self.act_auto = QtGui.QAction("Auto Contrast", self)

        self.act_set_ckpt = QtGui.QAction("Set Checkpoint…", self)
        self.act_run_infer = QtGui.QAction("Run Inference", self)

        tb.addAction(self.act_open_img)
        tb.addAction(self.act_open_mask)
        tb.addAction(self.act_open_gt)
        tb.addAction(self.act_clear_gt)
        tb.addAction(self.act_clear_mask)
        tb.addSeparator()
        tb.addAction(self.act_auto)
        tb.addSeparator()
        tb.addAction(self.act_set_ckpt)
        tb.addAction(self.act_run_infer)

        # ===== Left controls =====
        left_box = QGroupBox("Overlay")
        left_layout = QFormLayout(left_box)

        self.btn_toggle_mask = QPushButton("Show Mask")
        self.btn_toggle_mask.setEnabled(False)

        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(60)
        self.opacity.setEnabled(False)

        self.label_filter = QComboBox()
        self.label_filter.setEnabled(False)
        self.label_filter.addItem("All labels", 0)

        self.cursor_label = QLabel("Cursor (x,y,z): - , - , -")
        self.cursor_label.setStyleSheet("color:#bbb; font-size:11px;")

        self.dice_label = QLabel("Dice: load GT + prediction to compute")
        self.dice_label.setWordWrap(True)
        self.dice_label.setStyleSheet("color:#bbb; font-size:11px;")

        left_layout.addRow(self.btn_toggle_mask)
        left_layout.addRow("Opacity", self.opacity)
        left_layout.addRow("Show label", self.label_filter)
        left_layout.addRow(Divider())
        left_layout.addRow(self.cursor_label)
        left_layout.addRow("Dice vs GT", self.dice_label)

        # ===== Center: views + sliders =====
        center = QWidget()
        center_layout = QVBoxLayout(center)

        views_row = QWidget()
        views_grid = QGridLayout(views_row)
        views_grid.setContentsMargins(8, 8, 8, 8)
        views_grid.setHorizontalSpacing(16)
        views_grid.setVerticalSpacing(8)

        self.sag_label = make_view_label()
        self.cor_label = make_view_label()
        self.ax_label = make_view_label()

        self.sag_info = make_info_label()
        self.cor_info = make_info_label()
        self.ax_info = make_info_label()

        self.sag_box = QGroupBox("Sagittal")
        self.cor_box = QGroupBox("Coronal")
        self.ax_box = QGroupBox("Axial")

        views = [
            (self.sag_box, self.sag_label, self.sag_info),
            (self.cor_box, self.cor_label, self.cor_info),
            (self.ax_box,  self.ax_label,  self.ax_info),
        ]
        for box, img_lbl, info_lbl in views:
            box.setStyleSheet("QGroupBox{font-weight:600;}")
            lay = QVBoxLayout(box)
            lay.addWidget(img_lbl, stretch=1)
            lay.addWidget(info_lbl, stretch=0)

        views_grid.addWidget(self.sag_box, 0, 0)
        views_grid.addWidget(self.cor_box, 0, 1)
        views_grid.addWidget(self.ax_box, 0, 2)
        views_grid.setColumnStretch(0, 1)
        views_grid.setColumnStretch(1, 1)
        views_grid.setColumnStretch(2, 1)

        center_layout.addWidget(views_row, stretch=10)

        sliders = QWidget()
        g = QGridLayout(sliders)
        g.setColumnStretch(1, 1)

        self.sag_slider = QSlider(Qt.Horizontal)  # X
        self.cor_slider = QSlider(Qt.Horizontal)  # Y
        self.ax_slider  = QSlider(Qt.Horizontal)  # Z

        for s in (self.sag_slider, self.cor_slider, self.ax_slider):
            s.setRange(0, 100)
            s.setValue(0)
            s.setEnabled(False)

        g.addWidget(QLabel("Sagittal (X)"), 0, 0)
        g.addWidget(self.sag_slider, 0, 1)
        g.addWidget(QLabel("Coronal (Y)"), 1, 0)
        g.addWidget(self.cor_slider, 1, 1)
        g.addWidget(QLabel("Axial (Z)"), 2, 0)
        g.addWidget(self.ax_slider, 2, 1)

        center_layout.addWidget(sliders, stretch=0)

        # ===== Main layout =====
        main = QWidget()
        h = QHBoxLayout(main)
        h.addWidget(left_box, 2)
        h.addWidget(center, 9)
        self.setCentralWidget(main)

        # Status bar + progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Open an image (.hdf5/.im or .nii.gz)")

        # ===== Zoom state =====
        self._label_zoom = {self.sag_label: 1.0, self.cor_label: 1.0, self.ax_label: 1.0}
        self._min_zoom = 0.5
        self._max_zoom = 8.0

        # ===== Wiring =====
        self.act_open_img.triggered.connect(self.on_open_image)
        self.act_open_mask.triggered.connect(self.on_open_mask)
        self.act_open_gt.triggered.connect(self.on_open_gt)
        self.act_clear_gt.triggered.connect(self.on_clear_gt)
        self.act_clear_mask.triggered.connect(self.on_clear_mask)
        self.act_auto.triggered.connect(self.on_auto_contrast)

        self.act_set_ckpt.triggered.connect(self.on_set_checkpoint)
        self.act_run_infer.triggered.connect(self.on_run_inference)

        self.btn_toggle_mask.clicked.connect(self.on_toggle_mask)
        self.opacity.valueChanged.connect(lambda *_: self.render_all())
        self.label_filter.currentIndexChanged.connect(lambda *_: self.render_all())

        self.sag_slider.valueChanged.connect(self.on_x_changed)
        self.cor_slider.valueChanged.connect(self.on_y_changed)
        self.ax_slider.valueChanged.connect(self.on_z_changed)

        for lbl in (self.sag_label, self.cor_label, self.ax_label):
            lbl.installEventFilter(self)
            lbl.setFocusPolicy(Qt.WheelFocus)
            lbl.setMouseTracking(True)
            lbl.setCursor(Qt.CrossCursor)
            lbl.setToolTip(
                "Wheel: change slice | Shift+Wheel: jump 10 | Ctrl+Wheel: zoom | "
                "Click + drag: move crosshair | Double-click: reset zoom | Ctrl+0: reset all zoom"
            )

        self._label_to_slider = {
            self.sag_label: self.sag_slider,
            self.cor_label: self.cor_slider,
            self.ax_label:  self.ax_slider,
        }

        self._infer_thread: Optional[QtCore.QThread] = None
        self._infer_worker: Optional[InferWorker] = None

        self._refresh_infer_action_state()

    def _refresh_infer_action_state(self):
        has_img = self.image_path is not None
        has_ckpt = self.ckpt_path is not None and self.ckpt_path.exists()
        has_script = self.infer_script.exists()
        self.act_run_infer.setEnabled(has_img and has_ckpt and has_script)

    def _set_busy(self, busy: bool, msg: str = ""):
        self.progress.setVisible(busy)
        for a in (self.act_open_img, self.act_open_mask, self.act_open_gt, self.act_clear_gt,
                  self.act_clear_mask, self.act_auto, self.act_set_ckpt, self.act_run_infer):
            a.setEnabled(not busy)
        if not busy:
            self._refresh_infer_action_state()
        if msg:
            self.statusBar().showMessage(msg)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if (event.modifiers() & Qt.ControlModifier) and (event.key() == Qt.Key_0):
            for k in list(self._label_zoom.keys()):
                self._label_zoom[k] = 1.0
            self.render_all()
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonDblClick and obj in getattr(self, "_label_zoom", {}):
            self._label_zoom[obj] = 1.0
            self.render_all()
            event.accept()
            return True

        if event.type() == QtCore.QEvent.MouseButtonPress and obj in (self.sag_label, self.cor_label, self.ax_label):
            if event.button() == Qt.LeftButton:
                started = self._begin_drag_on_label(obj, event.position().x(), event.position().y())
                if started:
                    event.accept()
                    return True

        if event.type() == QtCore.QEvent.MouseMove and obj in (self.sag_label, self.cor_label, self.ax_label):
            if self._dragging_label is obj and (event.buttons() & Qt.LeftButton):
                self._handle_drag_on_label(obj, event.position().x(), event.position().y())
                event.accept()
                return True

        if event.type() == QtCore.QEvent.MouseButtonRelease and obj in (self.sag_label, self.cor_label, self.ax_label):
            if event.button() == Qt.LeftButton and self._dragging_label is obj:
                self._end_drag_on_label(obj)
                event.accept()
                return True

        if event.type() == QtCore.QEvent.Wheel and obj in getattr(self, "_label_to_slider", {}):
            slider = self._label_to_slider[obj]
            if not slider.isEnabled():
                return True

            delta = event.angleDelta().y()
            if delta == 0:
                delta = event.pixelDelta().y()

            steps = int(delta / 120)
            if steps == 0 and delta != 0:
                steps = 1 if delta > 0 else -1

            if event.modifiers() & Qt.ControlModifier:
                z = float(self._label_zoom.get(obj, 1.0))
                z *= (1.10 ** steps)
                z = max(self._min_zoom, min(self._max_zoom, z))
                self._label_zoom[obj] = z
                self.render_all()
                event.accept()
                return True

            if event.modifiers() & Qt.ShiftModifier:
                steps *= 10

            new_val = max(slider.minimum(), min(slider.maximum(), slider.value() + steps))
            slider.setValue(new_val)

            event.accept()
            return True

        if event.type() == QtCore.QEvent.Resize and obj in (self.sag_label, self.cor_label, self.ax_label):
            self.render_all()

        return super().eventFilter(obj, event)

    def on_x_changed(self, v: int):
        self.cursor_xyz[0] = int(v)
        self.render_all()

    def on_y_changed(self, v: int):
        self.cursor_xyz[1] = int(v)
        self.render_all()

    def on_z_changed(self, v: int):
        self.cursor_xyz[2] = int(v)
        self.render_all()

    # ---------- file loading ----------
    def on_open_image(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open Image", str(self.last_dir),
            "Images (*.hdf5 *.h5 *.im *.nii *.nii.gz)"
        )
        if not fp:
            return
        try:
            self.image_path = Path(fp)
            self.last_dir = self.image_path.parent

            self.vol_xyz = load_volume_xyz(self.image_path).astype(np.float32)
            self.win_lo, self.win_hi = robust_window(self.vol_xyz)

            X, Y, Z = self.vol_xyz.shape
            self.sag_slider.setRange(0, X - 1)
            self.cor_slider.setRange(0, Y - 1)
            self.ax_slider.setRange(0, Z - 1)

            self.cursor_xyz = [X // 2, Y // 2, Z // 2]
            self.sag_slider.setValue(self.cursor_xyz[0])
            self.cor_slider.setValue(self.cursor_xyz[1])
            self.ax_slider.setValue(self.cursor_xyz[2])

            for s in (self.sag_slider, self.cor_slider, self.ax_slider):
                s.setEnabled(True)

            self._dragging_label = None
            for k in list(self._label_zoom.keys()):
                self._label_zoom[k] = 1.0

            self.statusBar().showMessage(f"Loaded {self.image_path.name} shape={self.vol_xyz.shape}")
            self._refresh_infer_action_state()
            self.render_all()
        except Exception as e:
            msg_error(self, "Open Image Error", f"{type(e).__name__}: {e}")

    def on_open_mask(self):
        if self.vol_xyz is None:
            msg_error(self, "Mask", "Open an image first.")
            return
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open Mask", str(self.last_dir),
            "Masks (*.npy *.hdf5 *.h5 *.im *.seg *.nii *.nii.gz)"
        )
        if not fp:
            return
        try:
            self._load_mask_into_viewer(Path(fp), kind="mask")
        except Exception as e:
            msg_error(self, "Open Mask Error", f"{type(e).__name__}: {e}")

    def on_open_gt(self):
        if self.vol_xyz is None:
            msg_error(self, "Ground Truth", "Open an image first.")
            return
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open Ground Truth", str(self.last_dir),
            "Ground Truth (*.npy *.hdf5 *.h5 *.im *.seg *.nii *.nii.gz)"
        )
        if not fp:
            return
        try:
            self._load_mask_into_viewer(Path(fp), kind="gt")
        except Exception as e:
            msg_error(self, "Open Ground Truth Error", f"{type(e).__name__}: {e}")

    def _load_mask_into_viewer(self, path: Path, kind: str):
        arr = load_mask_xyz(path)
        if self.vol_xyz is not None and arr.shape != self.vol_xyz.shape:
            raise RuntimeError(f"{kind} shape {arr.shape} != image shape {self.vol_xyz.shape}")

        self.last_dir = path.parent

        if kind == "gt":
            self.gt_path = path
            self.gt_xyz = arr
            self.statusBar().showMessage(f"Loaded GT: {path.name} labels={np.unique(arr)}")
        else:
            self.mask_path = path
            self.mask_xyz = arr
            self.mask_visible = True
            self.btn_toggle_mask.setEnabled(True)
            self.opacity.setEnabled(True)
            self.btn_toggle_mask.setText("Hide Mask")

            uniq = sorted(np.unique(self.mask_xyz).astype(int).tolist())
            self.label_filter.blockSignals(True)
            self.label_filter.setEnabled(True)
            self.label_filter.clear()
            self.label_filter.addItem("All labels", 0)
            for lid in uniq:
                if lid == 0:
                    continue
                self.label_filter.addItem(f"Label {lid}", lid)
            self.label_filter.setCurrentIndex(0)
            self.label_filter.blockSignals(False)

            self.statusBar().showMessage(f"Loaded mask: {path.name} labels={np.array(uniq)}")

        self._update_dice()
        self.render_all()

    def on_clear_gt(self):
        self.gt_path = None
        self.gt_xyz = None
        self.dice_label.setText("Dice: load GT + prediction to compute")
        self.statusBar().showMessage("Ground truth cleared.")
        self._update_dice()

    def on_clear_mask(self):
        self.mask_xyz = None
        self.mask_path = None
        self.mask_visible = False
        self.btn_toggle_mask.setEnabled(False)
        self.opacity.setEnabled(False)
        self.btn_toggle_mask.setText("Show Mask")
        self.label_filter.setEnabled(False)
        self.label_filter.clear()
        self.label_filter.addItem("All labels", 0)
        self.statusBar().showMessage("Mask cleared.")
        self._update_dice()
        self.render_all()

    def on_toggle_mask(self):
        if self.mask_xyz is None:
            return
        self.mask_visible = not self.mask_visible
        self.btn_toggle_mask.setText("Hide Mask" if self.mask_visible else "Show Mask")
        self.render_all()

    def on_auto_contrast(self):
        if self.vol_xyz is None:
            return
        self.win_lo, self.win_hi = robust_window(self.vol_xyz)
        self.statusBar().showMessage(f"Auto contrast set: lo={self.win_lo:.6g}, hi={self.win_hi:.6g}")
        self.render_all()

    # ---------- inference ----------
    def on_set_checkpoint(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", str(Path.cwd()),
            "Checkpoint (*.tar *.pth *.pt *.pth.tar)"
        )
        if not fp:
            return
        self.ckpt_path = Path(fp)
        self.statusBar().showMessage(f"Checkpoint set: {self.ckpt_path.name}")
        self._refresh_infer_action_state()

    def on_run_inference(self):
        if self.image_path is None or self.vol_xyz is None:
            msg_error(self, "Inference", "Open an image first.")
            return
        if self.ckpt_path is None or not self.ckpt_path.exists():
            msg_error(self, "Inference", "Checkpoint not set/found. Use Set Checkpoint…")
            return
        if not self.infer_script.exists():
            msg_error(self, "Inference", f"Missing script: {self.infer_script}")
            return

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{self.image_path.stem}_pred.hdf5"

        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

        cmd = [
            sys.executable, str(self.infer_script),
            "--ckpt", str(self.ckpt_path),
            "--im", str(self.image_path),
            "--out", str(out_path),
            "--device", device,
        ]

        self._set_busy(True, f"Running inference on {device}…")

        self._infer_thread = QtCore.QThread(self)
        self._infer_worker = InferWorker(cmd=cmd, cwd=Path.cwd(), out_path=out_path)
        self._infer_worker.moveToThread(self._infer_thread)

        self._infer_thread.started.connect(self._infer_worker.run)
        self._infer_worker.finished.connect(self._infer_thread.quit)
        self._infer_worker.failed.connect(self._infer_thread.quit)

        self._infer_worker.finished.connect(self._infer_worker.deleteLater)
        self._infer_worker.failed.connect(self._infer_worker.deleteLater)
        self._infer_thread.finished.connect(self._infer_thread.deleteLater)

        self._infer_worker.finished.connect(self._on_infer_done)
        self._infer_worker.failed.connect(self._on_infer_failed)

        self._infer_thread.start()

    def _on_infer_done(self, out_path_str: str):
        self._set_busy(False, "Inference complete.")
        out_path = Path(out_path_str)
        if not out_path.exists():
            msg_error(self, "Inference", f"Output not found: {out_path}")
            return
        try:
            self._load_mask_into_viewer(out_path, kind="mask")
            self.statusBar().showMessage(f"Inference complete — loaded {out_path.name}")
        except Exception as e:
            msg_error(self, "Inference Output Error", f"{type(e).__name__}: {e}")

    def _on_infer_failed(self, log: str):
        self._set_busy(False, "Inference failed.")
        msg_error(self, "Inference Failed", (log or "Unknown error")[-6000:])

    # ---------- dice ----------
    def _update_dice(self):
        if self.mask_xyz is None or self.gt_xyz is None:
            return
        try:
            rep, fg, mean_d = dice_report(self.mask_xyz.astype(np.uint8), self.gt_xyz.astype(np.uint8))
            # short summary in UI
            # FG = Foreground (any cartilage) and Mean (per cartilage label)
            if np.isnan(mean_d):
                short = f"FG Dice: {fg:.4f} | Mean Dice: N/A"
            else:
                short = f"FG Dice: {fg:.4f} | Mean Dice: {mean_d:.4f}"
            self.dice_label.setText(short)
            # keep full report in status bar (trimmed) for quick view
            self.statusBar().showMessage(short)
        except Exception as e:
            self.dice_label.setText(f"Dice error: {type(e).__name__}")
            self.statusBar().showMessage(f"Dice error: {type(e).__name__}: {e}")

    # ---------- label filter ----------
    def _selected_label_id(self) -> int:
        data = self.label_filter.currentData()
        try:
            return int(data)
        except Exception:
            return 0

    # ---------- slices ----------
    def get_slices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.vol_xyz is not None
        X, Y, Z = self.vol_xyz.shape
        x, y, z = self.cursor_xyz

        sag = self.vol_xyz[x, :, :]   # (Y,Z)
        cor = self.vol_xyz[:, y, :]   # (X,Z)
        ax  = self.vol_xyz[:, :, z]   # (X,Y)

        sag2 = np.flipud(sag)         # (Y,Z)
        cor2 = np.flipud(cor)         # (X,Z)
        ax2  = np.flipud(ax.T)        # (Y,X)

        return sag2, cor2, ax2

    def get_mask_slices(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if self.mask_xyz is None or not self.mask_visible:
            return None, None, None

        X, Y, Z = self.mask_xyz.shape
        x, y, z = self.cursor_xyz

        sag = self.mask_xyz[x, :, :]   # (Y,Z)
        cor = self.mask_xyz[:, y, :]   # (X,Z)
        ax  = self.mask_xyz[:, :, z]   # (X,Y)

        sag2 = np.flipud(sag)
        cor2 = np.flipud(cor)
        ax2  = np.flipud(ax.T)

        solo = self._selected_label_id()
        if solo != 0:
            sag2 = np.where(sag2 == solo, sag2, 0)
            cor2 = np.where(cor2 == solo, cor2, 0)
            ax2  = np.where(ax2 == solo, ax2, 0)

        return sag2, cor2, ax2

    # ---------- crosshair ----------
    def _crosshair_slice_coords(self) -> Dict[QLabel, Tuple[int, int]]:
        assert self.vol_xyz is not None
        X, Y, Z = self.vol_xyz.shape
        x, y, z = self.cursor_xyz

        sag_rc = (int((Y - 1) - y), int(z))  # (Y,Z)
        cor_rc = (int((X - 1) - x), int(z))  # (X,Z)
        ax_rc  = (int((Y - 1) - y), int(x))  # (Y,X)

        return {self.sag_label: sag_rc, self.cor_label: cor_rc, self.ax_label: ax_rc}

    def _begin_drag_on_label(self, label: QLabel, lx: float, ly: float) -> bool:
        rowcol = self._label_pos_to_rowcol(label, lx, ly, clamp=False)
        if rowcol is None:
            return False
        self._dragging_label = label
        label.setCursor(Qt.ClosedHandCursor)
        self._apply_rowcol_to_cursor(label, *rowcol)
        return True

    def _handle_drag_on_label(self, label: QLabel, lx: float, ly: float):
        rowcol = self._label_pos_to_rowcol(label, lx, ly, clamp=True)
        if rowcol is None:
            return
        self._apply_rowcol_to_cursor(label, *rowcol)

    def _end_drag_on_label(self, label: QLabel):
        self._dragging_label = None
        label.setCursor(Qt.CrossCursor)

    def _label_pos_to_rowcol(self, label: QLabel, lx: float, ly: float, clamp: bool = False) -> Optional[Tuple[int, int]]:
        if self.vol_xyz is None or label not in self._label_xform:
            return None

        H, W = self._slice_hw.get(label, (0, 0))
        if H <= 0 or W <= 0:
            return None

        xf = self._label_xform[label]
        scale = float(xf["scale"])
        mode = int(xf["mode"])

        if mode == 0:
            offx = float(xf["offx"])
            offy = float(xf["offy"])
            draw_w = float(xf["draw_w"])
            draw_h = float(xf["draw_h"])

            if clamp:
                eps = 1e-6
                px = min(max(lx, offx), offx + max(draw_w - eps, 0.0))
                py = min(max(ly, offy), offy + max(draw_h - eps, 0.0))
            else:
                if lx < offx or ly < offy or lx >= (offx + draw_w) or ly >= (offy + draw_h):
                    return None
                px = lx
                py = ly

            col = (px - offx) / scale
            row = (py - offy) / scale
        else:
            cropx = float(xf["cropx"])
            cropy = float(xf["cropy"])
            if clamp:
                eps = 1e-6
                px = min(max(lx, 0.0), max(label.width() - eps, 0.0))
                py = min(max(ly, 0.0), max(label.height() - eps, 0.0))
            else:
                if lx < 0.0 or ly < 0.0 or lx >= label.width() or ly >= label.height():
                    return None
                px = lx
                py = ly

            col = (px + cropx) / scale
            row = (py + cropy) / scale

        row_i = int(round(row))
        col_i = int(round(col))

        if clamp:
            row_i = max(0, min(H - 1, row_i))
            col_i = max(0, min(W - 1, col_i))
        else:
            if row_i < 0 or row_i >= H or col_i < 0 or col_i >= W:
                return None

        return row_i, col_i

    def _apply_rowcol_to_cursor(self, label: QLabel, row_i: int, col_i: int):
        if self.vol_xyz is None:
            return

        X, Y, Z = self.vol_xyz.shape
        cur_x, cur_y, cur_z = self.cursor_xyz

        if label is self.sag_label:
            new_y = (Y - 1) - row_i
            new_z = col_i
            new_x = cur_x
            self._set_cursor_and_sliders(new_x, new_y, new_z)
            return

        if label is self.cor_label:
            new_x = (X - 1) - row_i
            new_z = col_i
            new_y = cur_y
            self._set_cursor_and_sliders(new_x, new_y, new_z)
            return

        if label is self.ax_label:
            new_y = (Y - 1) - row_i
            new_x = col_i
            new_z = cur_z
            self._set_cursor_and_sliders(new_x, new_y, new_z)
            return

    def _set_cursor_and_sliders(self, x: int, y: int, z: int):
        if self.vol_xyz is None:
            return
        X, Y, Z = self.vol_xyz.shape
        x = max(0, min(X - 1, int(x)))
        y = max(0, min(Y - 1, int(y)))
        z = max(0, min(Z - 1, int(z)))

        if [x, y, z] == self.cursor_xyz:
            return

        self.cursor_xyz = [x, y, z]

        self.sag_slider.blockSignals(True)
        self.cor_slider.blockSignals(True)
        self.ax_slider.blockSignals(True)
        self.sag_slider.setValue(x)
        self.cor_slider.setValue(y)
        self.ax_slider.setValue(z)
        self.sag_slider.blockSignals(False)
        self.cor_slider.blockSignals(False)
        self.ax_slider.blockSignals(False)

        self.render_all()

    # ---------- info + render ----------
    def _update_info_labels(self):
        if self.vol_xyz is None:
            self.cursor_label.setText("Cursor (x,y,z): - , - , -")
            return

        X, Y, Z = self.vol_xyz.shape
        x, y, z = self.cursor_xyz
        self.cursor_label.setText(f"Cursor (x,y,z): {x} , {y} , {z}")

        zs = float(self._label_zoom.get(self.sag_label, 1.0))
        zc = float(self._label_zoom.get(self.cor_label, 1.0))
        za = float(self._label_zoom.get(self.ax_label, 1.0))

        self.sag_info.setText(f"X Slice: {x+1} / {X}   •   Zoom: {zs:.2f}×   •   Vol: {X}×{Y}×{Z}")
        self.cor_info.setText(f"Y Slice: {y+1} / {Y}   •   Zoom: {zc:.2f}×   •   Vol: {X}×{Y}×{Z}")
        self.ax_info.setText(f"Z Slice: {z+1} / {Z}   •   Zoom: {za:.2f}×   •   Vol: {X}×{Y}×{Z}")

    def render_all(self):
        if self.vol_xyz is None:
            return

        sag, cor, ax = self.get_slices()
        ms, mc, ma = self.get_mask_slices()

        self._slice_hw[self.sag_label] = sag.shape
        self._slice_hw[self.cor_label] = cor.shape
        self._slice_hw[self.ax_label]  = ax.shape

        sag8 = to_uint8(sag, self.win_lo, self.win_hi)
        cor8 = to_uint8(cor, self.win_lo, self.win_hi)
        ax8  = to_uint8(ax,  self.win_lo, self.win_hi)

        op = self.opacity.value()

        sag_img = blend_overlay(sag8, ms, op)
        cor_img = blend_overlay(cor8, mc, op)
        ax_img  = blend_overlay(ax8,  ma, op)

        cross_rc = self._crosshair_slice_coords()

        self._set_pix(self.sag_label, sag_img, cross_rc[self.sag_label])
        self._set_pix(self.cor_label, cor_img, cross_rc[self.cor_label])
        self._set_pix(self.ax_label,  ax_img,  cross_rc[self.ax_label])

        self._update_info_labels()

    def _set_pix(self, label: QLabel, qimg: QtGui.QImage, cross_rowcol: Tuple[int, int]):
        pm = QtGui.QPixmap.fromImage(qimg)

        lw, lh = label.width(), label.height()
        if lw <= 10 or lh <= 10:
            label.setPixmap(pm)
            return

        zoom = float(self._label_zoom.get(label, 1.0))

        img_w = qimg.width()
        img_h = qimg.height()

        fit_scale = min(lw / img_w, lh / img_h)

        if zoom <= 1.0001:
            scale = fit_scale * zoom
            draw_w = img_w * scale
            draw_h = img_h * scale
            offx = (lw - draw_w) / 2.0
            offy = (lh - draw_h) / 2.0

            pm2 = pm.scaled(int(round(draw_w)), int(round(draw_h)), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

            canvas = QtGui.QPixmap(lw, lh)
            canvas.fill(QtGui.QColor(0, 0, 0, 0))
            painter = QtGui.QPainter(canvas)
            painter.drawPixmap(int(round(offx)), int(round(offy)), pm2)

            row, col = cross_rowcol
            lx = offx + col * scale
            ly = offy + row * scale
            pen = QtGui.QPen(QtGui.QColor(80, 160, 255, 200))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(int(round(lx)), 0, int(round(lx)), lh)
            painter.drawLine(0, int(round(ly)), lw, int(round(ly)))

            painter.end()
            label.setPixmap(canvas)

            self._label_xform[label] = {
                "mode": 0.0,
                "scale": float(scale),
                "offx": float(offx),
                "offy": float(offy),
                "draw_w": float(draw_w),
                "draw_h": float(draw_h),
                "cropx": 0.0,
                "cropy": 0.0,
            }
            return

        scale = max(lw / img_w, lh / img_h) * zoom
        scaled_w = img_w * scale
        scaled_h = img_h * scale

        pm2 = pm.scaled(int(round(scaled_w)), int(round(scaled_h)), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        cropx = (scaled_w - lw) / 2.0
        cropy = (scaled_h - lh) / 2.0

        x0 = int(round(cropx))
        y0 = int(round(cropy))
        cropped = pm2.copy(x0, y0, lw, lh)

        row, col = cross_rowcol
        lx = col * scale - cropx
        ly = row * scale - cropy
        painter = QtGui.QPainter(cropped)
        pen = QtGui.QPen(QtGui.QColor(80, 160, 255, 200))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(int(round(lx)), 0, int(round(lx)), lh)
        painter.drawLine(0, int(round(ly)), lw, int(round(ly)))
        painter.end()

        label.setPixmap(cropped)

        self._label_xform[label] = {
            "mode": 1.0,
            "scale": float(scale),
            "offx": 0.0,
            "offy": 0.0,
            "draw_w": float(scaled_w),
            "draw_h": float(scaled_h),
            "cropx": float(cropx),
            "cropy": float(cropy),
        }


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
