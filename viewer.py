import sys
import subprocess
import numpy as np
import h5py
import SimpleITK as sitk
from pathlib import Path
from typing import Optional, Tuple, Dict
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QSlider,
    QGroupBox,
    QFormLayout,
    QFrame,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QComboBox,
)


def to_binary_cartilage_mask(mask: np.ndarray) -> np.ndarray:
    """
    Converts any supported mask representation into binary cartilage labels:
      0 = no cartilage / background
      1 = cartilage

    For old multi-label ground truth files, every nonzero label is treated as cartilage.
    For one-hot/stacked .seg masks, any foreground channel being active becomes cartilage.
    """
    m = np.asarray(mask)

    if m.ndim == 3:
        return (m > 0).astype(np.uint8)

    if m.ndim == 4:
        # Most .seg files are channel-last: (X, Y, Z, C).
        # Some tools may save channel-first: (C, X, Y, Z).
        if m.shape[-1] <= 32:
            return np.any(m > 0, axis=-1).astype(np.uint8)
        if m.shape[0] <= 32:
            return np.any(m > 0, axis=0).astype(np.uint8)

    raise ValueError(
        f"Mask must be 3D labels or 4D one-hot/stacked channels. Got shape={m.shape}"
    )


"""
UI Helpers
"""


class Divider(QFrame):
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.setFrameShape(
            QFrame.HLine if orientation == Qt.Horizontal else QFrame.VLine
        )
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


"""
IO
"""


def is_hdf5(path: Path) -> bool:
    return path.suffix.lower() in {".h5", ".hdf5", ".im", ".seg"}


def load_hdf5_array(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        key = "data"
        if key not in f:
            raise KeyError(f"{path} does not contain `{key}` key")
        arr = np.array(f[key])
    return arr


def load_hdf5_volume_xyz(path: Path) -> np.ndarray:
    arr = load_hdf5_array(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3 dimensions, got shape={arr.shape}")
    return arr


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def load_nifti_volume_xyz(path: Path) -> np.ndarray:
    """
    Reads NIfTI and returns volume as (X,Y,Z).
    NOTE: NIfTI orientation conventions vary; this is best-effort for interoperability.
    Primary supported workflow is HDF5(.im/.hdf5) + .npy.
    """
    img = sitk.ReadImage(str(path))
    vol_zyx = sitk.GetArrayFromImage(img)  # (Z,Y,X)
    if vol_zyx.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape={vol_zyx.shape}")
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0))  # (X,Y,Z)
    return vol_xyz


def load_volume_xyz(path: Path) -> np.ndarray:
    if is_hdf5(path):
        return load_hdf5_volume_xyz(path)
    if is_nifti(path):
        return load_nifti_volume_xyz(path)
    raise ValueError(f"Unsupported image format: {path.name}")


def load_mask_xyz(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        m = np.load(path)
        return to_binary_cartilage_mask(m)

    if is_nifti(path):
        return to_binary_cartilage_mask(load_nifti_volume_xyz(path))

    if is_hdf5(path):
        m = load_hdf5_array(path)
        return to_binary_cartilage_mask(m)

    raise ValueError(f"Unsupported mask format: {path.name}")


"""
Dice evaluation
"""


def dice_report(pred: np.ndarray, gt: np.ndarray) -> Tuple[str, float, float]:
    """
    Returns (multi-line report, cartilage_dice, cartilage_dice).
    Binary project mode: prediction and GT are compared as cartilage vs no cartilage.
    Any nonzero label in either volume is treated as cartilage.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    pred_bin = pred > 0
    gt_bin = gt > 0

    pred_vox = pred_bin.sum()
    gt_vox = gt_bin.sum()
    inter = (pred_bin * gt_bin).sum()
    fp = (pred_bin * ~gt_bin).sum()
    fn = (~pred_bin * gt_bin).sum()
    dice = (2.0 * inter + 1e-4) / (pred_vox + gt_vox + 1e-4)

    lines = [
        f"Cartilage DSC: {dice:.4f}",
        "",
        "Binary comparison mode:",
        "  0 = no cartilage / background",
        "  1 = cartilage",
        "  any nonzero GT or prediction label is treated as cartilage",
        "",
        f"Prediction cartilage voxels: {pred_vox}",
        f"Ground truth cartilage voxels: {gt_vox}",
        f"Overlap voxels: {inter}",
        f"False positive voxels: {fp}",
        f"False negative voxels: {fn}",
        "",
        f"Prediction unique labels after binary load: {sorted(np.unique(pred).astype(int).tolist())}",
        f"Ground truth unique labels after binary load: {sorted(np.unique(gt).astype(int).tolist())}",
    ]

    return "\n".join(lines), float(dice), float(dice)


"""
Rendering Helpers
"""


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
    return np.array(
        [
            # 0 background/no cartilage
            [0, 0, 0, 0],
            # 1 cartilage
            [0, 255, 0, 255],
        ],
        dtype=np.uint8,
    )


def blend_overlay(
    gray8: np.ndarray, labels2d: Optional[np.ndarray], opacity_0_100: int
) -> QtGui.QImage:
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


"""
Inference worker
"""


class InferWorker(QtCore.QObject):
    # output path
    finished = QtCore.Signal(str)
    # error log
    failed = QtCore.Signal(str)

    def __init__(self, cmd: list[str], cwd: Path, out_path: Path):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.out_path = out_path

    @QtCore.Slot()
    def run(self):
        try:
            p = subprocess.run(
                self.cmd, cwd=str(self.cwd), capture_output=True, text=True
            )
            if p.returncode != 0:
                log = (p.stdout or "") + "\n" + (p.stderr or "")
                self.failed.emit(
                    log.strip() or f"Process failed with return code {p.returncode}"
                )
                return
            self.finished.emit(str(self.out_path))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cartilage Viewer")
        self.resize(1400, 960)

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
        self.ckpt_path: Optional[Path] = Path("checkpoints") / "vnet_model_best.pth.tar"
        if not self.ckpt_path.exists():
            self.ckpt_path = None
        self.infer_script = Path("infer.py")

        # Actions
        self.act_open_img = QtGui.QAction("Open Image…", self)
        self.act_open_mask = QtGui.QAction("Open Mask…", self)
        self.act_open_gt = QtGui.QAction("Open Ground Truth…", self)
        self.act_set_ckpt = QtGui.QAction("Set Checkpoint…", self)
        self.act_run_infer = QtGui.QAction("Run Inference", self)

        # Left controls
        left_box = QGroupBox("Controls")
        left_layout = QVBoxLayout(left_box)
        left_layout.setSpacing(10)

        self.btn_open_img = QPushButton("Open Image…")
        self.btn_open_mask = QPushButton("Open Mask…")
        self.btn_open_gt = QPushButton("Open Ground Truth…")
        self.btn_set_ckpt = QPushButton("Set Checkpoint…")
        self.btn_run_infer = QPushButton("Run Inference")
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["V-Net", "Triplanar"])

        self._action_buttons = [
            self.btn_open_img,
            self.btn_open_mask,
            self.btn_open_gt,
            self.btn_set_ckpt,
            self.btn_run_infer,
        ]
        for btn in self._action_buttons:
            btn.setMinimumHeight(34)
            left_layout.addWidget(btn)
        left_layout.addWidget(self.arch_combo)

        left_layout.addWidget(Divider())

        overlay_box = QGroupBox("Overlay / Info")
        overlay_form = QFormLayout(overlay_box)

        self.btn_toggle_mask = QPushButton("Show Mask")
        self.btn_toggle_mask.setEnabled(False)

        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(60)
        self.opacity.setEnabled(False)

        self.cursor_label = QLabel("Cursor: - , - , -")
        self.cursor_label.setStyleSheet("color:#bbb; font-size:11px;")

        self.ckpt_label = QLabel("Checkpoint: not set")
        self.ckpt_label.setWordWrap(True)
        self.ckpt_label.setStyleSheet("color:#bbb; font-size:11px;")

        overlay_form.addRow(self.btn_toggle_mask)
        overlay_form.addRow("Opacity", self.opacity)
        overlay_form.addRow(Divider())
        overlay_form.addRow(self.cursor_label)
        overlay_form.addRow(self.ckpt_label)

        left_layout.addWidget(overlay_box)
        left_layout.addStretch(1)

        # Right column
        center = QWidget()
        center_layout = QVBoxLayout(center)

        views_row = QWidget()
        views_grid = QGridLayout(views_row)
        views_grid.setContentsMargins(8, 8, 8, 8)
        views_grid.setHorizontalSpacing(16)
        views_grid.setVerticalSpacing(16)

        self.sag_label = make_view_label()
        self.cor_label = make_view_label()
        self.ax_label = make_view_label()

        self.sag_info = make_info_label()
        self.cor_info = make_info_label()
        self.ax_info = make_info_label()

        self.sag_box = QGroupBox("Axial")
        self.cor_box = QGroupBox("Coronal")
        self.ax_box = QGroupBox("Sagittal")
        self.dsc_box = QGroupBox("DSC vs Ground Truth")

        views = [
            (self.sag_box, self.sag_label, self.sag_info),
            (self.cor_box, self.cor_label, self.cor_info),
            (self.ax_box, self.ax_label, self.ax_info),
        ]
        self.sag_panel = QWidget()
        self.cor_panel = QWidget()
        self.ax_panel = QWidget()
        panels = [self.sag_panel, self.cor_panel, self.ax_panel]
        for panel, (box, img_lbl, info_lbl) in zip(panels, views):
            box.setStyleSheet("QGroupBox{font-weight:600;}")
            box_lay = QVBoxLayout(box)
            box_lay.setContentsMargins(8, 16, 8, 8)
            box_lay.addWidget(img_lbl, stretch=1)

            panel_lay = QVBoxLayout(panel)
            panel_lay.setContentsMargins(0, 0, 0, 0)
            panel_lay.setSpacing(4)
            panel_lay.addWidget(box, stretch=1)
            panel_lay.addWidget(info_lbl, stretch=0)

        dsc_lay = QVBoxLayout(self.dsc_box)
        dsc_lay.setContentsMargins(10, 16, 10, 10)
        self.dice_summary = QLabel("Cartilage DSC: —")
        self.dice_summary.setWordWrap(True)
        self.dice_summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.dice_summary.setStyleSheet("color:#ddd; font-size:18px; font-weight:600;")
        dsc_lay.addWidget(self.dice_summary, stretch=0)
        dsc_lay.addStretch(1)

        views_grid.addWidget(self.sag_panel, 0, 0)
        views_grid.addWidget(self.cor_panel, 0, 1)
        views_grid.addWidget(self.ax_panel, 1, 0)
        views_grid.addWidget(self.dsc_box, 1, 1)
        views_grid.setColumnStretch(0, 1)
        views_grid.setColumnStretch(1, 1)
        views_grid.setRowStretch(0, 1)
        views_grid.setRowStretch(1, 1)

        center_layout.addWidget(views_row, stretch=10)

        # Keep the slice sliders alive for wheel/crosshair logic, but hide their
        # container from the UI so the view panels can use the extra space.
        self.sliders_container = QWidget(center)
        g = QGridLayout(self.sliders_container)
        g.setColumnStretch(1, 1)

        self.sag_slider = QSlider(Qt.Horizontal, self.sliders_container)
        self.cor_slider = QSlider(Qt.Horizontal, self.sliders_container)
        self.ax_slider = QSlider(Qt.Horizontal, self.sliders_container)

        for s in (self.sag_slider, self.cor_slider, self.ax_slider):
            s.setRange(0, 100)
            s.setValue(0)
            s.setEnabled(False)

        g.addWidget(QLabel("Sagittal", self.sliders_container), 0, 0)
        g.addWidget(self.sag_slider, 0, 1)
        g.addWidget(QLabel("Coronal", self.sliders_container), 1, 0)
        g.addWidget(self.cor_slider, 1, 1)
        g.addWidget(QLabel("Axial", self.sliders_container), 2, 0)
        g.addWidget(self.ax_slider, 2, 1)
        center_layout.addWidget(self.sliders_container, stretch=0)
        self.sliders_container.hide()

        # Main layout
        main = QWidget()
        h = QHBoxLayout(main)
        h.addWidget(left_box, 3)
        h.addWidget(center, 10)
        self.setCentralWidget(main)

        # Status bar + progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Open an image (.hdf5/.im or .nii.gz)")

        # Zoom state
        self._label_zoom = {
            self.sag_label: 1.0,
            self.cor_label: 1.0,
            self.ax_label: 1.0,
        }
        self._min_zoom = 0.5
        self._max_zoom = 8.0

        # Wiring
        self.act_open_img.triggered.connect(self.on_open_image)
        self.act_open_mask.triggered.connect(self.on_open_mask)
        self.act_open_gt.triggered.connect(self.on_open_gt)
        self.act_set_ckpt.triggered.connect(self.on_set_checkpoint)
        self.act_run_infer.triggered.connect(self.on_run_inference)

        self.btn_open_img.clicked.connect(self.on_open_image)
        self.btn_open_mask.clicked.connect(self.on_open_mask)
        self.btn_open_gt.clicked.connect(self.on_open_gt)
        self.btn_set_ckpt.clicked.connect(self.on_set_checkpoint)
        self.btn_run_infer.clicked.connect(self.on_run_inference)

        self.btn_toggle_mask.clicked.connect(self.on_toggle_mask)
        self.opacity.valueChanged.connect(lambda *_: self.render_all())

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
            self.ax_label: self.ax_slider,
        }

        self._infer_thread: Optional[QtCore.QThread] = None
        self._infer_worker: Optional[InferWorker] = None

        self._refresh_infer_action_state()

    def _refresh_infer_action_state(self):
        has_img = self.image_path is not None
        has_ckpt = self.ckpt_path is not None and self.ckpt_path.exists()
        has_script = self.infer_script.exists()
        enabled = has_img and has_ckpt and has_script
        self.act_run_infer.setEnabled(enabled)
        if hasattr(self, "btn_run_infer"):
            self.btn_run_infer.setEnabled(enabled)
        ckpt_text = (
            self.ckpt_path.name
            if (self.ckpt_path is not None and self.ckpt_path.exists())
            else "not set"
        )
        if hasattr(self, "ckpt_label"):
            self.ckpt_label.setText(f"Checkpoint: {ckpt_text}")

    def _set_busy(self, busy: bool, msg: str = ""):
        self.progress.setVisible(busy)
        for a in (
            self.act_open_img,
            self.act_open_mask,
            self.act_open_gt,
            self.act_set_ckpt,
            self.act_run_infer,
        ):
            a.setEnabled(not busy)
        for btn in getattr(self, "_action_buttons", []):
            btn.setEnabled(not busy)
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
        if event.type() == QtCore.QEvent.MouseButtonDblClick and obj in getattr(
            self, "_label_zoom", {}
        ):
            self._label_zoom[obj] = 1.0
            self.render_all()
            event.accept()
            return True

        if event.type() == QtCore.QEvent.MouseButtonPress and obj in (
            self.sag_label,
            self.cor_label,
            self.ax_label,
        ):
            if event.button() == Qt.LeftButton:
                started = self._begin_drag_on_label(
                    obj, event.position().x(), event.position().y()
                )
                if started:
                    event.accept()
                    return True

        if event.type() == QtCore.QEvent.MouseMove and obj in (
            self.sag_label,
            self.cor_label,
            self.ax_label,
        ):
            if self._dragging_label is obj and (event.buttons() & Qt.LeftButton):
                self._handle_drag_on_label(
                    obj, event.position().x(), event.position().y()
                )
                event.accept()
                return True

        if event.type() == QtCore.QEvent.MouseButtonRelease and obj in (
            self.sag_label,
            self.cor_label,
            self.ax_label,
        ):
            if event.button() == Qt.LeftButton and self._dragging_label is obj:
                self._end_drag_on_label(obj)
                event.accept()
                return True

        if event.type() == QtCore.QEvent.Wheel and obj in getattr(
            self, "_label_to_slider", {}
        ):
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
                z *= 1.10**steps
                z = max(self._min_zoom, min(self._max_zoom, z))
                self._label_zoom[obj] = z
                self.render_all()
                event.accept()
                return True

            if event.modifiers() & Qt.ShiftModifier:
                steps *= 10

            new_val = max(
                slider.minimum(), min(slider.maximum(), slider.value() + steps)
            )
            slider.setValue(new_val)

            event.accept()
            return True

        if event.type() == QtCore.QEvent.Resize and obj in (
            self.sag_label,
            self.cor_label,
            self.ax_label,
        ):
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

    # file loading
    def on_open_image(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(self.last_dir),
            "Images (*.hdf5 *.h5 *.im *.nii *.nii.gz)",
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

            self.mask_path = None
            self.mask_xyz = None
            self.mask_visible = False
            self.btn_toggle_mask.setEnabled(False)
            self.btn_toggle_mask.setText("Show Mask")
            self.opacity.setEnabled(False)

            self.statusBar().showMessage(
                f"Loaded {self.image_path.name} shape={self.vol_xyz.shape}"
            )
            self._refresh_infer_action_state()
            self.render_all()
        except Exception as e:
            msg_error(self, "Open Image Error", f"{type(e).__name__}: {e}")

    def on_open_mask(self):
        if self.vol_xyz is None:
            msg_error(self, "Mask", "Open an image first.")
            return
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Open Mask",
            str(self.last_dir),
            "Masks (*.npy *.hdf5 *.h5 *.im *.seg *.nii *.nii.gz)",
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
            self,
            "Open Ground Truth",
            str(self.last_dir),
            "Ground Truth (*.npy *.hdf5 *.h5 *.im *.seg *.nii *.nii.gz)",
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
            raise RuntimeError(
                f"{kind} shape {arr.shape} != image shape {self.vol_xyz.shape}"
            )

        self.last_dir = path.parent

        if kind == "gt":
            self.gt_path = path
            self.gt_xyz = arr
            self.statusBar().showMessage(
                f"Loaded GT: {path.name} binary_labels={np.unique(arr)}"
            )
        else:
            self.mask_path = path
            self.mask_xyz = arr
            self.mask_visible = True
            self.btn_toggle_mask.setEnabled(True)
            self.opacity.setEnabled(True)
            self.btn_toggle_mask.setText("Hide Mask")

            uniq = sorted(np.unique(self.mask_xyz).astype(int).tolist())
            self.statusBar().showMessage(
                f"Loaded mask: {path.name} binary_labels={np.array(uniq)}"
            )

        self._update_dice()
        self.render_all()

    def on_toggle_mask(self):
        if self.mask_xyz is None:
            return
        self.mask_visible = not self.mask_visible
        self.btn_toggle_mask.setText("Hide Mask" if self.mask_visible else "Show Mask")
        self.render_all()

    # inference
    def on_set_checkpoint(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Select checkpoint",
            str(Path.cwd()),
            "Checkpoint (*.tar *.pth *.pt *.pth.tar)",
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
            msg_error(
                self, "Inference", "Checkpoint not set/found. Use Set Checkpoint…"
            )
            return
        if not self.infer_script.exists():
            msg_error(self, "Inference", f"Missing script: {self.infer_script}")
            return

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{self.image_path.stem}_pred.hdf5"

        device = "cpu"
        # import torch
        # if torch.cuda.is_available():
        #     device = "cuda"
        arch = "vnet" if self.arch_combo.currentIndex() == 0 else "triplanar"

        cmd = [
            sys.executable,
            str(self.infer_script),
            "--checkpoint",
            str(self.ckpt_path),
            "--im",
            str(self.image_path),
            "--out",
            str(out_path),
            "--device",
            device,
            "--arch",
            arch,
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

    # dice
    def _update_dice(self):
        if self.mask_xyz is None or self.gt_xyz is None:
            self.dice_summary.setText("Cartilage DSC: —")
            return
        try:
            _rep, fg, _mean_d = dice_report(
                self.mask_xyz.astype(np.uint8), self.gt_xyz.astype(np.uint8)
            )
            short = f"Cartilage DSC: {fg:.4f}"
            self.dice_summary.setText(short)
            self.statusBar().showMessage(short)
        except Exception as e:
            self.dice_summary.setText(f"Dice error: {type(e).__name__}")
            self.statusBar().showMessage(f"Dice error: {type(e).__name__}: {e}")

    # slices
    def get_slices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.vol_xyz is not None
        X, Y, Z = self.vol_xyz.shape
        x, y, z = self.cursor_xyz

        sag = self.vol_xyz[x, :, :]  # (Y,Z)
        cor = self.vol_xyz[:, y, :]  # (X,Z)
        ax = self.vol_xyz[:, :, z]  # (X,Y)

        # Keep sagittal/coronal as the accepted stretched-strip views.
        sag2 = np.fliplr(np.flipud(sag))  # (Y,Z), mirrored left-right
        cor2 = np.flipud(cor)  # (X,Z)
        ax2 = np.fliplr(ax)  # (Y,X)

        return sag2, cor2, ax2

    def get_mask_slices(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if self.mask_xyz is None or not self.mask_visible:
            return None, None, None

        X, Y, Z = self.mask_xyz.shape
        x, y, z = self.cursor_xyz

        sag = self.mask_xyz[x, :, :]  # (Y,Z)
        cor = self.mask_xyz[:, y, :]  # (X,Z)
        ax = self.mask_xyz[:, :, z]  # (X,Y)

        # Use the exact same display transforms as the image slices so overlay shapes match.
        sag2 = np.fliplr(np.flipud(sag))
        cor2 = np.flipud(cor)
        ax2 = np.fliplr(ax)  # (Y,X)

        return sag2, cor2, ax2

    # crosshair
    def _crosshair_slice_coords(self) -> Dict[QLabel, Tuple[int, int]]:
        assert self.vol_xyz is not None
        X, Y, Z = self.vol_xyz.shape
        x, y, z = self.cursor_xyz

        sag_rc = (
            int((Y - 1) - y),
            int((Z - 1) - z),
        )  # sagittal display: flipud + fliplr(Y,Z)
        cor_rc = (int((X - 1) - x), int(z))  # coronal display:  flipud(X,Z)
        ax_rc = (int((Y - 1) - y), int(x))  # axial display:  flipud(Y,X)

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

    def _label_pos_to_rowcol(
        self, label: QLabel, lx: float, ly: float, clamp: bool = False
    ) -> Optional[Tuple[int, int]]:
        if self.vol_xyz is None or label not in self._label_xform:
            return None

        H, W = self._slice_hw.get(label, (0, 0))
        if H <= 0 or W <= 0:
            return None

        xf = self._label_xform[label]
        scale_x = float(xf.get("scale_x", xf.get("scale", 1.0)))
        scale_y = float(xf.get("scale_y", xf.get("scale", 1.0)))
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
                if (
                    lx < offx
                    or ly < offy
                    or lx >= (offx + draw_w)
                    or ly >= (offy + draw_h)
                ):
                    return None
                px = lx
                py = ly

            col = (px - offx) / scale_x
            row = (py - offy) / scale_y
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

            col = (px + cropx) / scale_x
            row = (py + cropy) / scale_y

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
            new_z = (Z - 1) - col_i
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

    # info + render
    def _update_info_labels(self):
        if self.vol_xyz is None:
            self.cursor_label.setText("Cursor: - , - , -")
            return

        X, Y, Z = self.vol_xyz.shape
        x, y, z = self.cursor_xyz
        self.cursor_label.setText(f"Cursor: {x} , {y} , {z}")

        zs = float(self._label_zoom.get(self.sag_label, 1.0))
        zc = float(self._label_zoom.get(self.cor_label, 1.0))
        za = float(self._label_zoom.get(self.ax_label, 1.0))

        self.sag_info.setText(
            f"Slice: {x + 1} / {X}   •   Zoom: {zs:.2f}×   •   Vol: {X}×{Y}×{Z}"
        )
        self.cor_info.setText(
            f"Slice: {y + 1} / {Y}   •   Zoom: {zc:.2f}×   •   Vol: {X}×{Y}×{Z}"
        )
        self.ax_info.setText(
            f"Slice: {z + 1} / {Z}   •   Zoom: {za:.2f}×   •   Vol: {X}×{Y}×{Z}"
        )

    def render_all(self):
        if self.vol_xyz is None:
            return

        sag, cor, ax = self.get_slices()
        ms, mc, ma = self.get_mask_slices()

        self._slice_hw[self.sag_label] = sag.shape
        self._slice_hw[self.cor_label] = cor.shape
        self._slice_hw[self.ax_label] = ax.shape

        sag8 = to_uint8(sag, self.win_lo, self.win_hi)
        cor8 = to_uint8(cor, self.win_lo, self.win_hi)
        ax8 = to_uint8(ax, self.win_lo, self.win_hi)

        op = self.opacity.value()

        sag_img = blend_overlay(sag8, ms, op)
        cor_img = blend_overlay(cor8, mc, op)
        ax_img = blend_overlay(ax8, ma, op)

        cross_rc = self._crosshair_slice_coords()

        self._set_pix(self.sag_label, sag_img, cross_rc[self.sag_label])
        self._set_pix(self.cor_label, cor_img, cross_rc[self.cor_label])
        self._set_pix(self.ax_label, ax_img, cross_rc[self.ax_label])

        self._update_info_labels()

    def _set_pix(
        self, label: QLabel, qimg: QtGui.QImage, cross_rowcol: Tuple[int, int]
    ):
        pm = QtGui.QPixmap.fromImage(qimg)

        lw, lh = label.width(), label.height()
        if lw <= 10 or lh <= 10:
            label.setPixmap(pm)
            return

        zoom = float(self._label_zoom.get(label, 1.0))

        img_w = qimg.width()
        img_h = qimg.height()

        fit_scale = min(lw / img_w, lh / img_h)

        stretch_x = 1.0
        if label in (self.sag_label, self.cor_label):
            aspect = img_w / max(img_h, 1)
            desired_fill = 0.62 if aspect < 0.22 else 0.50
            current_draw_w = img_w * fit_scale * zoom
            desired_draw_w = lw * desired_fill
            if current_draw_w < desired_draw_w:
                stretch_x = min(4.0, desired_draw_w / max(current_draw_w, 1e-6))

        if zoom <= 1.0001:
            scale_y = fit_scale * zoom
            scale_x = scale_y * stretch_x
            draw_w = img_w * scale_x
            draw_h = img_h * scale_y
            offx = (lw - draw_w) / 2.0
            up_shift = min(14.0, max(0.0, (lh - draw_h) * 0.15))
            offy = (lh - draw_h) / 2.0 - up_shift

            pm2 = pm.scaled(
                int(round(draw_w)),
                int(round(draw_h)),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )

            canvas = QtGui.QPixmap(lw, lh)
            canvas.fill(QtGui.QColor(0, 0, 0, 0))
            painter = QtGui.QPainter(canvas)
            painter.drawPixmap(int(round(offx)), int(round(offy)), pm2)

            row, col = cross_rowcol
            lx = offx + col * scale_x
            ly = offy + row * scale_y
            pen = QtGui.QPen(QtGui.QColor(80, 160, 255, 200))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(int(round(lx)), 0, int(round(lx)), lh)
            painter.drawLine(0, int(round(ly)), lw, int(round(ly)))

            painter.end()
            label.setPixmap(canvas)

            self._label_xform[label] = {
                "mode": 0.0,
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "offx": float(offx),
                "offy": float(offy),
                "draw_w": float(draw_w),
                "draw_h": float(draw_h),
                "cropx": 0.0,
                "cropy": 0.0,
            }
            return

        scale_y = max(lw / img_w, lh / img_h) * zoom
        scale_x = scale_y * stretch_x
        scaled_w = img_w * scale_x
        scaled_h = img_h * scale_y

        pm2 = pm.scaled(
            int(round(scaled_w)),
            int(round(scaled_h)),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )

        cropx = (scaled_w - lw) / 2.0
        cropy = (scaled_h - lh) / 2.0

        x0 = int(round(cropx))
        y0 = int(round(cropy))
        cropped = pm2.copy(x0, y0, lw, lh)

        row, col = cross_rowcol
        lx = col * scale_x - cropx
        ly = row * scale_y - cropy
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
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
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
