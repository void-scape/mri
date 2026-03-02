from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np

from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QFormLayout, QFrame, QFileDialog,
    QMessageBox, QComboBox
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
    return path.suffix.lower() in {".h5", ".hdf5", ".im"}


def load_hdf5_volume_xyz(path: Path, key: str = "data") -> np.ndarray:
    if h5py is None:
        raise RuntimeError("h5py is not installed. Install with: python3 -m pip install h5py")
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
        raise RuntimeError("SimpleITK is not installed. Install with: python3 -m pip install SimpleITK")
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


# ----------------- main window -----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cartilage Viewer")
        self.resize(1200, 820)

        self.image_path: Optional[Path] = None
        self.mask_path: Optional[Path] = None

        self.vol_xyz: Optional[np.ndarray] = None    # (X,Y,Z)
        self.mask_xyz: Optional[np.ndarray] = None   # (X,Y,Z) labels

        self.mask_visible = False
        self.win_lo = 0.0
        self.win_hi = 1.0

        # 3D cursor (x,y,z) in voxel coords (ITK-SNAP style)
        self.cursor_xyz = [0, 0, 0]

        # For click mapping and crosshair drawing
        self._label_xform: Dict[QLabel, Dict[str, float]] = {}
        self._slice_hw: Dict[QLabel, Tuple[int, int]] = {}  # (H,W) of the slice shown in that label

        # ===== Toolbar =====
        tb = self.addToolBar("Main")
        act_open_img = QtGui.QAction("Open Image…", self)
        act_open_mask = QtGui.QAction("Open Mask…", self)
        act_clear_mask = QtGui.QAction("Clear Mask", self)
        act_auto = QtGui.QAction("Auto Contrast", self)

        tb.addAction(act_open_img)
        tb.addAction(act_open_mask)
        tb.addAction(act_clear_mask)
        tb.addSeparator()
        tb.addAction(act_auto)

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

        hint = QLabel('Use "Show label" to isolate one label and identify it.')
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#777; font-size:11px;")

        left_layout.addRow(self.btn_toggle_mask)
        left_layout.addRow("Opacity", self.opacity)
        left_layout.addRow("Show label", self.label_filter)
        left_layout.addRow(Divider())
        left_layout.addRow(self.cursor_label)
        left_layout.addRow(hint)

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
        self.ax_info  = make_info_label()

        self.sag_box = QGroupBox("Sagittal")
        self.cor_box = QGroupBox("Coronal")
        self.ax_box  = QGroupBox("Axial")

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
        views_grid.addWidget(self.ax_box,  0, 2)
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

        self.statusBar().showMessage("Open an image (.hdf5/.im or .nii.gz)")

        # ===== Zoom state =====
        self._label_zoom = {
            self.sag_label: 1.0,
            self.cor_label: 1.0,
            self.ax_label:  1.0,
        }
        self._min_zoom = 0.5
        self._max_zoom = 8.0

        # ===== Wiring =====
        act_open_img.triggered.connect(self.on_open_image)
        act_open_mask.triggered.connect(self.on_open_mask)
        act_clear_mask.triggered.connect(self.on_clear_mask)
        act_auto.triggered.connect(self.on_auto_contrast)

        self.btn_toggle_mask.clicked.connect(self.on_toggle_mask)
        self.opacity.valueChanged.connect(lambda *_: self.render_all())
        self.label_filter.currentIndexChanged.connect(lambda *_: self.render_all())

        self.sag_slider.valueChanged.connect(self.on_x_changed)
        self.cor_slider.valueChanged.connect(self.on_y_changed)
        self.ax_slider.valueChanged.connect(self.on_z_changed)

        # Wheel / click events on panes
        for lbl in (self.sag_label, self.cor_label, self.ax_label):
            lbl.installEventFilter(self)
            lbl.setFocusPolicy(Qt.WheelFocus)
            lbl.setToolTip(
                "Wheel: change slice | Shift+Wheel: jump 10 | Ctrl+Wheel: zoom | "
                "Click: move crosshair | Double-click: reset zoom | Ctrl+0: reset all zoom"
            )

        self._label_to_slider = {
            self.sag_label: self.sag_slider,
            self.cor_label: self.cor_slider,
            self.ax_label:  self.ax_slider,
        }

    # Ctrl+0 resets zoom for all panes
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if (event.modifiers() & Qt.ControlModifier) and (event.key() == Qt.Key_0):
            for k in list(self._label_zoom.keys()):
                self._label_zoom[k] = 1.0
            self.render_all()
            event.accept()
            return
        super().keyPressEvent(event)

    # ----------------- Event filter: wheel + clicks -----------------
    def eventFilter(self, obj, event):
        # Double-click resets zoom for that pane
        if event.type() == QtCore.QEvent.MouseButtonDblClick and obj in getattr(self, "_label_zoom", {}):
            self._label_zoom[obj] = 1.0
            self.render_all()
            event.accept()
            return True

        # Click places crosshair cursor (updates sliders)
        if event.type() == QtCore.QEvent.MouseButtonPress and obj in (self.sag_label, self.cor_label, self.ax_label):
            if event.button() == Qt.LeftButton:
                self._handle_click_on_label(obj, event.position().x(), event.position().y())
                event.accept()
                return True

        # Wheel behavior: Ctrl=zoom, otherwise slice scroll (Shift=jump)
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

        # Re-render on resize so pixmaps scale cleanly
        if event.type() == QtCore.QEvent.Resize and obj in (self.sag_label, self.cor_label, self.ax_label):
            self.render_all()

        return super().eventFilter(obj, event)

    # ----------------- Cursor/slider sync -----------------
    def on_x_changed(self, v: int):
        self.cursor_xyz[0] = int(v)
        self.render_all()

    def on_y_changed(self, v: int):
        self.cursor_xyz[1] = int(v)
        self.render_all()

    def on_z_changed(self, v: int):
        self.cursor_xyz[2] = int(v)
        self.render_all()

    # ----------------- Actions -----------------
    def on_open_image(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.hdf5 *.h5 *.im *.nii *.nii.gz)"
        )
        if not fp:
            return

        try:
            self.image_path = Path(fp)
            self.vol_xyz = load_volume_xyz(self.image_path).astype(np.float32)
            self.win_lo, self.win_hi = robust_window(self.vol_xyz)

            X, Y, Z = self.vol_xyz.shape

            self.sag_slider.setRange(0, X - 1)
            self.cor_slider.setRange(0, Y - 1)
            self.ax_slider.setRange(0, Z - 1)

            # Center cursor
            self.cursor_xyz = [X // 2, Y // 2, Z // 2]
            self.sag_slider.setValue(self.cursor_xyz[0])
            self.cor_slider.setValue(self.cursor_xyz[1])
            self.ax_slider.setValue(self.cursor_xyz[2])

            for s in (self.sag_slider, self.cor_slider, self.ax_slider):
                s.setEnabled(True)

            # Reset zoom
            for k in list(self._label_zoom.keys()):
                self._label_zoom[k] = 1.0

            self.statusBar().showMessage(f"Loaded {self.image_path.name} shape={self.vol_xyz.shape}")
            self.render_all()

        except Exception as e:
            msg_error(self, "Open Image Error", f"{type(e).__name__}: {e}")

    def on_open_mask(self):
        if self.vol_xyz is None:
            msg_error(self, "Mask Error", "Open an image first.")
            return

        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Open Mask",
            "",
            "Masks (*.npy *.nii *.nii.gz *.hdf5 *.h5 *.im)"
        )
        if not fp:
            return

        try:
            self.mask_path = Path(fp)
            m = load_mask_xyz(self.mask_path)

            if m.shape != self.vol_xyz.shape:
                raise RuntimeError(f"Mask shape {m.shape} does not match image shape {self.vol_xyz.shape}")

            self.mask_xyz = m
            self.mask_visible = True
            self.btn_toggle_mask.setEnabled(True)
            self.opacity.setEnabled(True)
            self.btn_toggle_mask.setText("Hide Mask")

            # Build label filter options based on labels present
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

            self.statusBar().showMessage(f"Loaded mask {self.mask_path.name} labels={np.array(uniq)}")
            self.render_all()

        except Exception as e:
            msg_error(self, "Open Mask Error", f"{type(e).__name__}: {e}")

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

    # ----------------- Label filter -----------------
    def _selected_label_id(self) -> int:
        data = self.label_filter.currentData()
        try:
            return int(data)
        except Exception:
            return 0

    # ----------------- Slice extraction (NO padding) -----------------
    def get_slices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns 2D slices for (sagittal, coronal, axial) as (H,W) arrays.
        Data stored as (X,Y,Z).
        We use the same orientations you were happy with before:

          - sagittal: (Y,Z) flipped vertically  => rows=Y', cols=Z
          - coronal:  (X,Z) flipped vertically  => rows=X', cols=Z
          - axial:    (Y,X) flipped vertically  => rows=Y', cols=X
        """
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

    # ----------------- Crosshair pixel positions in slice coords -----------------
    def _crosshair_slice_coords(self) -> Dict[QLabel, Tuple[int, int]]:
        """
        Returns crosshair position in slice pixel coords (row, col) for each view,
        matching the display orientations in get_slices().
        """
        assert self.vol_xyz is not None
        X, Y, Z = self.vol_xyz.shape
        x, y, z = self.cursor_xyz

        # sag2: (Y,Z) with Y flipped => row = (Y-1-y), col = z
        sag_rc = (int((Y - 1) - y), int(z))
        # cor2: (X,Z) with X flipped => row = (X-1-x), col = z
        cor_rc = (int((X - 1) - x), int(z))
        # ax2: (Y,X) with Y flipped => row = (Y-1-y), col = x
        ax_rc  = (int((Y - 1) - y), int(x))

        return {
            self.sag_label: sag_rc,
            self.cor_label: cor_rc,
            self.ax_label:  ax_rc,
        }

    # ----------------- Click mapping: label coords -> voxel cursor -----------------
    def _handle_click_on_label(self, label: QLabel, lx: float, ly: float):
        if self.vol_xyz is None:
            return
        if label not in self._label_xform:
            return

        X, Y, Z = self.vol_xyz.shape

        xf = self._label_xform[label]
        scale = xf["scale"]
        mode = int(xf["mode"])

        if mode == 0:
            offx = xf["offx"]
            offy = xf["offy"]
            if lx < offx or ly < offy or lx >= (offx + xf["draw_w"]) or ly >= (offy + xf["draw_h"]):
                return
            col = (lx - offx) / scale
            row = (ly - offy) / scale
        else:
            cropx = xf["cropx"]
            cropy = xf["cropy"]
            col = (lx + cropx) / scale
            row = (ly + cropy) / scale

        row_i = int(round(row))
        col_i = int(round(col))

        H, W = self._slice_hw.get(label, (0, 0))
        if H <= 0 or W <= 0:
            return

        if row_i < 0 or row_i >= H or col_i < 0 or col_i >= W:
            return

        # Map slice pixel (row_i, col_i) back to voxel (x,y,z)
        cur_x, cur_y, cur_z = self.cursor_xyz

        if label is self.sag_label:
            # sag2 rows = Y' (flipped), cols = Z
            new_y = (Y - 1) - row_i
            new_z = col_i
            new_x = cur_x
            self._set_cursor_and_sliders(new_x, new_y, new_z)
            return

        if label is self.cor_label:
            # cor2 rows = X' (flipped), cols = Z
            new_x = (X - 1) - row_i
            new_z = col_i
            new_y = cur_y
            self._set_cursor_and_sliders(new_x, new_y, new_z)
            return

        if label is self.ax_label:
            # ax2 rows = Y' (flipped), cols = X
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

    # ----------------- Info labels -----------------
    def _update_info_labels(self):
        if self.vol_xyz is None:
            self.sag_info.setText("Slice: - / -   •   Zoom: 1.00×   •   Vol: -×-×-")
            self.cor_info.setText("Slice: - / -   •   Zoom: 1.00×   •   Vol: -×-×-")
            self.ax_info.setText("Slice: - / -   •   Zoom: 1.00×   •   Vol: -×-×-")
            self.cursor_label.setText("Cursor (x,y,z): - , - , -")
            return

        X, Y, Z = self.vol_xyz.shape
        x, y, z = self.cursor_xyz

        zs = float(self._label_zoom.get(self.sag_label, 1.0))
        zc = float(self._label_zoom.get(self.cor_label, 1.0))
        za = float(self._label_zoom.get(self.ax_label, 1.0))

        self.sag_info.setText(f"X Slice: {x+1} / {X}   •   Zoom: {zs:.2f}×   •   Vol: {X}×{Y}×{Z}")
        self.cor_info.setText(f"Y Slice: {y+1} / {Y}   •   Zoom: {zc:.2f}×   •   Vol: {X}×{Y}×{Z}")
        self.ax_info.setText(f"Z Slice: {z+1} / {Z}   •   Zoom: {za:.2f}×   •   Vol: {X}×{Y}×{Z}")

        self.cursor_label.setText(f"Cursor (x,y,z): {x} , {y} , {z}")

    # ----------------- Rendering -----------------
    def render_all(self):
        if self.vol_xyz is None:
            return

        sag, cor, ax = self.get_slices()
        ms, mc, ma = self.get_mask_slices()

        # store slice sizes for click mapping
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
        """
        Render the slice into the label with zoom, then draw crosshair.
        Stores label->slice transform for click mapping.
        """
        pm = QtGui.QPixmap.fromImage(qimg)

        lw, lh = label.width(), label.height()
        if lw <= 10 or lh <= 10:
            label.setPixmap(pm)
            return

        zoom = float(self._label_zoom.get(label, 1.0))

        img_w = qimg.width()
        img_h = qimg.height()

        # Fit scale for the slice inside the label
        fit_scale = min(lw / img_w, lh / img_h)

        # mode 0: zoom<=1 keep aspect (letterbox), mode 1: zoom>1 expand+crop
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

            # crosshair
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

        # zoom > 1: expand + crop to fill label (like your zoom behavior)
        scale = max(lw / img_w, lh / img_h) * zoom
        scaled_w = img_w * scale
        scaled_h = img_h * scale

        pm2 = pm.scaled(int(round(scaled_w)), int(round(scaled_h)), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        cropx = (scaled_w - lw) / 2.0
        cropy = (scaled_h - lh) / 2.0

        x0 = int(round(cropx))
        y0 = int(round(cropy))
        cropped = pm2.copy(x0, y0, lw, lh)

        # crosshair
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