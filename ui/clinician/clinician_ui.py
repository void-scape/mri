from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QCheckBox, QSlider, QComboBox, QLineEdit, QGroupBox,
    QFormLayout, QFrame, QTextEdit
)

try:
    from ui.clinician.constants import LABELS  # noqa: F401
except Exception:
    LABELS = {"BACKGROUND": 0, "FEMUR_BONE": 1, "FEMUR_CART": 2, "TIBIA_BONE": 3, "TIBIA_CART": 4}

class Divider(QFrame):
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine if orientation == Qt.Horizontal else QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)

def make_placeholder(title: str) -> QWidget:
    box = QGroupBox(title)
    box.setStyleSheet("QGroupBox{font-weight:600;}")
    layout = QVBoxLayout(box)
    area = QLabel(f"{title} View")
    area.setMinimumSize(240, 200)
    area.setAlignment(Qt.AlignCenter)
    area.setStyleSheet("border: 1px dashed #999; color:#666;")
    layout.addWidget(area)
    return box

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cartilage Review —  UI (no logic)")
        self.resize(1280, 780)

        # ===== Toolbar =====
        tb = self.addToolBar("Main")
        self.act_open_case = tb.addAction("Open Case")
        self.act_open_imgs = tb.addAction("Open Image/Mask")
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Model: "))
        self.model_combo = QComboBox(); self.model_combo.addItems(["V-Net", "U-Net"])
        tb.addWidget(self.model_combo)
        tb.addSeparator()
        self.btn_run = QPushButton("Run Inference"); tb.addWidget(self.btn_run)
        tb.addSeparator()
        self.btn_reset = QPushButton("Reset UI"); tb.addWidget(self.btn_reset)

        # ===== Left: Overlay Controls =====
        left_box = QGroupBox("Overlay Controls")
        left_layout = QFormLayout(left_box)
        self.opacity = QSlider(Qt.Horizontal); self.opacity.setRange(0, 100); self.opacity.setValue(60)
        self.render_mode = QComboBox(); self.render_mode.addItems(["Fill", "Contour"])
        self.cb_femur = QCheckBox("Femur cartilage (label 2)"); self.cb_femur.setChecked(True)
        self.cb_tibia = QCheckBox("Tibia cartilage (label 4)"); self.cb_tibia.setChecked(True)
        self.cb_bg    = QCheckBox("Show background (label 0)")
        left_layout.addRow("Opacity", self.opacity)
        left_layout.addRow("Render mode", self.render_mode)
        left_layout.addRow(self.cb_femur)
        left_layout.addRow(self.cb_tibia)
        left_layout.addRow(self.cb_bg)
        left_buttons = QWidget(); lb_l = QHBoxLayout(left_buttons); lb_l.setContentsMargins(0,0,0,0)
        self.btn_reset_view = QPushButton("Reset View"); self.btn_screenshot = QPushButton("Snapshot")
        lb_l.addWidget(self.btn_reset_view); lb_l.addWidget(self.btn_screenshot)
        left_layout.addRow(left_buttons)

        # ===== Center: Viewer (3 panes + sliders) =====
        center = QWidget(); center_layout = QVBoxLayout(center)
        views_row = QWidget(); views_grid = QGridLayout(views_row)
        self.sag_box = make_placeholder("Sagittal")
        self.cor_box = make_placeholder("Coronal")
        self.ax_box  = make_placeholder("Axial")
        views_grid.addWidget(self.sag_box, 0, 0)
        views_grid.addWidget(self.cor_box, 0, 1)
        views_grid.addWidget(self.ax_box,  0, 2)
        center_layout.addWidget(views_row)

        # Slice sliders row
        sliders_row = QWidget(); sliders_layout = QGridLayout(sliders_row)
        self.sag_slider = QSlider(Qt.Horizontal); self.sag_slider.setRange(0, 100); self.sag_slider.setValue(50)
        self.cor_slider = QSlider(Qt.Horizontal); self.cor_slider.setRange(0, 100); self.cor_slider.setValue(50)
        self.ax_slider  = QSlider(Qt.Horizontal); self.ax_slider.setRange(0, 100); self.ax_slider.setValue(50)
        sliders_layout.addWidget(QLabel("Sagittal"), 0, 0); sliders_layout.addWidget(self.sag_slider, 0, 1)
        sliders_layout.addWidget(QLabel("Coronal"),  1, 0); sliders_layout.addWidget(self.cor_slider, 1, 1)
        sliders_layout.addWidget(QLabel("Axial"),    2, 0); sliders_layout.addWidget(self.ax_slider,  2, 1)
        self.cb_link_views = QCheckBox("Link views")
        sliders_layout.addWidget(self.cb_link_views, 3, 1, 1, 1, alignment=Qt.AlignRight)
        center_layout.addWidget(sliders_row)

        # ===== Right: Metrics + QC =====
        right_col = QWidget(); right_layout = QVBoxLayout(right_col)
        metrics_box = QGroupBox("Metrics"); m_layout = QVBoxLayout(metrics_box)
        self.lbl_femur = QLabel("Femur cartilage: —")
        self.lbl_tibia = QLabel("Tibia cartilage: —")
        self.lbl_comb  = QLabel("Combined: —")
        for w in (self.lbl_femur, self.lbl_tibia, self.lbl_comb):
            w.setStyleSheet("color:#333;")
            m_layout.addWidget(w)
        m_layout.addWidget(Divider())
        self.txt_flags = QTextEdit(); self.txt_flags.setPlaceholderText("Auto-QC flags…"); self.txt_flags.setReadOnly(True); self.txt_flags.setFixedHeight(80)
        m_layout.addWidget(QLabel("Auto-QC flags:")); m_layout.addWidget(self.txt_flags)

        qc_box = QGroupBox("QC Checklist"); qc_layout = QVBoxLayout(qc_box)
        self.qc_align  = QCheckBox("Alignment & orientation look correct")
        self.qc_labels = QCheckBox("Labels within {0..4} and not empty")
        self.qc_leaks  = QCheckBox("No gross leaks outside knee ROI")
        self.qc_vols   = QCheckBox("Volumes plausible (see badges)")
        self.qc_spot   = QCheckBox("Visual spot-check complete")
        for cb in (self.qc_align, self.qc_labels, self.qc_leaks, self.qc_vols, self.qc_spot):
            qc_layout.addWidget(cb)
        self.qc_note = QLineEdit(); self.qc_note.setPlaceholderText("Reviewer note (optional)")
        qc_layout.addWidget(self.qc_note)
        qc_btn_row = QWidget(); ql = QHBoxLayout(qc_btn_row); ql.setContentsMargins(0,0,0,0)
        self.btn_approve = QPushButton("Approve & Export"); self.btn_reject  = QPushButton("Reject")
        ql.addWidget(self.btn_approve); ql.addWidget(self.btn_reject)
        qc_layout.addWidget(qc_btn_row)

        right_layout.addWidget(metrics_box)
        right_layout.addWidget(qc_box)

        # ===== Main 3-column layout =====
        main = QWidget(); main_layout = QHBoxLayout(main)
        main_layout.addWidget(left_box, 2)
        main_layout.addWidget(center, 6)
        main_layout.addWidget(right_col, 3)
        self.setCentralWidget(main)

        # ===== Status bar & style =====
        self.statusBar().showMessage("UI loaded. No actions implemented.")
        self.setStyleSheet("""
            QToolBar { spacing: 8px; }
            QPushButton { padding: 6px 10px; }
            QGroupBox { margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """)

        # ===== NO-OP connections =====
        def noop(*args, **kwargs): pass
        for btn in (self.btn_run, self.btn_reset, self.btn_reset_view, self.btn_screenshot, self.btn_approve, self.btn_reject):
            btn.clicked.connect(noop)
        self.act_open_case.triggered.connect(noop)
        self.act_open_imgs.triggered.connect(noop)
        for s in (self.sag_slider, self.cor_slider, self.ax_slider):
            s.valueChanged.connect(noop)
        for cb in (self.cb_femur, self.cb_tibia, self.cb_bg, self.cb_link_views,
                   self.qc_align, self.qc_labels, self.qc_leaks, self.qc_vols, self.qc_spot):
            cb.stateChanged.connect(noop)
        self.opacity.valueChanged.connect(noop)
        self.render_mode.currentIndexChanged.connect(noop)
        self.model_combo.currentIndexChanged.connect(noop)

def main():
    app = QApplication([])
    w = MainWindow(); w.show()
    app.exec()

if __name__ == "__main__":
    main()
