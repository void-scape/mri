from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QCheckBox, QSlider, QComboBox, QGroupBox,
    QFormLayout, QFrame
)

# ---------------------------
# Dark/Light palette helpers
# ---------------------------
def apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window,        QtGui.QColor(53, 53, 53))
    pal.setColor(QtGui.QPalette.WindowText,    QtGui.QColor(220, 220, 220))
    pal.setColor(QtGui.QPalette.Base,          QtGui.QColor(35, 35, 35))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    pal.setColor(QtGui.QPalette.ToolTipBase,   QtGui.QColor(35, 35, 35))
    pal.setColor(QtGui.QPalette.ToolTipText,   QtGui.QColor(220, 220, 220))
    pal.setColor(QtGui.QPalette.Text,          QtGui.QColor(220, 220, 220))
    pal.setColor(QtGui.QPalette.Button,        QtGui.QColor(53, 53, 53))
    pal.setColor(QtGui.QPalette.ButtonText,    QtGui.QColor(220, 220, 220))
    pal.setColor(QtGui.QPalette.BrightText,    QtGui.QColor(255, 0, 0))
    pal.setColor(QtGui.QPalette.Link,          QtGui.QColor(42, 130, 218))
    pal.setColor(QtGui.QPalette.Highlight,     QtGui.QColor(42, 130, 218))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
    app.setPalette(pal)

def apply_light_palette(app: QApplication):
    app.setStyle("Fusion") 
    app.setPalette(QtGui.QPalette())  


# ---------------------------
# Small UI helpers
# ---------------------------
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
    area.setMinimumSize(320, 320)   
    area.setAlignment(Qt.AlignCenter)
    area.setStyleSheet("border: 1px dashed #999; color:#666;")
    layout.addWidget(area)
    return box


# ---------------------------
# Main Window
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cartilage Review — Skeleton UI (INITIAL VERSION)")
        self.resize(1400, 820)

        # ===== Toolbar =====
        tb = self.addToolBar("Main")
        self.act_open_case = tb.addAction("Open Case")
        self.act_open_imgs = tb.addAction("Open Image/Mask")
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Model: "))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["V-Net", "U-Net"])
        tb.addWidget(self.model_combo)
        tb.addSeparator()
        self.btn_run = QPushButton("Run Inference"); tb.addWidget(self.btn_run)
        tb.addSeparator()
        self.btn_reset = QPushButton("Reset UI"); tb.addWidget(self.btn_reset)

        # --- View toggles (Dark Mode + Full Screen) ---
        self.act_dark = QtGui.QAction("Dark Mode", self)      
        self.act_dark.setCheckable(True)
        self.act_dark.setChecked(True)
        tb.addSeparator()
        tb.addAction(self.act_dark)

        self.act_fullscreen = QtGui.QAction("Full Screen", self)
        tb.addAction(self.act_fullscreen)

        # Shortcuts: F11 (all), Ctrl+Cmd+F (mac), Ctrl+D (toggle dark)
        self.sc_fs = QtGui.QShortcut(QtGui.QKeySequence("F11"), self)
        self.sc_fs_mac = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Meta+F"), self)
        self.sc_dark = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+D"), self)

        # Wire toggles
        self.act_dark.toggled.connect(self.toggle_dark)
        self.act_fullscreen.triggered.connect(self.toggle_fullscreen)
        self.sc_fs.activated.connect(self.toggle_fullscreen)
        self.sc_fs_mac.activated.connect(self.toggle_fullscreen)
        self.sc_dark.activated.connect(lambda: self.act_dark.toggle())

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
        views_grid.setContentsMargins(8, 8, 8, 8)
        views_grid.setHorizontalSpacing(16)
        views_grid.setVerticalSpacing(8)

        self.sag_box = make_placeholder("Sagittal")
        self.cor_box = make_placeholder("Coronal")
        self.ax_box  = make_placeholder("Axial")

        views_grid.addWidget(self.sag_box, 0, 0)
        views_grid.addWidget(self.cor_box, 0, 1)
        views_grid.addWidget(self.ax_box,  0, 2)

        # Expand Coronal & Axial a bit more than Sagittal
        views_grid.setColumnStretch(0, 1)  # sagittal
        views_grid.setColumnStretch(1, 2)  # coronal
        views_grid.setColumnStretch(2, 2)  # axial

        center_layout.addWidget(views_row, stretch=10)

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
        center_layout.addWidget(sliders_row, stretch=0)

        # ===== Main 2-column layout (left controls + big center) =====
        main = QWidget(); main_layout = QHBoxLayout(main)
        main_layout.addWidget(left_box, 2)
        main_layout.addWidget(center, 9)  
        self.setCentralWidget(main)

        # ===== Status bar & style =====
        self.statusBar().showMessage("Slim UI loaded. No actions implemented.")
        self.setStyleSheet("""
            QToolBar { spacing: 8px; }
            QPushButton { padding: 6px 10px; }
            QGroupBox { margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """)

        # ===== NO-OP connections (keep UI logic-free for now) =====
        def noop(*args, **kwargs): pass
        for btn in (self.btn_run, self.btn_reset, self.btn_reset_view, self.btn_screenshot):
            btn.clicked.connect(noop)
        self.act_open_case.triggered.connect(noop)
        self.act_open_imgs.triggered.connect(noop)
        for s in (self.sag_slider, self.cor_slider, self.ax_slider):
            s.valueChanged.connect(noop)
        for cb in (self.cb_femur, self.cb_tibia, self.cb_bg, self.cb_link_views):
            cb.stateChanged.connect(noop)
        self.opacity.valueChanged.connect(noop)
        self.render_mode.currentIndexChanged.connect(noop)
        self.model_combo.currentIndexChanged.connect(noop)

    # --------- Handlers for view toggles ---------
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.statusBar().showMessage("Exited full screen.")
        else:
            self.showFullScreen()
            self.statusBar().showMessage("Entered full screen.")

    def toggle_dark(self, enabled: bool):
        app = QtWidgets.QApplication.instance()
        if enabled:
            apply_dark_palette(app)
            self.statusBar().showMessage("Dark mode on.")
        else:
            apply_light_palette(app)
            self.statusBar().showMessage("Light mode on.")


# ---------------------------
# Entrypoint
# ---------------------------
def main():
    app = QApplication([])
    apply_dark_palette(app)  # Starts program in dark mode
    w = MainWindow()
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
