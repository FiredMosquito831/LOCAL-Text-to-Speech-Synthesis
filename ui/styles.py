"""
Dark Theme Stylesheet for VoiceSynth Studio
============================================
Modern dark theme with accent colors and smooth aesthetics.
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt


DARK_STYLESHEET = """
/* ============================================
   Global Styles
   ============================================ */

QMainWindow {
    background-color: #1a1a2e;
}

QWidget {
    background-color: #1a1a2e;
    color: #eaeaea;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

/* ============================================
   Group Boxes
   ============================================ */

QGroupBox {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 8px;
    margin-top: 12px;
    padding: 15px;
    padding-top: 25px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 0 8px;
    color: #00d9ff;
    font-weight: bold;
    font-size: 14px;
}

/* ============================================
   Buttons
   ============================================ */

QPushButton {
    background-color: #0f3460;
    color: #eaeaea;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: 500;
    min-height: 20px;
}

QPushButton:hover {
    background-color: #1a4a7a;
}

QPushButton:pressed {
    background-color: #0a2540;
}

QPushButton:disabled {
    background-color: #2a2a4a;
    color: #666;
}

QPushButton#primaryButton {
    background-color: #e94560;
    color: white;
    font-weight: bold;
}

QPushButton#primaryButton:hover {
    background-color: #ff5577;
}

QPushButton#primaryButton:pressed {
    background-color: #c93550;
}

/* ============================================
   Text Inputs
   ============================================ */

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #0f1429;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 8px 12px;
    color: #eaeaea;
    selection-background-color: #e94560;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #00d9ff;
}

/* ============================================
   Combo Boxes
   ============================================ */

QComboBox {
    background-color: #0f1429;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 8px 12px;
    min-width: 120px;
}

QComboBox:hover {
    border-color: #1a4a7a;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #00d9ff;
    margin-right: 10px;
}

QComboBox QAbstractItemView {
    background-color: #16213e;
    border: 1px solid #0f3460;
    selection-background-color: #e94560;
    outline: none;
}

/* ============================================
   Sliders
   ============================================ */

QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #0f3460;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #00d9ff;
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #33e6ff;
}

QSlider::sub-page:horizontal {
    background: #e94560;
    border-radius: 3px;
}

/* ============================================
   Progress Bars
   ============================================ */

QProgressBar {
    background-color: #0f1429;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #e94560,
        stop:1 #00d9ff
    );
    border-radius: 4px;
}

/* ============================================
   Tab Widget
   ============================================ */

QTabWidget::pane {
    border: 1px solid #0f3460;
    border-radius: 8px;
    background-color: #16213e;
    padding: 10px;
}

QTabBar::tab {
    background-color: #1a1a2e;
    color: #888;
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #16213e;
    color: #00d9ff;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #0f3460;
    color: #eaeaea;
}

/* ============================================
   Scroll Bars
   ============================================ */

QScrollBar:vertical {
    background-color: #0f1429;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #0f3460;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #1a4a7a;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #0f1429;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #0f3460;
    border-radius: 6px;
    min-width: 30px;
}

/* ============================================
   Labels
   ============================================ */

QLabel {
    color: #eaeaea;
    background: transparent;
}

QLabel#heading {
    font-size: 18px;
    font-weight: bold;
    color: #00d9ff;
}

QLabel#subheading {
    font-size: 11px;
    color: #888;
}

/* ============================================
   Check Boxes
   ============================================ */

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #0f3460;
    background-color: #0f1429;
}

QCheckBox::indicator:checked {
    background-color: #e94560;
    border-color: #e94560;
}

QCheckBox::indicator:hover {
    border-color: #00d9ff;
}

/* ============================================
   Spin Boxes
   ============================================ */

QSpinBox, QDoubleSpinBox {
    background-color: #0f1429;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 6px 10px;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #0f3460;
    border: none;
    width: 20px;
}

/* ============================================
   Status Bar
   ============================================ */

QStatusBar {
    background-color: #0f1429;
    color: #888;
    border-top: 1px solid #0f3460;
}

/* ============================================
   Menu Bar
   ============================================ */

QMenuBar {
    background-color: #0f1429;
    border-bottom: 1px solid #0f3460;
}

QMenuBar::item {
    padding: 8px 15px;
}

QMenuBar::item:selected {
    background-color: #0f3460;
}

QMenu {
    background-color: #16213e;
    border: 1px solid #0f3460;
}

QMenu::item {
    padding: 8px 30px;
}

QMenu::item:selected {
    background-color: #e94560;
}

/* ============================================
   Tool Tips
   ============================================ */

QToolTip {
    background-color: #16213e;
    color: #eaeaea;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 6px;
}
"""


def apply_dark_theme(app: QApplication):
    """Apply the dark theme to the application."""
    app.setStyleSheet(DARK_STYLESHEET)
    
    # Set palette for native widgets
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1a1a2e"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#eaeaea"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0f1429"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#16213e"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#16213e"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#eaeaea"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#eaeaea"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#0f3460"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#eaeaea"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#e94560"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#00d9ff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#e94560"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    
    app.setPalette(palette)
