"""
Main Window
============
Primary application window assembling all panels.
"""

import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QStatusBar, QMenuBar, QMenu, QToolBar,
    QMessageBox, QFileDialog, QSplitter, QLabel, QGroupBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QFont

from ui.tts_panel import TTSPanel
from ui.cloning_panel import VoiceCloningPanel
from ui.monitor_widget import GPUMonitorWidget
from core.inference_manager import InferenceManager

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    VoiceSynth Studio Main Window.
    
    Layout:
    ┌─────────────────────────────────────────────────────┐
    │  Menu Bar                                           │
    ├─────────────────────────────────────────────────────┤
    │  ┌─────────────────────────┬───────────────────────┐│
    │  │                         │                       ││
    │  │   Tab Widget            │   GPU Monitor         ││
    │  │   - TTS Panel           │                       ││
    │  │   - Voice Cloning       │                       ││
    │  │   - Settings            │                       ││
    │  │                         │                       ││
    │  └─────────────────────────┴───────────────────────┘│
    ├─────────────────────────────────────────────────────┤
    │  Status Bar                                         │
    └─────────────────────────────────────────────────────┘
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VoiceSynth Studio v0.1.0")
        self.setMinimumSize(1200, 800)
        
        # Initialize inference manager
        self.inference_manager = InferenceManager(
            backend="f5-tts",
            quantization="int8",
            device="auto"
        )
        
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._connect_signals()
        
        logger.info("Main window initialized")
    
    def _setup_ui(self):
        """Initialize the main UI layout."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout: horizontal splitter
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ===== Left: Tab Widget with main panels =====
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setDocumentMode(True)
        
        # TTS Panel
        self.tts_panel = TTSPanel()
        self.tts_panel.set_inference_manager(self.inference_manager)
        self.tab_widget.addTab(self.tts_panel, "🔊 Text-to-Speech")
        
        # Voice Cloning Panel
        self.cloning_panel = VoiceCloningPanel()
        self.tab_widget.addTab(self.cloning_panel, "🎤 Voice Cloning")
        
        # Backend Configuration Panel
        from ui.backend_config import BackendConfigPanel
        self.backend_config = BackendConfigPanel()
        self.tab_widget.addTab(self.backend_config, "⚡ Backends")
        
        # Quantization Panel
        from ui.quantization_panel import QuantizationPanel
        self.quantization_panel = QuantizationPanel()
        self.tab_widget.addTab(self.quantization_panel, "🔢 Quantization")
        
        # Presets Panel
        from ui.presets_panel import PresetsPanel
        self.presets_panel = PresetsPanel()
        self.tab_widget.addTab(self.presets_panel, "📦 Presets")
        
        # Batch Processing Panel
        from ui.batch_panel import BatchPanel
        self.batch_panel = BatchPanel(inference_manager=self.inference_manager)
        self.tab_widget.addTab(self.batch_panel, "📋 Batch")
        
        # Settings Panel
        settings_widget = self._create_settings_placeholder()
        self.tab_widget.addTab(settings_widget, "⚙️ Settings")
        
        splitter.addWidget(self.tab_widget)
        
        # ===== Right: GPU Monitor =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.gpu_monitor = GPUMonitorWidget()
        right_layout.addWidget(self.gpu_monitor)
        
        # Model info
        model_info = self._create_model_info_widget()
        right_layout.addWidget(model_info)
        
        right_layout.addStretch()
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([900, 300])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
    
    def _create_settings_placeholder(self) -> QWidget:
        """Create placeholder settings panel."""
        from PyQt6.QtWidgets import QGroupBox, QFormLayout, QComboBox, QCheckBox, QSpinBox
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Backend settings
        backend_group = QGroupBox("🔧 Backend Configuration")
        backend_layout = QFormLayout(backend_group)
        
        backend_combo = QComboBox()
        backend_combo.addItems(["F5-TTS", "GPT-SoVITS", "Fish Speech"])
        backend_layout.addRow("Primary Backend:", backend_combo)
        
        quant_combo = QComboBox()
        quant_combo.addItems(["FP16", "INT8", "INT4 (GPTQ)", "GGUF Q4", "GGUF Q8"])
        quant_combo.setCurrentText("INT8")
        backend_layout.addRow("Quantization:", quant_combo)
        
        device_combo = QComboBox()
        device_combo.addItems(["Auto", "CUDA", "CPU"])
        backend_layout.addRow("Device:", device_combo)
        
        layout.addWidget(backend_group)
        
        # Memory settings
        memory_group = QGroupBox("💾 Memory Management")
        memory_layout = QFormLayout(memory_group)
        
        vram_limit = QSpinBox()
        vram_limit.setRange(1, 24)
        vram_limit.setValue(5)
        vram_limit.setSuffix(" GB")
        memory_layout.addRow("VRAM Limit:", vram_limit)
        
        auto_offload = QCheckBox("CPU offload when VRAM full")
        auto_offload.setChecked(True)
        memory_layout.addRow("", auto_offload)
        
        flash_attn = QCheckBox("Enable Flash Attention 2")
        flash_attn.setChecked(True)
        memory_layout.addRow("", flash_attn)
        
        layout.addWidget(memory_group)
        
        # Whisper settings
        whisper_group = QGroupBox("🎙 Faster-Whisper Settings")
        whisper_layout = QFormLayout(whisper_group)
        
        whisper_model = QComboBox()
        whisper_model.addItems(["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"])
        whisper_model.setCurrentText("small")
        whisper_layout.addRow("Default Model:", whisper_model)
        
        whisper_compute = QComboBox()
        whisper_compute.addItems(["int8", "float16", "float32"])
        whisper_layout.addRow("Compute Type:", whisper_compute)
        
        layout.addWidget(whisper_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_model_info_widget(self) -> QGroupBox:
        """Create model info display widget."""
        group = QGroupBox("📦 Loaded Models")
        layout = QVBoxLayout(group)
        
        self.model_status_label = QLabel("No models loaded")
        self.model_status_label.setObjectName("subheading")
        layout.addWidget(self.model_status_label)
        
        from PyQt6.QtWidgets import QPushButton
        
        load_btn = QPushButton("🔄 Load Model")
        load_btn.clicked.connect(self._load_model)
        layout.addWidget(load_btn)
        
        unload_btn = QPushButton("📤 Unload Model")
        unload_btn.clicked.connect(self._unload_model)
        layout.addWidget(unload_btn)
        
        return group
    
    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("New Project", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Project...", self)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Project", self)
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Audio...", self)
        export_action.setShortcut("Ctrl+E")
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        model_manager_action = QAction("Model Manager...", self)
        tools_menu.addAction(model_manager_action)
        
        batch_process_action = QAction("Batch Processing...", self)
        tools_menu.addAction(batch_process_action)
        
        tools_menu.addSeparator()
        
        clear_cache_action = QAction("Clear GPU Cache", self)
        clear_cache_action.triggered.connect(self._clear_gpu_cache)
        tools_menu.addAction(clear_cache_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        docs_action = QAction("Documentation", self)
        help_menu.addAction(docs_action)
        
        about_action = QAction("About VoiceSynth Studio", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Permanent widgets
        self.backend_label = QLabel("Backend: F5-TTS")
        self.statusbar.addPermanentWidget(self.backend_label)
        
        self.device_label = QLabel("Device: Detecting...")
        self.statusbar.addPermanentWidget(self.device_label)
        
        # Update device info
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.device_label.setText(f"Device: {device_name}")
            else:
                self.device_label.setText("Device: CPU")
        except ImportError:
            self.device_label.setText("Device: CPU (PyTorch not loaded)")
        
        self.statusbar.showMessage("Ready")
    
    def _connect_signals(self):
        """Connect panel signals."""
        # When a voice is selected in cloning panel, update TTS panel
        self.cloning_panel.voice_selected.connect(self._on_voice_selected)
        
        # When a profile is saved/deleted, refresh voice lists in all panels
        self.cloning_panel.profile_saved.connect(self.tts_panel.refresh_voices)
        self.cloning_panel.profile_saved.connect(self.batch_panel.refresh_voices)
        
        # When a preset is applied, reload the model with new settings
        self.presets_panel.preset_applied.connect(self._on_preset_applied)
        
        # GPU monitor warnings
        self.gpu_monitor.memory_warning.connect(self._on_memory_warning)
    
    def _on_preset_applied(self, config: dict):
        """Handle preset application - reload model with new settings."""
        logger.info(f"Preset applied: {config}")
        
        # Update inference manager settings
        if 'quantization' in config:
            self.inference_manager.quantization = config['quantization']
        if 'backend' in config:
            self.inference_manager.backend_name = config['backend']
        
        # Unload and reload with new settings
        self.inference_manager.unload_model()
        self.statusbar.showMessage("Preset applied - model will reload on next generation", 5000)
    
    def _on_voice_selected(self, name: str, path: str):
        """Handle voice selection from cloning panel."""
        # Add to TTS voice combo with path stored
        combo = self.tts_panel.voice_combo
        
        # Check if already exists
        existing_idx = combo.findText(f"👤 {name}")
        if existing_idx == -1:
            existing_idx = combo.findText(name)
        
        if existing_idx >= 0:
            # Update existing item's path
            combo.setItemData(existing_idx, path, Qt.ItemDataRole.UserRole)
            combo.setCurrentIndex(existing_idx)
        else:
            # Add new item with path
            idx = combo.count()
            combo.addItem(f"👤 {name}")
            combo.setItemData(idx, path, Qt.ItemDataRole.UserRole)
            combo.setCurrentIndex(idx)
        
        # Set voice in inference manager
        if hasattr(self.tts_panel, '_inference_manager') and self.tts_panel._inference_manager:
            self.tts_panel._inference_manager.set_voice(path)
        
        # Switch to TTS tab
        self.tab_widget.setCurrentWidget(self.tts_panel)
        
        self.statusbar.showMessage(f"Voice '{name}' selected", 3000)
        logger.info(f"Voice selected from cloning: {name} -> {path}")
    
    def _on_memory_warning(self, percent: float):
        """Handle GPU memory warning."""
        self.statusbar.showMessage(f"⚠️ GPU memory at {percent:.0f}%!", 5000)
    
    def _load_model(self):
        """Load the TTS model."""
        self.statusbar.showMessage("Loading model...")
        self.model_status_label.setText("Loading...")
        
        try:
            self.inference_manager.load_model()
            self.model_status_label.setText("✓ F5-TTS (INT8) loaded")
            self.statusbar.showMessage("Model loaded successfully", 3000)
        except Exception as e:
            self.model_status_label.setText(f"✗ Error: {e}")
            self.statusbar.showMessage(f"Failed to load model: {e}", 5000)
            logger.error(f"Model load error: {e}")
    
    def _unload_model(self):
        """Unload the TTS model."""
        self.inference_manager.unload_model()
        self.model_status_label.setText("No models loaded")
        self.statusbar.showMessage("Model unloaded", 3000)
    
    def _clear_gpu_cache(self):
        """Clear GPU cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            self.statusbar.showMessage("GPU cache cleared", 3000)
        except Exception as e:
            self.statusbar.showMessage(f"Error: {e}", 3000)
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About VoiceSynth Studio",
            """<h2>VoiceSynth Studio</h2>
            <p>Version 0.1.0</p>
            <p>A state-of-the-art local voice synthesis application.</p>
            <p><b>Features:</b></p>
            <ul>
                <li>Multiple TTS backends (F5-TTS, GPT-SoVITS, Fish Speech)</li>
                <li>Voice cloning with faster-whisper</li>
                <li>Memory-optimized for 6GB VRAM GPUs</li>
                <li>INT8/INT4 quantization support</li>
            </ul>
            <p>&copy; 2026 VoiceSynth Project</p>
            """
        )
    
    def closeEvent(self, event):
        """Handle window close."""
        # Stop GPU monitoring
        self.gpu_monitor.stop_monitoring()
        
        # Unload models
        self.inference_manager.unload_model()
        
        logger.info("Application closing")
        event.accept()
