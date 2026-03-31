"""
Optimization Presets Panel
===========================
UI panel for managing and applying optimization presets.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QComboBox, QPushButton, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTextEdit, QMessageBox, QInputDialog, QListWidget,
    QListWidgetItem, QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)


class PresetsPanel(QWidget):
    """
    Panel for managing optimization presets.
    
    Features:
    - Built-in preset selection
    - Custom preset creation
    - Preset details display
    - Apply preset action
    """
    
    preset_applied = pyqtSignal(dict)  # Emits preset config when applied
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_presets()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # ===== Active Preset =====
        active_group = QGroupBox("⚡ Active Preset")
        active_layout = QFormLayout(active_group)
        
        self.preset_combo = QComboBox()
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        active_layout.addRow("Preset:", self.preset_combo)
        
        self.recommendation_label = QLabel("Checking recommendation...")
        self.recommendation_label.setObjectName("subheading")
        active_layout.addRow("", self.recommendation_label)
        
        layout.addWidget(active_group)
        
        # ===== Preset Details =====
        details_group = QGroupBox("📋 Preset Details")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        self.details_text.setStyleSheet("""
            QTextEdit {
                background-color: #0f1429;
                border: 1px solid #333;
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
        """)
        details_layout.addWidget(self.details_text)
        
        layout.addWidget(details_group)
        
        # ===== Preset Comparison =====
        comparison_group = QGroupBox("📊 Preset Comparison")
        comparison_layout = QVBoxLayout(comparison_group)
        
        # Create comparison table as text
        comparison_text = """
╔═══════════════╦═════════╦═══════════════╦════════════╦═══════════╦══════════╗
║    Preset     ║  VRAM   ║  Quantization ║  Whisper   ║  Quality  ║  Speed   ║
╠═══════════════╬═════════╬═══════════════╬════════════╬═══════════╬══════════╣
║  Ultra Low    ║  3 GB   ║  GGUF Q4      ║  tiny      ║  ★★☆☆☆   ║  0.5x RT ║
║  Low          ║  4 GB   ║  INT8         ║  small     ║  ★★★☆☆   ║  0.4x RT ║
║  Balanced     ║  5 GB   ║  INT8         ║  small     ║  ★★★★☆   ║  0.3x RT ║
║  Quality      ║  6 GB   ║  FP16         ║  medium    ║  ★★★★★   ║  0.25x RT║
╚═══════════════╩═════════╩═══════════════╩════════════╩═══════════╩══════════╝
        """
        
        comparison_label = QLabel(comparison_text)
        comparison_label.setFont(comparison_label.font())
        comparison_label.setStyleSheet("""
            QLabel {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                background-color: #0f1429;
                padding: 10px;
                border-radius: 4px;
            }
        """)
        comparison_layout.addWidget(comparison_label)
        
        layout.addWidget(comparison_group)
        
        # ===== Custom Preset =====
        custom_group = QGroupBox("✏️ Custom Preset")
        custom_layout = QVBoxLayout(custom_group)
        
        btn_layout = QHBoxLayout()
        
        create_btn = QPushButton("➕ Create Custom")
        create_btn.clicked.connect(self._create_custom_preset)
        btn_layout.addWidget(create_btn)
        
        edit_btn = QPushButton("✏️ Edit Current")
        edit_btn.clicked.connect(self._edit_preset)
        btn_layout.addWidget(edit_btn)
        
        delete_btn = QPushButton("🗑 Delete Custom")
        delete_btn.clicked.connect(self._delete_preset)
        btn_layout.addWidget(delete_btn)
        
        custom_layout.addLayout(btn_layout)
        layout.addWidget(custom_group)
        
        # ===== Apply Button =====
        self.apply_btn = QPushButton("💾 Apply Preset")
        self.apply_btn.setObjectName("primaryButton")
        self.apply_btn.setMinimumHeight(40)
        self.apply_btn.clicked.connect(self._apply_preset)
        layout.addWidget(self.apply_btn)
        
        layout.addStretch()
    
    def _load_presets(self):
        """Load available presets into combo box."""
        try:
            from core.presets import PresetManager, PresetType, get_recommended_preset
            
            manager = PresetManager()
            
            # Add built-in presets
            self.preset_combo.addItem("Ultra Low VRAM (3 GB)")
            self.preset_combo.addItem("Low VRAM (4 GB)")
            self.preset_combo.addItem("Balanced (5 GB)")
            self.preset_combo.addItem("Quality (6 GB)")
            
            # Add separator
            self.preset_combo.insertSeparator(4)
            
            # Add custom presets
            for name in manager._custom_presets.keys():
                self.preset_combo.addItem(f"Custom: {name}")
            
            # Get recommendation
            recommended = get_recommended_preset()
            self.recommendation_label.setText(f"💡 Recommended: {recommended.name}")
            
            # Select recommended
            for i in range(self.preset_combo.count()):
                if recommended.name in self.preset_combo.itemText(i):
                    self.preset_combo.setCurrentIndex(i)
                    break
            
        except Exception as e:
            logger.error(f"Failed to load presets: {e}")
            self.preset_combo.addItem("Default (Balanced)")
    
    def _on_preset_changed(self, text: str):
        """Handle preset selection change."""
        try:
            from core.presets import PresetManager, PresetType, BUILTIN_PRESETS
            
            manager = PresetManager()
            
            # Determine which preset was selected
            if "Ultra Low" in text:
                preset = BUILTIN_PRESETS[PresetType.ULTRA_LOW]
            elif "Low VRAM" in text:
                preset = BUILTIN_PRESETS[PresetType.LOW]
            elif "Balanced" in text:
                preset = BUILTIN_PRESETS[PresetType.BALANCED]
            elif "Quality" in text:
                preset = BUILTIN_PRESETS[PresetType.QUALITY]
            elif "Custom:" in text:
                name = text.replace("Custom: ", "")
                preset = manager.get_custom_preset(name)
            else:
                preset = BUILTIN_PRESETS[PresetType.BALANCED]
            
            if preset:
                summary = manager.get_preset_summary(preset)
                self.details_text.setText(summary)
            
        except Exception as e:
            logger.error(f"Failed to update preset details: {e}")
    
    def _create_custom_preset(self):
        """Create a new custom preset."""
        name, ok = QInputDialog.getText(
            self,
            "Create Custom Preset",
            "Enter preset name:"
        )
        
        if ok and name:
            try:
                from core.presets import OptimizationPreset, PresetManager
                
                # Create new preset with default values
                preset = OptimizationPreset(
                    name=name,
                    description="Custom preset",
                    target_vram_gb=5.0,
                    backend="pytorch",
                    quantization="int8-dynamic",
                    whisper_model="small",
                    whisper_compute_type="int8",
                    enable_flash_attention=True,
                    enable_torch_compile=False,
                    enable_attention_slicing=False,
                    enable_cpu_offload=False,
                    max_batch_size=2,
                    nfe_steps=16,
                    expected_quality=4,
                    expected_rtf=0.3
                )
                
                # Save it
                manager = PresetManager()
                manager.save_custom_preset(preset)
                
                # Add to combo
                self.preset_combo.addItem(f"Custom: {name}")
                self.preset_combo.setCurrentText(f"Custom: {name}")
                
                QMessageBox.information(
                    self,
                    "Preset Created",
                    f"Custom preset '{name}' created!\n\n"
                    "Use 'Edit Current' to customize the settings."
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create preset: {e}")
    
    def _edit_preset(self):
        """Edit the current preset."""
        current = self.preset_combo.currentText()
        
        if "Custom:" not in current:
            QMessageBox.warning(
                self,
                "Cannot Edit",
                "Built-in presets cannot be edited.\n\n"
                "Create a custom preset to make changes."
            )
            return
        
        # TODO: Open preset editor dialog
        QMessageBox.information(
            self,
            "Edit Preset",
            "Preset editor coming soon!\n\n"
            "For now, edit the JSON file in:\n"
            "~/.config/voicesynth/presets/"
        )
    
    def _delete_preset(self):
        """Delete the current custom preset."""
        current = self.preset_combo.currentText()
        
        if "Custom:" not in current:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "Built-in presets cannot be deleted."
            )
            return
        
        name = current.replace("Custom: ", "")
        
        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Delete custom preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from core.presets import PresetManager
                
                manager = PresetManager()
                manager.delete_custom_preset(name)
                
                # Remove from combo
                idx = self.preset_combo.currentIndex()
                self.preset_combo.removeItem(idx)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete preset: {e}")
    
    def _apply_preset(self):
        """Apply the selected preset."""
        current = self.preset_combo.currentText()
        
        try:
            from core.presets import PresetManager, PresetType, BUILTIN_PRESETS
            
            manager = PresetManager()
            
            # Determine which preset
            if "Ultra Low" in current:
                preset = BUILTIN_PRESETS[PresetType.ULTRA_LOW]
            elif "Low VRAM" in current:
                preset = BUILTIN_PRESETS[PresetType.LOW]
            elif "Balanced" in current:
                preset = BUILTIN_PRESETS[PresetType.BALANCED]
            elif "Quality" in current:
                preset = BUILTIN_PRESETS[PresetType.QUALITY]
            elif "Custom:" in current:
                name = current.replace("Custom: ", "")
                preset = manager.get_custom_preset(name)
            else:
                preset = BUILTIN_PRESETS[PresetType.BALANCED]
            
            if preset:
                config = manager.apply_preset(preset)
                self.preset_applied.emit(config)
                
                QMessageBox.information(
                    self,
                    "Preset Applied",
                    f"Applied preset: {preset.name}\n\n"
                    f"Backend: {preset.backend}\n"
                    f"Quantization: {preset.quantization}\n"
                    f"The model will be reloaded with new settings."
                )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply preset: {e}")
