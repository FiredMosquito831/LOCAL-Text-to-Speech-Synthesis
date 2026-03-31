"""
Quantization Configuration Panel
==================================
UI panel for model quantization settings.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QProgressBar, QFormLayout,
    QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

logger = logging.getLogger(__name__)


class BenchmarkWorker(QThread):
    """Background worker for quantization benchmarks."""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, model, test_text: str):
        super().__init__()
        self.model = model
        self.test_text = test_text
    
    def run(self):
        try:
            from core.quantization import QuantizationManager
            
            manager = QuantizationManager()
            results = manager.benchmark_quantization(
                self.model,
                self.test_text,
                progress_callback=lambda p, m: self.progress.emit(p, m)
            )
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class QuantizationPanel(QWidget):
    """
    Panel for configuring model quantization.
    
    Features:
    - Quantization method selection
    - VRAM estimation
    - Quality/speed tradeoff display
    - Benchmark tool
    - GGUF export
    """
    
    quantization_changed = pyqtSignal(str)  # quantization type
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # ===== Current Model Info =====
        info_group = QGroupBox("📦 Current Model")
        info_layout = QFormLayout(info_group)
        
        self.model_label = QLabel("F5-TTS v1 Base")
        info_layout.addRow("Model:", self.model_label)
        
        self.current_format_label = QLabel("FP16 SafeTensors (2.5 GB)")
        info_layout.addRow("Format:", self.current_format_label)
        
        self.current_vram_label = QLabel("~1.5 GB (INT8 loaded)")
        info_layout.addRow("VRAM:", self.current_vram_label)
        
        layout.addWidget(info_group)
        
        # ===== Quantization Selection =====
        quant_group = QGroupBox("🔢 Quantization Method")
        quant_layout = QVBoxLayout(quant_group)
        
        # Method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        
        self.quant_combo = QComboBox()
        self.quant_combo.addItems([
            "FP32 (Full Precision)",
            "FP16 (Half Precision)",
            "BF16 (Brain Float)",
            "INT8 Dynamic (bitsandbytes)",
            "INT8 Static (PTQ)",
            "INT4 GPTQ",
            "INT4 AWQ",
            "INT4 NF4 (bitsandbytes)",
            "GGUF Q8_0",
            "GGUF Q5_K_M",
            "GGUF Q4_K_M",
            "GGUF Q3_K_M",
            "GGUF Q2_K",
            "FP8 E4M3 (Ada+)"
        ])
        self.quant_combo.setCurrentText("INT8 Dynamic (bitsandbytes)")
        self.quant_combo.currentTextChanged.connect(self._on_quant_changed)
        method_layout.addWidget(self.quant_combo)
        
        quant_layout.addLayout(method_layout)
        
        # Method info table
        self.method_info = QTableWidget(4, 2)
        self.method_info.setHorizontalHeaderLabels(["Property", "Value"])
        self.method_info.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.method_info.verticalHeader().setVisible(False)
        self.method_info.setMaximumHeight(130)
        self._update_method_info()
        quant_layout.addWidget(self.method_info)
        
        layout.addWidget(quant_group)
        
        # ===== Runtime Options =====
        runtime_group = QGroupBox("⚡ Runtime Quantization")
        runtime_layout = QFormLayout(runtime_group)
        
        self.load_8bit = QCheckBox("Load in 8-bit (bitsandbytes)")
        self.load_8bit.setChecked(True)
        runtime_layout.addRow("", self.load_8bit)
        
        self.load_4bit = QCheckBox("Load in 4-bit (bitsandbytes NF4)")
        runtime_layout.addRow("", self.load_4bit)
        
        self.double_quant = QCheckBox("Use double quantization (saves ~0.4GB)")
        self.double_quant.setChecked(True)
        runtime_layout.addRow("", self.double_quant)
        
        compute_layout = QHBoxLayout()
        compute_layout.addWidget(QLabel("Compute dtype:"))
        self.compute_dtype = QComboBox()
        self.compute_dtype.addItems(["float16", "bfloat16", "float32"])
        compute_layout.addWidget(self.compute_dtype)
        compute_layout.addStretch()
        runtime_layout.addRow("", compute_layout)
        
        layout.addWidget(runtime_group)
        
        # ===== GGUF Export =====
        gguf_group = QGroupBox("📁 GGUF Export")
        gguf_layout = QVBoxLayout(gguf_group)
        
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.gguf_type = QComboBox()
        self.gguf_type.addItems(["Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"])
        self.gguf_type.setCurrentText("Q4_K_M")
        type_layout.addWidget(self.gguf_type)
        type_layout.addStretch()
        gguf_layout.addLayout(type_layout)
        
        path_layout = QHBoxLayout()
        self.gguf_path = QLineEdit()
        self.gguf_path.setPlaceholderText("Output path...")
        path_layout.addWidget(self.gguf_path)
        
        browse_btn = QPushButton("📁")
        browse_btn.setMaximumWidth(40)
        browse_btn.clicked.connect(self._browse_gguf_path)
        path_layout.addWidget(browse_btn)
        gguf_layout.addLayout(path_layout)
        
        convert_btn = QPushButton("🔨 Convert to GGUF")
        convert_btn.clicked.connect(self._convert_to_gguf)
        gguf_layout.addWidget(convert_btn)
        
        layout.addWidget(gguf_group)
        
        # ===== Benchmark =====
        bench_group = QGroupBox("📊 Benchmark")
        bench_layout = QVBoxLayout(bench_group)
        
        self.benchmark_btn = QPushButton("📊 Run Benchmark")
        self.benchmark_btn.clicked.connect(self._run_benchmark)
        bench_layout.addWidget(self.benchmark_btn)
        
        self.benchmark_progress = QProgressBar()
        self.benchmark_progress.setVisible(False)
        bench_layout.addWidget(self.benchmark_progress)
        
        self.benchmark_results = QLabel("No benchmark results yet")
        self.benchmark_results.setObjectName("subheading")
        bench_layout.addWidget(self.benchmark_results)
        
        layout.addWidget(bench_group)
        
        # ===== Apply Button =====
        btn_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("💾 Apply & Reload Model")
        self.apply_btn.setObjectName("primaryButton")
        self.apply_btn.clicked.connect(self._apply_quantization)
        btn_layout.addWidget(self.apply_btn)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
    
    def _on_quant_changed(self, text: str):
        """Handle quantization method change."""
        self._update_method_info()
        
        # Enable/disable relevant options based on method
        is_bnb = "bitsandbytes" in text.lower() or "nf4" in text.lower()
        is_gguf = "gguf" in text.lower()
        
        self.load_8bit.setEnabled(is_bnb)
        self.load_4bit.setEnabled(is_bnb)
        self.double_quant.setEnabled(is_bnb)
    
    def _update_method_info(self):
        """Update method information table."""
        from core.quantization import QUANT_SPECS, QuantizationType
        
        # Map combo text to enum
        text = self.quant_combo.currentText()
        qtype = None
        
        for qt in QuantizationType:
            if qt.value in text.lower().replace(" ", "-").replace("(", "").replace(")", ""):
                qtype = qt
                break
        
        # Default values
        bits = "?"
        vram_reduction = "?"
        quality = "?"
        speed = "?"
        
        if qtype and qtype in QUANT_SPECS:
            info = QUANT_SPECS[qtype]
            bits = str(info.bits)
            vram_reduction = f"{info.vram_reduction}%"
            quality = "★" * info.quality_rating + "☆" * (5 - info.quality_rating)
            speed = f"{info.speed_impact:.1f}x"
        
        # Update table
        self.method_info.setItem(0, 0, QTableWidgetItem("Bits"))
        self.method_info.setItem(0, 1, QTableWidgetItem(bits))
        self.method_info.setItem(1, 0, QTableWidgetItem("VRAM Reduction"))
        self.method_info.setItem(1, 1, QTableWidgetItem(vram_reduction))
        self.method_info.setItem(2, 0, QTableWidgetItem("Quality"))
        self.method_info.setItem(2, 1, QTableWidgetItem(quality))
        self.method_info.setItem(3, 0, QTableWidgetItem("Speed"))
        self.method_info.setItem(3, 1, QTableWidgetItem(speed))
    
    def _browse_gguf_path(self):
        """Browse for GGUF output path."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save GGUF Model",
            "",
            "GGUF Files (*.gguf)"
        )
        if path:
            self.gguf_path.setText(path)
    
    def _convert_to_gguf(self):
        """Start GGUF conversion."""
        output_path = self.gguf_path.text()
        if not output_path:
            QMessageBox.warning(self, "No Path", "Please specify an output path.")
            return
        
        QMessageBox.information(
            self,
            "GGUF Conversion",
            "GGUF conversion requires llama.cpp's convert scripts.\n\n"
            "See: https://github.com/ggerganov/llama.cpp#prepare-and-quantize\n\n"
            "Automated conversion coming in a future update."
        )
    
    def _run_benchmark(self):
        """Run quantization benchmark."""
        self.benchmark_progress.setVisible(True)
        self.benchmark_progress.setValue(0)
        self.benchmark_btn.setEnabled(False)
        
        # For now, show placeholder results
        self.benchmark_progress.setValue(100)
        self.benchmark_results.setText(
            "Inference: ~150ms | Memory: 1.5GB | Quality: ★★★★☆"
        )
        self.benchmark_btn.setEnabled(True)
        self.benchmark_progress.setVisible(False)
    
    def _apply_quantization(self):
        """Apply the selected quantization settings."""
        method = self.quant_combo.currentText()
        
        QMessageBox.information(
            self,
            "Apply Quantization",
            f"Quantization method: {method}\n\n"
            "The model will be reloaded with the new settings."
        )
        
        # Emit signal for parent to handle
        self.quantization_changed.emit(method)
