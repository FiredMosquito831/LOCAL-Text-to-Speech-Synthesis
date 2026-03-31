"""
Backend Configuration Panel
=============================
UI panel for configuring inference backends.
"""

import logging
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QStackedWidget, QFormLayout,
    QLineEdit, QFileDialog, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)


class BackendConfigPanel(QWidget):
    """
    Configurable panel for all inference backend options.
    
    Displays backend-specific configuration options
    based on the selected backend.
    """
    
    backend_changed = pyqtSignal(str)  # backend name
    config_changed = pyqtSignal(dict)  # config dict
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # ===== Backend Selection =====
        selection_group = QGroupBox("⚙️ Inference Backend")
        selection_layout = QFormLayout(selection_group)
        
        self.backend_combo = QComboBox()
        self.backend_combo.addItems([
            "PyTorch Native (F5-TTS)",
            "vLLM (Optimized)",
            "TensorRT-LLM (Fastest)",
            "ONNX Runtime (Portable)",
            "llama.cpp (GGUF)",
            "OpenVINO (Intel)",
            "DirectML (Windows)"
        ])
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        selection_layout.addRow("Primary Backend:", self.backend_combo)
        
        self.auto_fallback = QCheckBox("Auto-fallback to CPU if VRAM exhausted")
        self.auto_fallback.setChecked(True)
        selection_layout.addRow("", self.auto_fallback)
        
        layout.addWidget(selection_group)
        
        # ===== Stacked Widget for Backend-specific Options =====
        self.options_stack = QStackedWidget()
        
        # PyTorch Native options
        self.pytorch_widget = self._create_pytorch_options()
        self.options_stack.addWidget(self.pytorch_widget)
        
        # vLLM options
        self.vllm_widget = self._create_vllm_options()
        self.options_stack.addWidget(self.vllm_widget)
        
        # TensorRT options
        self.tensorrt_widget = self._create_tensorrt_options()
        self.options_stack.addWidget(self.tensorrt_widget)
        
        # ONNX options
        self.onnx_widget = self._create_onnx_options()
        self.options_stack.addWidget(self.onnx_widget)
        
        # llama.cpp options
        self.llamacpp_widget = self._create_llamacpp_options()
        self.options_stack.addWidget(self.llamacpp_widget)
        
        # OpenVINO options
        self.openvino_widget = self._create_openvino_options()
        self.options_stack.addWidget(self.openvino_widget)
        
        # DirectML options
        self.directml_widget = self._create_directml_options()
        self.options_stack.addWidget(self.directml_widget)
        
        layout.addWidget(self.options_stack)
        
        # ===== Backend Status =====
        status_group = QGroupBox("📊 Backend Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Status: Not loaded")
        status_layout.addWidget(self.status_label)
        
        self.availability_label = QLabel("Checking availability...")
        self.availability_label.setObjectName("subheading")
        status_layout.addWidget(self.availability_label)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
        
        # Check availability on load
        self._check_backend_availability()
    
    def _create_pytorch_options(self) -> QGroupBox:
        """Create PyTorch native options."""
        group = QGroupBox("PyTorch Native Options")
        layout = QFormLayout(group)
        
        quant_combo = QComboBox()
        quant_combo.addItems(["FP32", "FP16", "INT8 (bitsandbytes)"])
        quant_combo.setCurrentText("INT8 (bitsandbytes)")
        layout.addRow("Quantization:", quant_combo)
        
        device_combo = QComboBox()
        device_combo.addItems(["Auto", "CUDA", "CPU"])
        layout.addRow("Device:", device_combo)
        
        flash_attn = QCheckBox("Enable Flash Attention 2")
        flash_attn.setChecked(True)
        layout.addRow("", flash_attn)
        
        compile_check = QCheckBox("Use torch.compile() (slower startup)")
        layout.addRow("", compile_check)
        
        return group
    
    def _create_vllm_options(self) -> QGroupBox:
        """Create vLLM options."""
        group = QGroupBox("vLLM Options")
        layout = QFormLayout(group)
        
        gpu_util = QDoubleSpinBox()
        gpu_util.setRange(0.1, 1.0)
        gpu_util.setValue(0.85)
        gpu_util.setSingleStep(0.05)
        gpu_util.setSuffix(" (85%)")
        layout.addRow("GPU Memory Utilization:", gpu_util)
        
        chunked_prefill = QCheckBox("Enable Chunked Prefill")
        chunked_prefill.setChecked(True)
        layout.addRow("", chunked_prefill)
        
        prefix_cache = QCheckBox("Enable Prefix Caching")
        prefix_cache.setChecked(True)
        layout.addRow("", prefix_cache)
        
        tp_size = QSpinBox()
        tp_size.setRange(1, 8)
        tp_size.setValue(1)
        layout.addRow("Tensor Parallel Size:", tp_size)
        
        # Warning for Windows
        warning = QLabel("⚠️ vLLM has limited Windows support. Linux recommended.")
        warning.setStyleSheet("color: #ff9800;")
        layout.addRow(warning)
        
        return group
    
    def _create_tensorrt_options(self) -> QGroupBox:
        """Create TensorRT options."""
        group = QGroupBox("TensorRT-LLM Options")
        layout = QFormLayout(group)
        
        cache_path = QLineEdit()
        cache_path.setPlaceholderText("C:\\Users\\...\\trt_engines\\")
        browse_btn = QPushButton("📁")
        browse_btn.setMaximumWidth(40)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(cache_path)
        path_layout.addWidget(browse_btn)
        layout.addRow("Engine Cache:", path_layout)
        
        precision_layout = QHBoxLayout()
        fp16_check = QCheckBox("FP16")
        fp16_check.setChecked(True)
        int8_check = QCheckBox("INT8")
        fp8_check = QCheckBox("FP8 (Ada+)")
        precision_layout.addWidget(fp16_check)
        precision_layout.addWidget(int8_check)
        precision_layout.addWidget(fp8_check)
        layout.addRow("Precision:", precision_layout)
        
        batch_size = QSpinBox()
        batch_size.setRange(1, 16)
        batch_size.setValue(4)
        layout.addRow("Max Batch Size:", batch_size)
        
        btn_layout = QHBoxLayout()
        rebuild_btn = QPushButton("🔨 Rebuild Engine")
        clear_btn = QPushButton("📁 Clear Cache")
        btn_layout.addWidget(rebuild_btn)
        btn_layout.addWidget(clear_btn)
        layout.addRow("", btn_layout)
        
        return group
    
    def _create_onnx_options(self) -> QGroupBox:
        """Create ONNX Runtime options."""
        group = QGroupBox("ONNX Runtime Options")
        layout = QFormLayout(group)
        
        provider_combo = QComboBox()
        provider_combo.addItems(["Auto", "CUDA", "ROCm", "DirectML", "CPU"])
        layout.addRow("Execution Provider:", provider_combo)
        
        threads = QSpinBox()
        threads.setRange(1, 32)
        threads.setValue(4)
        layout.addRow("CPU Threads:", threads)
        
        onnx_path = QLineEdit()
        onnx_path.setPlaceholderText("Path to .onnx model file")
        layout.addRow("Model Path:", onnx_path)
        
        return group
    
    def _create_llamacpp_options(self) -> QGroupBox:
        """Create llama.cpp options."""
        group = QGroupBox("llama.cpp (GGUF) Options")
        layout = QFormLayout(group)
        
        ctx_length = QComboBox()
        ctx_length.addItems(["512", "1024", "2048", "4096", "8192"])
        ctx_length.setCurrentText("2048")
        layout.addRow("Context Length:", ctx_length)
        
        gpu_layers = QSpinBox()
        gpu_layers.setRange(0, 100)
        gpu_layers.setValue(32)
        gpu_layers.setToolTip("0 = CPU only, higher = more GPU usage")
        layout.addRow("GPU Layers:", gpu_layers)
        
        threads = QSpinBox()
        threads.setRange(1, 32)
        threads.setValue(8)
        layout.addRow("CPU Threads:", threads)
        
        quant_combo = QComboBox()
        quant_combo.addItems(["Q8_0 (Best)", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K (Smallest)"])
        quant_combo.setCurrentText("Q4_K_M")
        layout.addRow("Quantization:", quant_combo)
        
        return group
    
    def _create_openvino_options(self) -> QGroupBox:
        """Create OpenVINO options."""
        group = QGroupBox("OpenVINO Options")
        layout = QFormLayout(group)
        
        device_combo = QComboBox()
        device_combo.addItems(["AUTO", "CPU", "GPU", "NPU"])
        layout.addRow("Device:", device_combo)
        
        threads = QSpinBox()
        threads.setRange(1, 32)
        threads.setValue(4)
        layout.addRow("CPU Threads:", threads)
        
        ir_path = QLineEdit()
        ir_path.setPlaceholderText("Path to OpenVINO IR model (.xml)")
        layout.addRow("Model Path:", ir_path)
        
        return group
    
    def _create_directml_options(self) -> QGroupBox:
        """Create DirectML options."""
        group = QGroupBox("DirectML Options")
        layout = QFormLayout(group)
        
        device_id = QSpinBox()
        device_id.setRange(0, 7)
        device_id.setValue(0)
        layout.addRow("GPU Device ID:", device_id)
        
        onnx_path = QLineEdit()
        onnx_path.setPlaceholderText("Path to .onnx model file")
        layout.addRow("Model Path:", onnx_path)
        
        # Info
        info = QLabel("Works with any DirectX 12 GPU (AMD, Intel, NVIDIA)")
        info.setObjectName("subheading")
        layout.addRow(info)
        
        return group
    
    def _on_backend_changed(self, backend_text: str):
        """Handle backend selection change."""
        # Map text to index
        backend_map = {
            "PyTorch Native (F5-TTS)": 0,
            "vLLM (Optimized)": 1,
            "TensorRT-LLM (Fastest)": 2,
            "ONNX Runtime (Portable)": 3,
            "llama.cpp (GGUF)": 4,
            "OpenVINO (Intel)": 5,
            "DirectML (Windows)": 6,
        }
        
        idx = backend_map.get(backend_text, 0)
        self.options_stack.setCurrentIndex(idx)
        
        # Check availability
        self._check_backend_availability()
        
        # Emit signal
        backend_name = backend_text.split(" (")[0].lower().replace(" ", "-")
        self.backend_changed.emit(backend_name)
    
    def _check_backend_availability(self):
        """Check if the selected backend is available."""
        try:
            from core.backend_registry import BackendRegistry, BackendType
            
            # Map combo text to backend type
            text = self.backend_combo.currentText()
            backend_map = {
                "PyTorch Native (F5-TTS)": BackendType.PYTORCH_NATIVE,
                "vLLM (Optimized)": BackendType.VLLM,
                "TensorRT-LLM (Fastest)": BackendType.TENSORRT,
                "ONNX Runtime (Portable)": BackendType.ONNX,
                "llama.cpp (GGUF)": BackendType.LLAMA_CPP,
                "OpenVINO (Intel)": BackendType.OPENVINO,
                "DirectML (Windows)": BackendType.DIRECTML,
            }
            
            backend_type = backend_map.get(text)
            if backend_type:
                backend_class = BackendRegistry.get(backend_type)
                is_available = backend_class.is_available()
                
                if is_available:
                    self.availability_label.setText("✅ Backend available")
                    self.availability_label.setStyleSheet("color: #00ff88;")
                else:
                    instructions = backend_class.get_install_instructions()
                    self.availability_label.setText(f"⚠️ Not installed. {instructions[:50]}...")
                    self.availability_label.setStyleSheet("color: #ff9800;")
                    
        except Exception as e:
            self.availability_label.setText(f"Status unknown: {e}")
            self.availability_label.setStyleSheet("color: #666;")
    
    def get_config(self) -> Dict:
        """Get current backend configuration."""
        # This would collect all settings from the active options panel
        return {
            "backend": self.backend_combo.currentText(),
            "auto_fallback": self.auto_fallback.isChecked(),
            # Add more settings based on active panel
        }
