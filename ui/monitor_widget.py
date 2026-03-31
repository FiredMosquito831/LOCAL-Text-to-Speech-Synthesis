"""
GPU Monitor Widget
===================
Real-time GPU memory and status monitoring with breakdown.
"""

import logging
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QGroupBox, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from utils.gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)


class MemoryBar(QProgressBar):
    """Custom progress bar with color coding."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(True)
        self._update_style(0)
    
    def setValue(self, value: int):
        super().setValue(value)
        self._update_style(value)
    
    def _update_style(self, value: int):
        if value >= 90:
            color = "#ff4444"  # Red
        elif value >= 75:
            color = "#ff9800"  # Orange
        elif value >= 50:
            color = "#ffeb3b"  # Yellow
        else:
            color = "#00ff88"  # Green
        
        self.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #333;
                border-radius: 4px;
                text-align: center;
                background: #1a1a2e;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 3px;
            }}
        """)


class GPUMonitorWidget(QGroupBox):
    """
    Widget for monitoring GPU status.
    
    Features:
    - Real-time VRAM usage
    - Temperature and utilization
    - Memory breakdown by component
    - Cache clearing controls
    - Whisper offload button
    """
    
    memory_warning = pyqtSignal(float)  # Emitted when memory exceeds threshold
    
    def __init__(self, parent=None):
        super().__init__("📊 GPU Monitor", parent)
        self._setup_ui()
        
        self._gpu_monitor = GPUMonitor()
        self._memory_manager = None
        
        # Update timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_stats)
        self._timer.start(1000)  # Update every second
        
        # Initial update
        self._update_stats()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # ===== GPU Name =====
        self.gpu_name_label = QLabel("Detecting GPU...")
        self.gpu_name_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(self.gpu_name_label)
        
        # ===== VRAM Bar =====
        vram_section = QVBoxLayout()
        
        vram_header = QHBoxLayout()
        vram_header.addWidget(QLabel("VRAM:"))
        self.vram_text = QLabel("0 / 0 GB")
        vram_header.addWidget(self.vram_text)
        vram_header.addStretch()
        vram_section.addLayout(vram_header)
        
        self.vram_bar = MemoryBar()
        self.vram_bar.setRange(0, 100)
        vram_section.addWidget(self.vram_bar)
        
        layout.addLayout(vram_section)
        
        # ===== Stats Row =====
        stats_layout = QHBoxLayout()
        
        self.temp_label = QLabel("🌡 --°C")
        stats_layout.addWidget(self.temp_label)
        
        self.util_label = QLabel("⚡ --%")
        stats_layout.addWidget(self.util_label)
        
        self.power_label = QLabel("🔌 --W")
        stats_layout.addWidget(self.power_label)
        
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        # ===== Memory Breakdown =====
        self.breakdown_group = QGroupBox("Memory Breakdown")
        breakdown_layout = QVBoxLayout(self.breakdown_group)
        
        self.breakdown_table = QTableWidget(6, 2)
        self.breakdown_table.setHorizontalHeaderLabels(["Component", "Usage"])
        self.breakdown_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.breakdown_table.verticalHeader().setVisible(False)
        self.breakdown_table.setMaximumHeight(180)
        
        # Initialize with placeholder data
        components = [
            ("TTS Model", "--"),
            ("Vocoder", "--"),
            ("Whisper", "--"),
            ("KV Cache", "--"),
            ("Audio Buffer", "--"),
            ("CUDA Overhead", "~500 MB"),
        ]
        for i, (comp, usage) in enumerate(components):
            self.breakdown_table.setItem(i, 0, QTableWidgetItem(comp))
            self.breakdown_table.setItem(i, 1, QTableWidgetItem(usage))
        
        breakdown_layout.addWidget(self.breakdown_table)
        layout.addWidget(self.breakdown_group)
        
        # ===== Status =====
        self.status_label = QLabel("Status: ✅ Healthy")
        self.status_label.setObjectName("subheading")
        layout.addWidget(self.status_label)
        
        # ===== Control Buttons =====
        btn_layout = QHBoxLayout()
        
        self.clear_cache_btn = QPushButton("🧹 Clear Cache")
        self.clear_cache_btn.clicked.connect(self._clear_cache)
        btn_layout.addWidget(self.clear_cache_btn)
        
        self.offload_btn = QPushButton("📤 Offload Whisper")
        self.offload_btn.clicked.connect(self._offload_whisper)
        btn_layout.addWidget(self.offload_btn)
        
        layout.addLayout(btn_layout)
        
        self.gc_btn = QPushButton("⚠️ Force GC")
        self.gc_btn.clicked.connect(self._force_gc)
        layout.addWidget(self.gc_btn)
    
    def _update_stats(self):
        """Update GPU statistics."""
        stats = self._gpu_monitor.get_stats()
        
        if stats.available:
            # GPU name
            self.gpu_name_label.setText(stats.name)
            
            # VRAM (convert MB to GB)
            used = stats.used_memory_mb / 1024
            total = stats.total_memory_mb / 1024
            percent = stats.memory_percent
            
            self.vram_bar.setValue(int(percent))
            self.vram_text.setText(f"{used:.1f} / {total:.1f} GB ({percent:.0f}%)")
            
            # Temp, Util
            self.temp_label.setText(f"🌡 {stats.temperature_c}°C")
            self.util_label.setText(f"⚡ {stats.utilization_percent:.0f}%")
            self.power_label.setText("🔌 --W")  # Power not available in current GPUStats
            
            # Status based on memory
            if percent >= 90:
                self.status_label.setText("Status: 🔴 Critical (VRAM low!)")
                self.status_label.setStyleSheet("color: #ff4444;")
                self.memory_warning.emit(percent)
            elif percent >= 75:
                self.status_label.setText("Status: 🟠 Warning (VRAM high)")
                self.status_label.setStyleSheet("color: #ff9800;")
            else:
                self.status_label.setText(f"Status: ✅ Healthy ({total - used:.1f} GB free)")
                self.status_label.setStyleSheet("color: #00ff88;")
            
            # Update breakdown if memory manager is connected
            self._update_breakdown()
        else:
            self.gpu_name_label.setText("No GPU detected")
            self.vram_bar.setValue(0)
            self.vram_text.setText("CPU Mode")
            self.status_label.setText("Status: 💻 CPU Only")
    
    def _update_breakdown(self):
        """Update memory breakdown table."""
        try:
            from core.memory_manager import get_memory_manager
            
            manager = get_memory_manager()
            breakdown = manager.get_memory_breakdown()
            
            data = breakdown.to_dict()
            
            row = 0
            for component, size_mb in data.items():
                if row < self.breakdown_table.rowCount():
                    self.breakdown_table.setItem(row, 0, QTableWidgetItem(component))
                    if size_mb > 0:
                        self.breakdown_table.setItem(row, 1, QTableWidgetItem(f"{size_mb:.0f} MB"))
                    else:
                        self.breakdown_table.setItem(row, 1, QTableWidgetItem("--"))
                row += 1
                
        except Exception as e:
            logger.debug(f"Could not update breakdown: {e}")
    
    def _clear_cache(self):
        """Clear CUDA cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("CUDA cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
        
        self._update_stats()
    
    def _offload_whisper(self):
        """Offload Whisper model to CPU."""
        try:
            from core.memory_manager import get_memory_manager
            
            manager = get_memory_manager()
            
            # Simply evict whisper component (moves to CPU/unloads)
            manager.evict_by_component("whisper")
            
            logger.info("Whisper model offloaded/evicted")
            self._update_stats()
            
        except Exception as e:
            logger.error(f"Failed to offload Whisper: {e}")
    
    def _force_gc(self):
        """Force garbage collection."""
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        
        try:
            from core.memory_manager import get_memory_manager
            manager = get_memory_manager()
            manager.force_garbage_collect()
        except:
            pass
        
        logger.info("Forced garbage collection")
        self._update_stats()
    
    def set_memory_manager(self, manager):
        """Connect to the memory manager for detailed breakdown."""
        self._memory_manager = manager
    
    def stop_monitoring(self):
        """Stop the monitoring timer."""
        self._timer.stop()
