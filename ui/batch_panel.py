"""
Batch Processing Panel
=======================
UI panel for managing batch TTS jobs.
"""

import logging
from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QFileDialog, QTextEdit,
    QSpinBox, QComboBox, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from core.adapter_manager import get_adapter_manager

logger = logging.getLogger(__name__)


class BatchPanel(QWidget):
    """
    Panel for managing batch TTS processing.
    
    Features:
    - Add jobs from text or file
    - Queue management (start/stop/clear)
    - Progress tracking
    - Job statistics
    """
    
    job_completed = pyqtSignal(str, object)  # job_id, audio_data
    
    def __init__(self, inference_manager=None, parent=None):
        super().__init__(parent)
        self.inference_manager = inference_manager
        self._batch_queue = None
        self._setup_ui()
        self._setup_queue()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # ===== Add Jobs Section =====
        add_group = QGroupBox("📝 Add Jobs")
        add_layout = QVBoxLayout(add_group)
        
        # Text input
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Enter text here. Each line becomes a separate job.\n"
            "Or load from a text file using the button below."
        )
        self.text_input.setMaximumHeight(100)
        add_layout.addWidget(self.text_input)
        
        # Controls
        btn_layout = QHBoxLayout()
        
        self.load_file_btn = QPushButton("📁 Load from File")
        self.load_file_btn.clicked.connect(self._load_from_file)
        btn_layout.addWidget(self.load_file_btn)
        
        self.voice_combo.addItems(["Default", "Custom..."])
        self.voice_combo.setMinimumWidth(150)
        btn_layout.addWidget(QLabel("Voice:"))
        btn_layout.addWidget(self.voice_combo)
        
        # Adapter selection
        self.adapter_combo = QComboBox()
        self.adapter_combo.setMinimumWidth(150)
        self.adapter_combo.addItem("None")
        self.refresh_adapters()
        btn_layout.addWidget(QLabel("Adapter:"))
        btn_layout.addWidget(self.adapter_combo)
        
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["Normal", "High", "Low"])
        btn_layout.addWidget(QLabel("Priority:"))
        btn_layout.addWidget(self.priority_combo)
        
        btn_layout.addStretch()
        
        self.add_btn = QPushButton("➕ Add to Queue")
        self.add_btn.clicked.connect(self._add_jobs)
        btn_layout.addWidget(self.add_btn)
        
        add_layout.addLayout(btn_layout)
        layout.addWidget(add_group)
        
        # ===== Queue Table =====
        queue_group = QGroupBox("📋 Job Queue")
        queue_layout = QVBoxLayout(queue_group)
        
        self.job_table = QTableWidget()
        self.job_table.setColumnCount(5)
        self.job_table.setHorizontalHeaderLabels([
            "ID", "Text", "Status", "Progress", "Duration"
        ])
        self.job_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.job_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        queue_layout.addWidget(self.job_table)
        
        # Queue controls
        ctrl_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self._start_queue)
        ctrl_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self._pause_queue)
        self.pause_btn.setEnabled(False)
        ctrl_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop_queue)
        self.stop_btn.setEnabled(False)
        ctrl_layout.addWidget(self.stop_btn)
        
        ctrl_layout.addStretch()
        
        self.cancel_btn = QPushButton("❌ Cancel Selected")
        self.cancel_btn.clicked.connect(self._cancel_selected)
        ctrl_layout.addWidget(self.cancel_btn)
        
        self.clear_btn = QPushButton("🗑 Clear Completed")
        self.clear_btn.clicked.connect(self._clear_completed)
        ctrl_layout.addWidget(self.clear_btn)
        
        queue_layout.addLayout(ctrl_layout)
        layout.addWidget(queue_group)
        
        # ===== Statistics =====
        stats_group = QGroupBox("📊 Statistics")
        stats_layout = QHBoxLayout(stats_group)
        
        self.total_label = QLabel("Total: 0")
        stats_layout.addWidget(self.total_label)
        
        self.pending_label = QLabel("Pending: 0")
        stats_layout.addWidget(self.pending_label)
        
        self.completed_label = QLabel("Completed: 0")
        stats_layout.addWidget(self.completed_label)
        
        self.failed_label = QLabel("Failed: 0")
        stats_layout.addWidget(self.failed_label)
        
        stats_layout.addStretch()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(200)
        stats_layout.addWidget(self.progress_bar)
        
        layout.addWidget(stats_group)
        
        # ===== Output Directory =====
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        
        self.output_path = QLabel("Not set (audio kept in memory)")
        self.output_path.setStyleSheet("color: #666;")
        output_layout.addWidget(self.output_path)
        
        self.set_output_btn = QPushButton("📁 Set")
        self.set_output_btn.clicked.connect(self._set_output_dir)
        output_layout.addWidget(self.set_output_btn)
        
        output_layout.addStretch()
        layout.addLayout(output_layout)
        
        # Update timer
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_display)
        self._update_timer.start(500)  # Update every 500ms
    
    def _setup_queue(self):
        """Set up the batch queue with callbacks."""
        try:
            from core.batch_queue import get_batch_queue, JobStatus
            
            self._batch_queue = get_batch_queue()
            self._batch_queue.inference_manager = self.inference_manager
            
            # Register callbacks
            self._batch_queue.on_job_complete(self._on_job_complete)
            self._batch_queue.on_job_error(self._on_job_error)
            
        except ImportError:
            logger.warning("Batch queue not available")
    
    def _load_from_file(self):
        """Load text from file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Text File", "",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.text_input.setText(content)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
    
    def _add_jobs(self):
        """Add jobs from text input."""
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter some text first.")
            return
        
        # Split into lines
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        if not lines:
            return
        
        try:
            from core.batch_queue import JobPriority
            
            # Get priority
            priority_map = {
                "Normal": JobPriority.NORMAL,
                "High": JobPriority.HIGH,
                "Low": JobPriority.LOW
            }
            priority = priority_map.get(self.priority_combo.currentText(), JobPriority.NORMAL)
            
            # Get voice
            voice = self.voice_combo.currentText()
            if voice == "Default":
                voice = None
            
            # Get adapter
            adapter = self.adapter_combo.currentText()
            if adapter == "None":
                adapter = None
            
            # Add jobs
            job_ids = self._batch_queue.add_batch(
                texts=lines,
                voice=voice,
                adapter=adapter,
                priority=priority
            )
            
            # Clear input
            self.text_input.clear()
            
            QMessageBox.information(
                self, "Jobs Added",
                f"Added {len(job_ids)} jobs to the queue."
            )
            
            self._update_display()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add jobs: {e}")
    
    def _start_queue(self):
        """Start processing the queue."""
        if self._batch_queue:
            self._batch_queue.start()
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
    
    def _pause_queue(self):
        """Pause/resume the queue."""
        if self._batch_queue:
            if self._batch_queue._paused:
                self._batch_queue.resume()
                self.pause_btn.setText("⏸ Pause")
            else:
                self._batch_queue.pause()
                self.pause_btn.setText("▶ Resume")
    
    def _stop_queue(self):
        """Stop the queue."""
        if self._batch_queue:
            self._batch_queue.stop()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
    
    def _cancel_selected(self):
        """Cancel selected jobs."""
        if not self._batch_queue:
            return
        
        for item in self.job_table.selectedItems():
            row = item.row()
            job_id = self.job_table.item(row, 0).text()
            self._batch_queue.cancel_job(job_id)
        
        self._update_display()
    
    def _clear_completed(self):
        """Clear completed jobs."""
        if self._batch_queue:
            self._batch_queue.clear_completed()
            self._update_display()
    
    def _set_output_dir(self):
        """Set output directory for saved files."""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_path.setText(path)
            self.output_path.setStyleSheet("color: #00ff88;")
    
    def _update_display(self):
        """Update the job table and statistics."""
        if not self._batch_queue:
            return
        
        jobs = self._batch_queue.get_all_jobs()
        
        # Update table
        self.job_table.setRowCount(len(jobs))
        
        for i, job in enumerate(jobs):
            self.job_table.setItem(i, 0, QTableWidgetItem(job.id))
            self.job_table.setItem(i, 1, QTableWidgetItem(job.text[:50] + "..." if len(job.text) > 50 else job.text))
            self.job_table.setItem(i, 2, QTableWidgetItem(job.status.value))
            self.job_table.setItem(i, 3, QTableWidgetItem(f"{job.progress}%"))
            self.job_table.setItem(i, 4, QTableWidgetItem(f"{job.duration_s:.1f}s" if job.duration_s > 0 else "--"))
        
        # Update statistics
        stats = self._batch_queue.get_statistics()
        pending = self._batch_queue.get_pending_count()
        
        self.total_label.setText(f"Total: {stats.total_jobs}")
        self.pending_label.setText(f"Pending: {pending}")
        self.completed_label.setText(f"Completed: {stats.completed_jobs}")
        self.failed_label.setText(f"Failed: {stats.failed_jobs}")
        
        # Update progress bar
        if stats.total_jobs > 0:
            progress = int((stats.completed_jobs + stats.failed_jobs) / stats.total_jobs * 100)
            self.progress_bar.setValue(progress)
        else:
            self.progress_bar.setValue(0)
    
    def _on_job_complete(self, job):
        """Handle job completion."""
        self.job_completed.emit(job.id, job.audio_data)
        self._update_display()
    
    def _on_job_error(self, job, error):
        """Handle job error."""
        logger.error(f"Job {job.id} failed: {error}")
        self._update_display()
    
    def set_inference_manager(self, manager):
        """Set the inference manager."""
        self.inference_manager = manager
        if self._batch_queue:
            self._batch_queue.inference_manager = manager
    
    def refresh_voices(self):
        """Refresh voice dropdown from profile database."""
        try:
            from core.profile_database import get_profile_database
            from PyQt6.QtCore import Qt
            
            # Store current selection
            current = self.voice_combo.currentText()
            
            # Clear and add defaults
            self.voice_combo.clear()
            self.voice_combo.addItem("Default")
            self.voice_combo.addItem("Custom...")
            
            # Load profiles
            db = get_profile_database()
            profiles = db.list_profiles()
            
            for profile in profiles:
                if profile.reference_samples:
                    sample = profile.reference_samples[0]
                    audio_path = sample.get("path", "")
                    
                    self.voice_combo.addItem(f"👤 {profile.name}")
                    idx = self.voice_combo.count() - 1
                    self.voice_combo.setItemData(idx, audio_path, Qt.ItemDataRole.UserRole)
            
            # Restore selection
            if current:
                idx = self.voice_combo.findText(current)
                if idx >= 0:
                    self.voice_combo.setCurrentIndex(idx)
            
        except Exception as e:
            logger.error(f"Failed to refresh voices in batch panel: {e}")

    def refresh_adapters(self):
        """Refresh adapter list."""
        try:
            current = self.adapter_combo.currentText()
            self.adapter_combo.clear()
            self.adapter_combo.addItem("None")
            
            manager = get_adapter_manager()
            manager.scan_adapters()
            adapters = manager.get_all_adapters()
            adapters.sort()
            self.adapter_combo.addItems(adapters)
            
            if current in adapters:
                self.adapter_combo.setCurrentText(current)
                
        except Exception as e:
            logger.error(f"Failed to refresh adapters in batch panel: {e}")
