"""
Waveform Visualization Widget
==============================
Real-time waveform display with playback position tracking.
"""

import logging
from typing import Optional
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QLinearGradient, QFont

logger = logging.getLogger(__name__)


class WaveformWidget(QWidget):
    """
    Widget for displaying audio waveform.
    
    Features:
    - Waveform visualization
    - Playback position indicator
    - Zoom and scroll
    - Selection regions
    """
    
    position_clicked = pyqtSignal(float)  # Emits position in seconds when clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: int = 24000
        self._position: float = 0.0  # Current position in seconds
        self._duration: float = 0.0
        
        # Display settings
        self._waveform_color = QColor(0, 255, 136)  # Green
        self._position_color = QColor(255, 68, 68)  # Red
        self._background_color = QColor(15, 20, 41)
        self._grid_color = QColor(50, 50, 70)
        
        # Cached downsampled waveform
        self._display_samples: Optional[np.ndarray] = None
        self._max_samples: int = 2000  # Max samples to display
        
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
    
    def set_audio(self, audio_data: np.ndarray, sample_rate: int):
        """Set the audio data to display."""
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._duration = len(audio_data) / sample_rate
        self._position = 0.0
        
        # Create downsampled version for display
        self._create_display_samples()
        
        self.update()
    
    def _create_display_samples(self):
        """Create downsampled waveform for display."""
        if self._audio_data is None:
            self._display_samples = None
            return
        
        samples = len(self._audio_data)
        
        if samples <= self._max_samples:
            self._display_samples = self._audio_data
        else:
            # Downsample by taking min/max in each bucket
            bucket_size = samples // (self._max_samples // 2)
            
            # Reshape and get min/max per bucket
            trimmed = self._audio_data[:bucket_size * (samples // bucket_size)]
            reshaped = trimmed.reshape(-1, bucket_size)
            
            mins = reshaped.min(axis=1)
            maxs = reshaped.max(axis=1)
            
            # Interleave min/max
            self._display_samples = np.column_stack((mins, maxs)).ravel()
    
    def set_position(self, position_seconds: float):
        """Set the current playback position."""
        self._position = max(0, min(position_seconds, self._duration))
        self.update()
    
    def clear(self):
        """Clear the waveform display."""
        self._audio_data = None
        self._display_samples = None
        self._duration = 0.0
        self._position = 0.0
        self.update()
    
    def paintEvent(self, event):
        """Paint the waveform."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self._background_color)
        
        width = self.width()
        height = self.height()
        center_y = height // 2
        
        # Draw grid
        painter.setPen(QPen(self._grid_color, 1))
        painter.drawLine(0, center_y, width, center_y)
        
        # Time markers
        if self._duration > 0:
            # Every second
            for t in range(int(self._duration) + 1):
                x = int(t / self._duration * width)
                painter.drawLine(x, 0, x, height)
        
        # Draw waveform
        if self._display_samples is not None and len(self._display_samples) > 0:
            # Create gradient
            gradient = QLinearGradient(0, 0, 0, height)
            gradient.setColorAt(0, self._waveform_color.lighter(150))
            gradient.setColorAt(0.5, self._waveform_color)
            gradient.setColorAt(1, self._waveform_color.lighter(150))
            
            pen = QPen(QColor(self._waveform_color))
            pen.setWidth(1)
            painter.setPen(pen)
            
            samples = self._display_samples
            num_samples = len(samples)
            
            # Scale factor for height
            max_amp = max(abs(samples.min()), abs(samples.max()))
            if max_amp > 0:
                scale = (height * 0.45) / max_amp
            else:
                scale = 1
            
            # Draw waveform as envelope
            for i in range(num_samples - 1):
                x1 = int(i / num_samples * width)
                x2 = int((i + 1) / num_samples * width)
                
                y1 = int(center_y - samples[i] * scale)
                y2 = int(center_y - samples[i + 1] * scale)
                
                painter.drawLine(x1, y1, x2, y2)
        
        # Draw position indicator
        if self._duration > 0 and self._position > 0:
            pos_x = int(self._position / self._duration * width)
            
            painter.setPen(QPen(self._position_color, 2))
            painter.drawLine(pos_x, 0, pos_x, height)
            
            # Time label
            painter.setPen(QPen(self._position_color))
            painter.setFont(QFont("Segoe UI", 8))
            time_str = f"{self._position:.1f}s"
            painter.drawText(pos_x + 5, 15, time_str)
        
        # Duration label
        if self._duration > 0:
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.setFont(QFont("Segoe UI", 9))
            duration_str = f"Duration: {self._duration:.2f}s"
            painter.drawText(width - 120, height - 5, duration_str)
    
    def mousePressEvent(self, event):
        """Handle mouse click to seek."""
        if self._duration > 0:
            x = event.position().x()
            position = x / self.width() * self._duration
            self.position_clicked.emit(position)
    
    def mouseMoveEvent(self, event):
        """Show time on hover."""
        if self._duration > 0:
            x = event.position().x()
            time = x / self.width() * self._duration
            self.setToolTip(f"{time:.2f}s")


class RealtimeWaveformWidget(WaveformWidget):
    """
    Extended waveform widget with real-time playback tracking.
    
    Features:
    - Real-time position updates during playback
    - Audio stream position sync
    - Playback state management
    """
    
    playback_finished = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._is_playing = False
        self._playback_start_time: float = 0.0
        self._playback_start_position: float = 0.0
        
        # Timer for position updates
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._update_playback_position)
        self._playback_timer.setInterval(30)  # ~33 fps
    
    def start_playback(self, from_position: float = 0.0):
        """Start tracking playback position."""
        import time
        
        self._is_playing = True
        self._playback_start_time = time.time()
        self._playback_start_position = from_position
        self._position = from_position
        
        self._playback_timer.start()
    
    def stop_playback(self):
        """Stop tracking playback position."""
        self._is_playing = False
        self._playback_timer.stop()
    
    def pause_playback(self):
        """Pause playback tracking."""
        self._is_playing = False
        self._playback_timer.stop()
    
    def resume_playback(self):
        """Resume playback tracking."""
        import time
        
        self._is_playing = True
        self._playback_start_time = time.time()
        self._playback_start_position = self._position
        self._playback_timer.start()
    
    def _update_playback_position(self):
        """Update position based on elapsed time."""
        import time
        
        if not self._is_playing:
            return
        
        elapsed = time.time() - self._playback_start_time
        new_position = self._playback_start_position + elapsed
        
        if new_position >= self._duration:
            new_position = self._duration
            self.stop_playback()
            self.playback_finished.emit()
        
        self._position = new_position
        self.update()
    
    @property
    def is_playing(self) -> bool:
        return self._is_playing
    
    @property
    def current_position(self) -> float:
        return self._position


class SpectrogramWidget(QWidget):
    """
    Widget for displaying audio spectrogram.
    
    Features:
    - Mel spectrogram visualization
    - Color mapping
    - Time/frequency axes
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._spectrogram: Optional[np.ndarray] = None
        self._sample_rate: int = 24000
        
        self.setMinimumHeight(100)
        self.setMaximumHeight(200)
    
    def set_audio(self, audio_data: np.ndarray, sample_rate: int):
        """Compute and display spectrogram."""
        try:
            import librosa
            
            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(
                y=audio_data.astype(np.float32),
                sr=sample_rate,
                n_mels=80,
                fmax=8000
            )
            
            # Convert to dB
            S_db = librosa.power_to_db(S, ref=np.max)
            
            self._spectrogram = S_db
            self._sample_rate = sample_rate
            
            self.update()
            
        except ImportError:
            logger.warning("librosa not available for spectrogram")
            self._spectrogram = None
    
    def paintEvent(self, event):
        """Paint the spectrogram."""
        painter = QPainter(self)
        
        # Dark background
        painter.fillRect(self.rect(), QColor(15, 20, 41))
        
        if self._spectrogram is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No spectrogram data")
            return
        
        width = self.width()
        height = self.height()
        
        # Get spectrogram dimensions
        n_mels, n_frames = self._spectrogram.shape
        
        # Normalize spectrogram to 0-1
        S_norm = (self._spectrogram - self._spectrogram.min()) / (self._spectrogram.max() - self._spectrogram.min() + 1e-8)
        
        # Draw each cell as a colored rectangle
        cell_width = max(1, width // n_frames)
        cell_height = max(1, height // n_mels)
        
        for t in range(n_frames):
            x = int(t / n_frames * width)
            
            for m in range(n_mels):
                y = height - int((m + 1) / n_mels * height)
                
                # Color mapping (blue to red)
                val = S_norm[m, t]
                r = int(val * 255)
                g = int(val * 100)
                b = int((1 - val) * 200)
                
                painter.fillRect(x, y, cell_width + 1, cell_height + 1, QColor(r, g, b))


class AudioVisualizationPanel(QWidget):
    """
    Complete audio visualization panel with waveform and controls.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Waveform
        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)
        
        # Info row
        info_layout = QHBoxLayout()
        
        self.duration_label = QLabel("Duration: --")
        info_layout.addWidget(self.duration_label)
        
        self.position_label = QLabel("Position: --")
        info_layout.addWidget(self.position_label)
        
        info_layout.addStretch()
        
        self.zoom_in_btn = QPushButton("🔍+")
        self.zoom_in_btn.setMaximumWidth(40)
        info_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("🔍-")
        self.zoom_out_btn.setMaximumWidth(40)
        info_layout.addWidget(self.zoom_out_btn)
        
        layout.addLayout(info_layout)
        
        # Connect signals
        self.waveform.position_clicked.connect(self._on_position_clicked)
    
    def set_audio(self, audio_data: np.ndarray, sample_rate: int):
        """Set audio for visualization."""
        self.waveform.set_audio(audio_data, sample_rate)
        
        duration = len(audio_data) / sample_rate
        self.duration_label.setText(f"Duration: {duration:.2f}s")
    
    def set_position(self, position: float):
        """Update playback position."""
        self.waveform.set_position(position)
        self.position_label.setText(f"Position: {position:.2f}s")
    
    def _on_position_clicked(self, position: float):
        """Handle waveform click."""
        self.position_label.setText(f"Position: {position:.2f}s")
    
    def clear(self):
        """Clear the display."""
        self.waveform.clear()
        self.duration_label.setText("Duration: --")
        self.position_label.setText("Position: --")
