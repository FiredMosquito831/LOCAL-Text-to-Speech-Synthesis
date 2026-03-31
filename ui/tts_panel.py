"""
TTS Panel Widget
=================
Main text-to-speech interface with voice selection,
text input, generation controls, and audio output.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import logging
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTextEdit, QPushButton, QComboBox, QSlider,
    QGroupBox, QSpinBox, QDoubleSpinBox, QFileDialog,
    QProgressBar, QFrame, QSplitter, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont
from core.adapter_manager import get_adapter_manager

logger = logging.getLogger(__name__)


class TTSWorker(QThread):
    """Background worker for TTS inference."""
    
    progress = pyqtSignal(int, str)  # percent, message
    finished = pyqtSignal(object, int)  # audio data, sample_rate
    error = pyqtSignal(str)
    
    def __init__(self, inference_manager, text: str, voice: Optional[str], settings: dict):
        super().__init__()
        self.inference_manager = inference_manager
        self.text = text
        self.voice = voice
        self.settings = settings
        self._cancelled = False
    
    def _progress_callback(self, percent: int, message: str):
        """Forward progress to Qt signal."""
        if not self._cancelled:
            self.progress.emit(percent, message)
    
    def run(self):
        try:
            # Set voice if specified
            if self.voice and self.voice not in ["Default", "Custom Reference..."]:
                self.inference_manager.set_voice(self.voice)
            
            # Run synthesis with progress callback
            audio, sample_rate = self.inference_manager.synthesize(
                text=self.text,
                voice=self.voice if self.voice not in ["Default", "Custom Reference..."] else None,
                progress_callback=self._progress_callback,
                **self.settings
            )
            
            if not self._cancelled:
                self.finished.emit(audio, sample_rate)
            
        except Exception as e:
            logger.error(f"TTS inference error: {e}", exc_info=True)
            if not self._cancelled:
                self.error.emit(str(e))
    
    def cancel(self):
        self._cancelled = True


class AudioPlayerWidget(QGroupBox):
    """Audio player with waveform display."""
    
    def __init__(self, parent=None):
        super().__init__("🔊 Output", parent)
        self._setup_ui()
        self.audio_data = None
        self.sample_rate = 24000
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Waveform visualization
        try:
            from ui.waveform_widget import WaveformWidget
            self.waveform = WaveformWidget()
            layout.addWidget(self.waveform)
            self._has_waveform = True
        except ImportError:
            # Fallback to simple label
            self.waveform = QLabel("No audio generated yet")
            self.waveform.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.waveform.setMinimumHeight(80)
            self.waveform.setStyleSheet(
                "background-color: #0f1429; border-radius: 6px; color: #666;"
            )
            layout.addWidget(self.waveform)
            self._has_waveform = False
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._play_audio)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop_audio)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        self.time_label = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.time_label)
        
        controls_layout.addStretch()
        
        self.save_wav_btn = QPushButton("💾 Save WAV")
        self.save_wav_btn.clicked.connect(lambda: self._save_audio("wav"))
        self.save_wav_btn.setEnabled(False)
        controls_layout.addWidget(self.save_wav_btn)
        
        self.save_mp3_btn = QPushButton("💾 Save MP3")
        self.save_mp3_btn.clicked.connect(lambda: self._save_audio("mp3"))
        self.save_mp3_btn.setEnabled(False)
        controls_layout.addWidget(self.save_mp3_btn)
        
        layout.addLayout(controls_layout)
    
    def set_audio(self, audio_data, sample_rate: int = 24000):
        """Set the audio data to display and play."""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        
        # Enable buttons
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.save_wav_btn.setEnabled(True)
        self.save_mp3_btn.setEnabled(True)
        
        # Update duration and waveform
        if audio_data is not None:
            duration = len(audio_data) / sample_rate
            self.time_label.setText(f"00:00 / {int(duration//60):02d}:{int(duration%60):02d}")
            
            if self._has_waveform:
                self.waveform.set_audio(audio_data, sample_rate)
            else:
                self.waveform.setText(f"✓ Audio ready ({duration:.1f}s)")
                self.waveform.setStyleSheet(
                    "background-color: #0f1429; border-radius: 6px; color: #00ff88;"
                )
    
    def _play_audio(self):
        """Play the audio using sounddevice."""
        if self.audio_data is None:
            return
        try:
            import sounddevice as sd
            sd.play(self.audio_data, self.sample_rate)
        except ImportError:
            logger.warning("sounddevice not installed")
            QMessageBox.warning(self, "Error", "sounddevice not installed")
        except Exception as e:
            logger.error(f"Playback error: {e}")
            QMessageBox.warning(self, "Playback Error", str(e))
    
    def _stop_audio(self):
        """Stop audio playback."""
        try:
            import sounddevice as sd
            sd.stop()
        except:
            pass
    
    def _save_audio(self, format: str):
        """Save audio to file."""
        if self.audio_data is None:
            return
        
        file_filter = "WAV Files (*.wav)" if format == "wav" else "MP3 Files (*.mp3)"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "", file_filter
        )
        
        if path:
            try:
                import soundfile as sf
                sf.write(path, self.audio_data, self.sample_rate)
                logger.info(f"Audio saved to {path}")
                QMessageBox.information(self, "Saved", f"Audio saved to:\n{path}")
            except Exception as e:
                logger.error(f"Save error: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")


class TTSPanel(QWidget):
    """
    Main TTS panel widget.
    
    Features:
    - Voice/model selection
    - Text input with stats
    - Generation settings (NFE, sampler, speed, etc.)
    - Progress display
    - Audio output with playback/save
    """
    
    # Signal emitted when generation starts/stops
    generation_started = pyqtSignal()
    generation_finished = pyqtSignal(bool)  # success
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._worker: Optional[TTSWorker] = None
        self._inference_manager = None
        self._current_audio: Optional[np.ndarray] = None
        self._current_sample_rate: int = 24000
        
        # Setup keyboard shortcuts
        self._setup_shortcuts()
        
        # Load saved voice profiles
        self.refresh_voices()
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        from PyQt6.QtGui import QShortcut, QKeySequence
        
        # Ctrl+Enter to generate
        generate_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        generate_shortcut.activated.connect(self._start_generation)
        
        # Ctrl+Shift+Enter for preview  
        preview_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Return"), self)
        preview_shortcut.activated.connect(self._generate_preview)
        
        # Ctrl+S to save audio
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self._save_current_audio)
        
        # Escape to stop generation
        stop_shortcut = QShortcut(QKeySequence("Escape"), self)
        stop_shortcut.activated.connect(self._stop_generation)
    
    def _save_current_audio(self):
        """Save current audio via keyboard shortcut."""
        if hasattr(self, 'audio_player') and self.audio_player._audio is not None:
            self.audio_player._save_audio()
    
    def refresh_voices(self):
        """Load voice profiles from database into the dropdown."""
        try:
            from core.profile_database import get_profile_database
            
            # Store current selection
            current_text = self.voice_combo.currentText() if self.voice_combo.count() > 0 else ""
            
            # Clear and add default options
            self.voice_combo.clear()
            self.voice_combo.addItem("Default")
            self.voice_combo.addItem("Custom Reference...")
            
            # Load profiles from database
            db = get_profile_database()
            profiles = db.list_profiles()
            
            for profile in profiles:
                if profile.reference_samples:
                    # Get primary audio path
                    sample = profile.reference_samples[0]
                    audio_path = sample.get("path", "")
                    
                    # Add to combo with path as data
                    self.voice_combo.addItem(f"👤 {profile.name}")
                    idx = self.voice_combo.count() - 1
                    self.voice_combo.setItemData(idx, audio_path, Qt.ItemDataRole.UserRole)
                    self.voice_combo.setItemData(idx, profile.id, Qt.ItemDataRole.UserRole + 1)
            
            # Restore selection if possible
            if current_text:
                idx = self.voice_combo.findText(current_text)
                if idx >= 0:
                    self.voice_combo.setCurrentIndex(idx)
            
            logger.info(f"Loaded {len(profiles)} voice profiles")
            
        except Exception as e:
            logger.error(f"Failed to load voice profiles: {e}")
        
    def _setup_ui(self):
        """Initialize the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # ===== Top bar: Voice, Language, and Backend selection =====
        top_bar = QHBoxLayout()
        
        # Voice selection
        top_bar.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["Default", "Custom Reference..."])
        self.voice_combo.setMinimumWidth(200)
        self.voice_combo.currentTextChanged.connect(self._on_voice_changed)
        top_bar.addWidget(self.voice_combo)
        
        top_bar.addSpacing(15)
        
        # Language selection
        top_bar.addWidget(QLabel("🌍 Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            "English (EN)",
            "Romanian (RO)",
            "Chinese (ZH)",
            "Auto-detect"
        ])
        self.language_combo.setToolTip(
            "Select synthesis language.\n"
            "Romanian requires F5-TTS backend.\n"
            "Auto-detect will analyze the input text."
        )
        self.language_combo.currentTextChanged.connect(self._on_language_changed)
        top_bar.addWidget(self.language_combo)
        
        top_bar.addSpacing(15)
        
        # Backend selection
        top_bar.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["F5-TTS", "GPT-SoVITS", "Fish Speech"])
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        top_bar.addWidget(self.backend_combo)
        
        top_bar.addStretch()
        layout.addLayout(top_bar)
        
        # ===== Main content: Text input + Settings =====
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- Left: Text Input ---
        text_group = QGroupBox("📝 Text Input")
        text_layout = QVBoxLayout(text_group)
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText(
            "Enter the text you want to synthesize...\n\n"
            "Tips:\n"
            "• Use punctuation for natural pauses\n"
            "• [laugh], [sigh] for expressions (F5-TTS)\n"
            "• Keep sentences under 200 chars for best quality"
        )
        self.text_edit.textChanged.connect(self._update_text_stats)
        text_layout.addWidget(self.text_edit)
        
        # Text stats
        stats_layout = QHBoxLayout()
        self.char_count_label = QLabel("Characters: 0")
        self.char_count_label.setObjectName("subheading")
        stats_layout.addWidget(self.char_count_label)
        
        self.word_count_label = QLabel("Words: 0")
        self.word_count_label.setObjectName("subheading")
        stats_layout.addWidget(self.word_count_label)
        
        self.duration_estimate_label = QLabel("Est. Duration: ~0s")
        self.duration_estimate_label.setObjectName("subheading")
        stats_layout.addWidget(self.duration_estimate_label)
        
        stats_layout.addStretch()
        text_layout.addLayout(stats_layout)
        
        content_splitter.addWidget(text_group)
        
        # --- Right: Generation Settings ---
        settings_group = QGroupBox("⚙️ Generation Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Voice Selection
        voice_layout = QHBoxLayout()
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(200)
        self.voice_combo.currentTextChanged.connect(self._on_voice_changed)
        
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setToolTip("Refresh Voices")
        self.refresh_btn.clicked.connect(self.refresh_voices)
        self.refresh_btn.setFixedWidth(30)
        
        voice_layout.addWidget(QLabel("Voice:"))
        voice_layout.addWidget(self.voice_combo)
        voice_layout.addWidget(self.refresh_btn)
        
        # Adapter Selection (New)
        adapter_layout = QHBoxLayout()
        self.adapter_combo = QComboBox()
        self.adapter_combo.setMinimumWidth(200)
        self.adapter_combo.addItem("None")
        self.adapter_combo.currentTextChanged.connect(self._on_adapter_changed)
        self.refresh_adapters() # Populate initial list
        
        adapter_layout.addWidget(QLabel("Adapter:"))
        adapter_layout.addWidget(self.adapter_combo)
        
        settings_layout.addLayout(voice_layout)
        settings_layout.addLayout(adapter_layout)
        
        # NFE Steps
        nfe_layout = QHBoxLayout()
        nfe_layout.addWidget(QLabel("NFE Steps:"))
        self.nfe_spin = QSpinBox()
        self.nfe_spin.setRange(4, 64)
        self.nfe_spin.setValue(16)
        self.nfe_spin.setToolTip("Number of function evaluations. Higher = better quality, slower.")
        nfe_layout.addWidget(self.nfe_spin)
        nfe_layout.addWidget(QLabel("(8=fast, 32=quality)"))
        nfe_layout.addStretch()
        settings_layout.addLayout(nfe_layout)
        
        # Sampler
        sampler_layout = QHBoxLayout()
        sampler_layout.addWidget(QLabel("Sampler:"))
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(["Euler", "Midpoint", "RK4"])
        sampler_layout.addWidget(self.sampler_combo)
        sampler_layout.addStretch()
        settings_layout.addLayout(sampler_layout)
        
        # Speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickInterval(25)
        speed_layout.addWidget(self.speed_slider)
        self.speed_value_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_value_label)
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_value_label.setText(f"{v/100:.1f}x")
        )
        settings_layout.addLayout(speed_layout)
        
        # Seed
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 999999999)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText("Random")
        self.seed_spin.setToolTip("-1 for random seed")
        seed_layout.addWidget(self.seed_spin)
        seed_layout.addStretch()
        settings_layout.addLayout(seed_layout)
        
        # Chunking
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("Chunking:"))
        self.chunk_combo = QComboBox()
        self.chunk_combo.addItems(["Sentence", "Fixed (200 chars)", "Paragraph", "None"])
        chunk_layout.addWidget(self.chunk_combo)
        chunk_layout.addStretch()
        settings_layout.addLayout(chunk_layout)
        
        settings_layout.addStretch()
        content_splitter.addWidget(settings_group)
        
        # Set splitter sizes
        content_splitter.setSizes([600, 300])
        layout.addWidget(content_splitter)
        
        # ===== Status/Progress =====
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("subheading")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # ===== Action buttons =====
        action_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("▶ Generate")
        self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.clicked.connect(self._start_generation)
        self.generate_btn.setMinimumHeight(40)
        action_layout.addWidget(self.generate_btn)
        
        self.preview_btn = QPushButton("🎧 Preview 5s")
        self.preview_btn.clicked.connect(self._generate_preview)
        action_layout.addWidget(self.preview_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop_generation)
        self.stop_btn.setEnabled(False)
        action_layout.addWidget(self.stop_btn)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
        
        # ===== Audio output =====
        self.audio_player = AudioPlayerWidget()
        layout.addWidget(self.audio_player)
    
    def _update_text_stats(self):
        """Update character/word counts and duration estimate."""
        text = self.text_edit.toPlainText()
        chars = len(text)
        words = len(text.split()) if text.strip() else 0
        
        # Rough estimate: 150 words per minute
        duration_sec = (words / 150) * 60
        
        self.char_count_label.setText(f"Characters: {chars}")
        self.word_count_label.setText(f"Words: {words}")
        self.duration_estimate_label.setText(f"Est. Duration: ~{int(duration_sec)}s")
    
    def _get_generation_settings(self) -> dict:
        """Collect current generation settings."""
        # Extract language code from selection
        lang_text = self.language_combo.currentText()
        lang_map = {
            "English (EN)": "en",
            "Romanian (RO)": "ro",
            "Chinese (ZH)": "zh",
            "Auto-detect": "auto"
        }
        language = lang_map.get(lang_text, "en")
        
        return {
            "nfe_steps": self.nfe_spin.value(),
            "sampler": self.sampler_combo.currentText().lower(),
            "speed": self.speed_slider.value() / 100.0,
            "seed": self.seed_spin.value(),
            "language": language,
        }
    
    def _on_voice_changed(self, voice: str):
        """Handle voice selection change."""
        if voice == "Custom Reference...":
            # Open file dialog
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Reference Audio",
                "",
                "Audio Files (*.wav *.mp3 *.flac *.ogg)"
            )
            if path:
                # Add as new voice option
                name = Path(path).stem
                self.voice_combo.insertItem(2, name)  # After Default and Custom
                self.voice_combo.setItemData(2, path, Qt.ItemDataRole.UserRole)
                self.voice_combo.setCurrentIndex(2)
                
                # Set in inference manager
                if self._inference_manager:
                    self._inference_manager.set_voice(path)
            else:
                self.voice_combo.setCurrentIndex(0)
        
        elif voice == "Default":
            # Clear voice (use model default)
            if self._inference_manager:
                self._inference_manager.set_voice("")
        
        elif voice.startswith("👤 "):
            # Profile selected - get stored audio path from item data
            idx = self.voice_combo.currentIndex()
            audio_path = self.voice_combo.itemData(idx, Qt.ItemDataRole.UserRole)
            
            if audio_path and Path(audio_path).exists():
                logger.info(f"Using profile voice: {voice} → {audio_path}")
                if self._inference_manager:
                    self._inference_manager.set_voice(audio_path)
            else:
                # Path invalid, try to resolve from profile name
                profile_name = voice.replace("👤 ", "")
                logger.info(f"Resolving profile by name: {profile_name}")
                if self._inference_manager:
                    self._inference_manager.set_voice(profile_name)
        
        else:
            # Direct voice name (custom added)
            idx = self.voice_combo.currentIndex()
            audio_path = self.voice_combo.itemData(idx, Qt.ItemDataRole.UserRole)
            
            if audio_path and self._inference_manager:
                self._inference_manager.set_voice(audio_path)
            elif self._inference_manager:
                # Fallback: try to resolve as profile name
                self._inference_manager.set_voice(voice)
    
    def _on_language_changed(self, language: str):
        """Handle language selection change."""
        logger.info(f"Language changed to: {language}")
        
        # Show warning if Romanian selected with non-F5-TTS backend
        if "Romanian" in language:
            current_backend = self.backend_combo.currentText()
            if current_backend != "F5-TTS":
                QMessageBox.information(
                    self,
                    "Language Note",
                    "Romanian works best with F5-TTS backend.\n"
                    "Consider switching to F5-TTS for optimal results."
                )
    
    def _on_backend_changed(self, backend: str):
        """Handle backend selection change."""
        logger.info(f"Backend changed to: {backend}")
        if self._inference_manager:
            backend_map = {
                "F5-TTS": "f5-tts",
                "GPT-SoVITS": "gpt-sovits", 
                "Fish Speech": "fish-speech"
            }
            self._inference_manager.switch_backend(backend_map.get(backend, "f5-tts"))
    
    def _start_generation(self):
        """Start the TTS generation process."""
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter some text to synthesize.")
            return
        
        if not self._inference_manager:
            QMessageBox.warning(self, "No Model", "Inference manager not initialized.")
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.generate_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Generating...")
        
        # Get settings
        settings = self._get_generation_settings()
        
        # Get voice
        voice = self.voice_combo.currentText()
        voice_path = self.voice_combo.currentData(Qt.ItemDataRole.UserRole)
        if voice_path:
            voice = voice_path
        elif voice in ["Default", "Custom Reference..."]:
            voice = None
        
        # Emit signal
        self.generation_started.emit()
        
        # Start worker thread
        logger.info(f"Starting generation: {len(text)} chars, settings={settings}")
        
        self._worker = TTSWorker(
            self._inference_manager,
            text,
            voice,
            settings
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_generation_finished)
        self._worker.error.connect(self._on_generation_error)
        self._worker.start()
    
    def refresh_adapters(self):
        """Refresh the list of available adapters."""
        from core.adapter_manager import get_adapter_manager
        current = self.adapter_combo.currentText()
        self.adapter_combo.blockSignals(True)
        self.adapter_combo.clear()
        self.adapter_combo.addItem("None")
        
        manager = get_adapter_manager()
        manager.scan_adapters() # Refresh scan
        adapters = manager.get_all_adapters()
        
        # Sort for better UX (Romanian first if exists?)
        adapters.sort()
        self.adapter_combo.addItems(adapters)
        
        # Restore selection if possible
        if current in adapters:
            self.adapter_combo.setCurrentText(current)
        else:
            self.adapter_combo.setCurrentIndex(0)
            
        self.adapter_combo.blockSignals(False)

    def _on_adapter_changed(self, adapter_name: str):
        """Handle adapter selection change."""
        if self._inference_manager:
            adapter = adapter_name if adapter_name != "None" else None
            self._inference_manager.set_adapter(adapter)
            logger.info(f"UI: Adapter changed to {adapter}")
        
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    @pyqtSlot(int, str)
    def _on_progress(self, percent: int, message: str):
        """Handle progress updates from worker."""
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{percent}% - {message}")
        self.status_label.setText(message)
    
    @pyqtSlot(object, int)
    def _on_generation_finished(self, audio: np.ndarray, sample_rate: int):
        """Handle successful generation."""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Set audio in player
        self.audio_player.set_audio(audio, sample_rate)
        
        duration = len(audio) / sample_rate
        self.status_label.setText(f"✓ Generated {duration:.1f}s of audio")
        
        logger.info(f"Generation complete: {duration:.1f}s audio")
        self.generation_finished.emit(True)
        
        self._worker = None
    
    @pyqtSlot(str)
    def _on_generation_error(self, error: str):
        """Handle generation error."""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"✗ Error: {error}")
        
        QMessageBox.critical(self, "Generation Error", f"Failed to generate audio:\n\n{error}")
        
        logger.error(f"Generation error: {error}")
        self.generation_finished.emit(False)
        
        self._worker = None
    
    def _generate_preview(self):
        """Generate a 5-second preview."""
        # Truncate text for preview
        text = self.text_edit.toPlainText().strip()
        preview_text = " ".join(text.split()[:20])  # First 20 words
        
        if not preview_text:
            QMessageBox.warning(self, "No Text", "Please enter some text to preview.")
            return
        
        if not self._inference_manager:
            QMessageBox.warning(self, "No Model", "Inference manager not initialized.")
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Generating preview...")
        self.generate_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Generating preview...")
        
        # Get settings
        settings = self._get_generation_settings()
        
        # Emit signal
        self.generation_started.emit()
        
        # Start worker with preview text directly (don't modify the text field)
        self._worker = TTSWorker(
            self._inference_manager,
            preview_text,  # Use preview text directly
            None,
            settings
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_generation_finished)
        self._worker.error.connect(self._on_generation_error)
        self._worker.start()
    
    def _stop_generation(self):
        """Stop the current generation."""
        if self._worker:
            self._worker.cancel()
            self._worker.wait(1000)  # Wait up to 1 second
            self._worker = None
        
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopped")
        logger.info("Generation stopped")
    
    def set_inference_manager(self, manager):
        """Set the inference manager for actual TTS generation."""
        self._inference_manager = manager
