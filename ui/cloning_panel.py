"""
Voice Cloning Panel
===================
Reference audio upload, transcription, and speaker profile management.
"""

from pathlib import Path
from typing import Optional, List
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTextEdit, QPushButton, QComboBox, QGroupBox,
    QFileDialog, QListWidget, QListWidgetItem,
    QLineEdit, QProgressBar, QFrame, QSplitter,
    QMessageBox, QInputDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent

logger = logging.getLogger(__name__)


class TranscriptionWorker(QThread):
    """Background worker for audio transcription using faster-whisper."""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)  # transcript
    error = pyqtSignal(str)
    
    def __init__(self, audio_path: Path, model_size: str = "small", compute_type: str = "int8"):
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        self.compute_type = compute_type
    
    def run(self):
        try:
            self.progress.emit(10, "Loading Whisper model...")
            
            from faster_whisper import WhisperModel
            
            model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type=self.compute_type
            )
            
            self.progress.emit(50, "Transcribing...")
            
            segments, info = model.transcribe(
                str(self.audio_path),
                beam_size=5,
                vad_filter=True
            )
            
            transcript = " ".join([seg.text.strip() for seg in segments])
            
            self.progress.emit(100, "Done!")
            self.finished.emit(transcript)
            
        except ImportError:
            self.error.emit("faster-whisper not installed. Run: pip install faster-whisper")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self.error.emit(str(e))


class AudioDropZone(QLabel):
    """Drag-and-drop zone for audio files."""
    
    file_dropped = pyqtSignal(Path)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("📁 Drop audio file here\nor click to browse\n\nSupported: WAV, MP3, FLAC, OGG")
        self.setMinimumHeight(120)
        self.setStyleSheet("""
            QLabel {
                background-color: #0f1429;
                border: 2px dashed #0f3460;
                border-radius: 10px;
                color: #666;
                padding: 20px;
            }
            QLabel:hover {
                border-color: #1a4a7a;
                color: #888;
            }
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def mousePressEvent(self, event):
        """Handle click to browse."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )
        if path:
            self.file_dropped.emit(Path(path))
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self.styleSheet().replace("#0f3460", "#00d9ff"))
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.styleSheet().replace("#00d9ff", "#0f3460"))
    
    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self.styleSheet().replace("#00d9ff", "#0f3460"))
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]:
                self.file_dropped.emit(path)
                break


class VoiceCloningPanel(QWidget):
    """
    Voice cloning panel with:
    - Reference audio upload
    - Faster-whisper transcription
    - Speaker profile management
    - Quality scoring
    """
    
    voice_selected = pyqtSignal(str, str)  # profile_name, profile_path
    profile_saved = pyqtSignal()  # Emitted when a profile is saved/deleted
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._current_audio_path: Optional[Path] = None
        self._transcription_worker: Optional[TranscriptionWorker] = None
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Create horizontal splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ===== Left: Reference Audio Section =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Drop zone
        ref_group = QGroupBox("🎤 Reference Audio")
        ref_layout = QVBoxLayout(ref_group)
        
        self.drop_zone = AudioDropZone()
        self.drop_zone.file_dropped.connect(self._on_file_dropped)
        ref_layout.addWidget(self.drop_zone)
        
        # Audio info
        self.audio_info_label = QLabel("No audio loaded")
        self.audio_info_label.setObjectName("subheading")
        ref_layout.addWidget(self.audio_info_label)
        
        # Playback button
        audio_controls = QHBoxLayout()
        self.play_ref_btn = QPushButton("▶ Play Reference")
        self.play_ref_btn.setEnabled(False)
        self.play_ref_btn.clicked.connect(self._play_reference)
        audio_controls.addWidget(self.play_ref_btn)
        audio_controls.addStretch()
        ref_layout.addLayout(audio_controls)
        
        left_layout.addWidget(ref_group)
        
        # Transcription section
        trans_group = QGroupBox("📝 Transcription (Faster-Whisper)")
        trans_layout = QVBoxLayout(trans_group)
        
        # Model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"])
        self.whisper_model_combo.setCurrentText("small")
        model_row.addWidget(self.whisper_model_combo)
        
        model_row.addWidget(QLabel("Compute:"))
        self.compute_type_combo = QComboBox()
        self.compute_type_combo.addItems(["int8", "float16", "float32"])
        model_row.addWidget(self.compute_type_combo)
        
        model_row.addStretch()
        trans_layout.addLayout(model_row)
        
        # Transcribe button
        self.transcribe_btn = QPushButton("🔄 Auto-Transcribe")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.clicked.connect(self._start_transcription)
        trans_layout.addWidget(self.transcribe_btn)
        
        # Progress
        self.trans_progress = QProgressBar()
        self.trans_progress.setVisible(False)
        trans_layout.addWidget(self.trans_progress)
        
        # Transcript text
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setPlaceholderText(
            "Transcript will appear here after auto-transcription.\n"
            "You can also type or paste the transcript manually."
        )
        self.transcript_edit.setMaximumHeight(100)
        trans_layout.addWidget(self.transcript_edit)
        
        left_layout.addWidget(trans_group)
        
        # Profile creation
        profile_create_group = QGroupBox("💾 Save as Profile")
        profile_create_layout = QVBoxLayout(profile_create_group)
        
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.profile_name_edit = QLineEdit()
        self.profile_name_edit.setPlaceholderText("e.g., John Narrator")
        name_row.addWidget(self.profile_name_edit)
        profile_create_layout.addLayout(name_row)
        
        tags_row = QHBoxLayout()
        tags_row.addWidget(QLabel("Tags:"))
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("male, narrator, deep (comma separated)")
        tags_row.addWidget(self.tags_edit)
        profile_create_layout.addLayout(tags_row)
        
        self.save_profile_btn = QPushButton("💾 Save Profile")
        self.save_profile_btn.setEnabled(False)
        self.save_profile_btn.clicked.connect(self._save_profile)
        profile_create_layout.addWidget(self.save_profile_btn)
        
        left_layout.addWidget(profile_create_group)
        left_layout.addStretch()
        
        splitter.addWidget(left_widget)
        
        # ===== Right: Saved Profiles =====
        profiles_group = QGroupBox("👤 Saved Profiles")
        profiles_layout = QVBoxLayout(profiles_group)
        
        # Search
        self.profile_search = QLineEdit()
        self.profile_search.setPlaceholderText("🔍 Search profiles...")
        self.profile_search.textChanged.connect(self._filter_profiles)
        profiles_layout.addWidget(self.profile_search)
        
        # Profile list
        self.profile_list = QListWidget()
        self.profile_list.itemDoubleClicked.connect(self._load_profile)
        profiles_layout.addWidget(self.profile_list)
        
        # Profile actions
        profile_actions = QHBoxLayout()
        self.use_profile_btn = QPushButton("✓ Use Selected")
        self.use_profile_btn.clicked.connect(self._use_selected_profile)
        profile_actions.addWidget(self.use_profile_btn)
        
        self.delete_profile_btn = QPushButton("🗑 Delete")
        self.delete_profile_btn.clicked.connect(self._delete_profile)
        profile_actions.addWidget(self.delete_profile_btn)
        
        self.export_btn = QPushButton("📤 Export")
        self.export_btn.clicked.connect(self._export_profile)
        profile_actions.addWidget(self.export_btn)
        
        profiles_layout.addLayout(profile_actions)
        
        splitter.addWidget(profiles_group)
        splitter.setSizes([500, 300])
        
        layout.addWidget(splitter)
        
        # Load existing profiles
        self._load_profiles_list()
    
    def _on_file_dropped(self, path: Path):
        """Handle dropped audio file."""
        self._current_audio_path = path
        
        # Get audio info
        try:
            import soundfile as sf
            info = sf.info(str(path))
            duration = info.duration
            sample_rate = info.samplerate
            self.audio_info_label.setText(
                f"✓ {path.name} | {duration:.1f}s | {sample_rate}Hz"
            )
        except:
            self.audio_info_label.setText(f"✓ {path.name}")
        
        # Enable buttons
        self.play_ref_btn.setEnabled(True)
        self.transcribe_btn.setEnabled(True)
        self.save_profile_btn.setEnabled(True)
        
        # Update drop zone appearance
        self.drop_zone.setText(f"✓ {path.name}\n\nClick to change")
        
        logger.info(f"Loaded reference audio: {path}")
    
    def _play_reference(self):
        """Play the reference audio."""
        if not self._current_audio_path:
            return
        try:
            import sounddevice as sd
            import soundfile as sf
            data, sr = sf.read(str(self._current_audio_path))
            sd.play(data, sr)
        except Exception as e:
            logger.error(f"Playback error: {e}")
    
    def _start_transcription(self):
        """Start faster-whisper transcription."""
        if not self._current_audio_path:
            return
        
        self.trans_progress.setVisible(True)
        self.trans_progress.setValue(0)
        self.transcribe_btn.setEnabled(False)
        
        self._transcription_worker = TranscriptionWorker(
            self._current_audio_path,
            self.whisper_model_combo.currentText(),
            self.compute_type_combo.currentText()
        )
        self._transcription_worker.progress.connect(
            lambda p, m: self.trans_progress.setValue(p)
        )
        self._transcription_worker.finished.connect(self._on_transcription_done)
        self._transcription_worker.error.connect(self._on_transcription_error)
        self._transcription_worker.start()
    
    def _on_transcription_done(self, transcript: str):
        """Handle completed transcription."""
        self.transcript_edit.setText(transcript)
        self.trans_progress.setVisible(False)
        self.transcribe_btn.setEnabled(True)
        logger.info(f"Transcription complete: {len(transcript)} chars")
    
    def _on_transcription_error(self, error: str):
        """Handle transcription error."""
        self.trans_progress.setVisible(False)
        self.transcribe_btn.setEnabled(True)
        QMessageBox.critical(self, "Transcription Error", error)
    
    def _save_profile(self):
        """Save current reference as a voice profile to SQLite database."""
        name = self.profile_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Name Required", "Please enter a profile name.")
            return
        
        if not self._current_audio_path:
            QMessageBox.warning(self, "No Audio", "Please load a reference audio first.")
            return
        
        try:
            from core.profile_database import (
                get_profile_database, SpeakerProfile, QualityScorer
            )
            import uuid
            
            # Get transcript
            transcript = self.transcript_edit.toPlainText().strip()
            
            # Parse tags
            tags_text = self.tags_edit.text().strip()
            tags = [t.strip() for t in tags_text.split(',') if t.strip()]
            
            # Score the sample
            scorer = QualityScorer()
            score_result = scorer.score_sample(self._current_audio_path)
            quality_score = score_result.get('overall', 0.5)
            
            # Create profile
            profile = SpeakerProfile(
                id=str(uuid.uuid4()),
                name=name,
                tags=tags,
                description=f"Created from {self._current_audio_path.name}",
                reference_samples=[{
                    "path": str(self._current_audio_path),
                    "transcript": transcript,
                    "duration_s": score_result.get('duration_s', 0)
                }],
                quality_score=quality_score
            )
            
            # Save to database
            db = get_profile_database()
            success = db.save_profile(profile)
            
            if success:
                # Add to list
                item = QListWidgetItem(f"👤 {name}")
                item.setData(Qt.ItemDataRole.UserRole, profile.id)
                item.setToolTip(f"Quality: {quality_score:.0%} | Tags: {', '.join(tags)}")
                self.profile_list.addItem(item)
                
                # Clear inputs
                self.profile_name_edit.clear()
                self.tags_edit.clear()
                
                quality_stars = "★" * int(quality_score * 5) + "☆" * (5 - int(quality_score * 5))
                QMessageBox.information(
                    self, "Saved",
                    f"Profile '{name}' saved successfully!\n\n"
                    f"Quality Score: {quality_stars}"
                )
                # Notify other panels to refresh
                self.profile_saved.emit()
            else:
                QMessageBox.warning(self, "Error", "Failed to save profile to database.")
                
        except ImportError as e:
            logger.error(f"Import error: {e}")
            # Fallback to in-memory only
            item = QListWidgetItem(f"👤 {name}")
            item.setData(Qt.ItemDataRole.UserRole, str(self._current_audio_path))
            self.profile_list.addItem(item)
            QMessageBox.information(self, "Saved", f"Profile '{name}' saved (in-memory only).")
        except Exception as e:
            logger.error(f"Save profile error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save profile: {e}")
    
    def _load_profiles_list(self):
        """Load existing profiles from SQLite database."""
        try:
            from core.profile_database import get_profile_database
            
            db = get_profile_database()
            profiles = db.list_profiles()
            
            for profile in profiles:
                item = QListWidgetItem(f"👤 {profile.name}")
                item.setData(Qt.ItemDataRole.UserRole, profile.id)
                item.setToolTip(f"Quality: {profile.quality_score:.0%} | Tags: {', '.join(profile.tags)}")
                self.profile_list.addItem(item)
            
            logger.info(f"Loaded {len(profiles)} profiles from database")
            
        except ImportError:
            logger.warning("Profile database not available")
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
    
    def _filter_profiles(self, text: str):
        """Filter profile list by search text."""
        for i in range(self.profile_list.count()):
            item = self.profile_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())
    
    def _load_profile(self, item: QListWidgetItem):
        """Load a profile for editing."""
        profile_id = item.data(Qt.ItemDataRole.UserRole)
        
        try:
            from core.profile_database import get_profile_database
            
            db = get_profile_database()
            profile = db.get_profile(profile_id)
            
            if profile and profile.reference_samples:
                # Load first sample
                sample = profile.reference_samples[0]
                audio_path = Path(sample.get("path", ""))
                
                if audio_path.exists():
                    self._on_file_dropped(audio_path)
                    
                    # Restore transcript
                    transcript = sample.get("transcript", "")
                    if transcript:
                        self.transcript_edit.setText(transcript)
                    
                    # Show profile info
                    self.profile_name_edit.setText(profile.name)
                    self.tags_edit.setText(", ".join(profile.tags))
                else:
                    QMessageBox.warning(
                        self, "File Not Found",
                        f"Reference audio not found:\n{audio_path}"
                    )
                    
        except ImportError:
            # Fallback - profile_id is audio path
            if profile_id:
                self._on_file_dropped(Path(profile_id))
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
    
    def _use_selected_profile(self):
        """Use the selected profile for TTS."""
        item = self.profile_list.currentItem()
        if not item:
            return
        
        try:
            from core.profile_database import get_profile_database
            
            profile_id = item.data(Qt.ItemDataRole.UserRole)
            db = get_profile_database()
            profile = db.get_profile(profile_id)
            
            if profile and profile.reference_samples:
                name = profile.name
                path = profile.reference_samples[0].get("path", "")
                self.voice_selected.emit(name, path)
                logger.info(f"Selected voice profile: {name}")
            else:
                # Fallback
                name = item.text().replace("👤 ", "")
                path = item.data(Qt.ItemDataRole.UserRole)
                self.voice_selected.emit(name, path)
                
        except ImportError:
            name = item.text().replace("👤 ", "")
            path = item.data(Qt.ItemDataRole.UserRole)
            self.voice_selected.emit(name, path)
    
    def _delete_profile(self):
        """Delete the selected profile from database."""
        item = self.profile_list.currentItem()
        if not item:
            return
        
        reply = QMessageBox.question(
            self, "Delete Profile",
            f"Delete profile '{item.text()}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from core.profile_database import get_profile_database
                
                profile_id = item.data(Qt.ItemDataRole.UserRole)
                db = get_profile_database()
                db.delete_profile(profile_id)
                
            except ImportError:
                pass
            except Exception as e:
                logger.error(f"Failed to delete profile: {e}")
            
            self.profile_list.takeItem(self.profile_list.row(item))
    
    def _export_profile(self):
        """Export profile to .vsp file."""
        item = self.profile_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Selection", "Please select a profile to export.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Profile", "", "Voice Profile (*.vsp)"
        )
        
        if path:
            try:
                from core.profile_database import get_profile_database
                
                profile_id = item.data(Qt.ItemDataRole.UserRole)
                db = get_profile_database()
                
                success = db.export_profile(profile_id, Path(path))
                
                if success:
                    QMessageBox.information(
                        self, "Exported",
                        f"Profile exported to:\n{path}"
                    )
                else:
                    QMessageBox.warning(self, "Error", "Failed to export profile.")
                    
            except ImportError:
                QMessageBox.warning(
                    self, "Not Available",
                    "Profile database not available for export."
                )
            except Exception as e:
                logger.error(f"Export error: {e}")
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
    
    def _import_profile(self):
        """Import profile from .vsp file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Profile", "", "Voice Profile (*.vsp)"
        )
        
        if path:
            try:
                from core.profile_database import get_profile_database
                
                db = get_profile_database()
                profile = db.import_profile(Path(path))
                
                if profile:
                    item = QListWidgetItem(f"👤 {profile.name}")
                    item.setData(Qt.ItemDataRole.UserRole, profile.id)
                    self.profile_list.addItem(item)
                    
                    QMessageBox.information(
                        self, "Imported",
                        f"Profile '{profile.name}' imported successfully!"
                    )
                else:
                    QMessageBox.warning(self, "Error", "Failed to import profile.")
                    
            except Exception as e:
                logger.error(f"Import error: {e}")
                QMessageBox.critical(self, "Error", f"Import failed: {e}")

