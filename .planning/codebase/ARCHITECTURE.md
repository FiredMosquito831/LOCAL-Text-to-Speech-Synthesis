# Architecture

**Analysis Date:** 2026-03-31

## Pattern Overview

**Overall:** Multi-Backend TTS Orchestration with Registry Pattern

VoiceSynth Studio is a voice synthesis application that abstracts multiple TTS backends (F5-TTS, GPT-SoVITS, Fish Speech) behind a unified inference interface. The architecture follows a layered approach with clear separation between UI, orchestration, backend implementations, and utility services.

**Key Characteristics:**
- **Registry Pattern** - Central backend registry with auto-selection based on hardware capabilities
- **Strategy Pattern** - Pluggable TTS backends implementing common `TTSBackend` interface
- **Observer Pattern** - Progress callbacks and event signals throughout the inference pipeline
- **Singleton Pattern** - Global managers for memory, profiles, adapters, and batch queue
- **Producer-Consumer Pattern** - Background worker threads for non-blocking TTS synthesis

## Layers

**UI Layer (PyQt6):**
- Purpose: User interface and interaction handling
- Location: `ui/`
- Contains: `MainWindow`, `TTSPanel`, `VoiceCloningPanel`, `BatchPanel`, `GPUMonitorWidget`, waveform visualization
- Depends on: `core.inference_manager`, `core.profile_database`, `utils.gpu_monitor`
- Used by: End users for voice synthesis and cloning

**Orchestration Layer:**
- Purpose: Coordinate inference across backends, manage state
- Location: `core/inference_manager.py`, `core/backend_registry.py`
- Contains: `InferenceManager`, `BackendRegistry`, `BackendAutoSelector`, `QuantizationManager`
- Depends on: Individual backend implementations
- Used by: UI panels, CLI entry point, batch queue

**Backend Abstraction Layer:**
- Purpose: Unified interface for diverse TTS models
- Location: `core/backend_base.py`, `core/backend_*.py`
- Contains: `TTSBackend` (abstract), `F5TTSBackend`, `GPTSoVITSBackend`, `FishSpeechBackend`, etc.
- Depends on: External TTS model libraries (f5_tts, gpt_sovits, fish_speech)
- Used by: `InferenceManager`

**Service Layer:**
- Purpose: Cross-cutting concerns (memory, profiles, fusion, batch processing)
- Location: `core/memory_manager.py`, `core/profile_database.py`, `core/voice_fusion.py`, `core/batch_queue.py`
- Contains: `AdaptiveMemoryManager`, `ProfileDatabase`, `VoiceFusion`, `BatchQueue`
- Depends on: torch, sqlite3, numpy
- Used by: Orchestration layer and UI

**Utility Layer:**
- Purpose: Common utilities and helpers
- Location: `utils/`, `core/chunking.py`, `core/quantization.py`
- Contains: `GPUMonitor`, `TextChunker`, `AudioStitcher`, `QualityScorer`
- Depends on: pynvml, librosa, soundfile
- Used by: All layers

## Data Flow

**Text-to-Speech Flow:**

1. User enters text in `ui/tts_panel.py` and clicks generate
2. `TTSPanel` creates `TTSWorker` thread with text and settings
3. Worker calls `InferenceManager.synthesize()` with progress callback
4. `InferenceManager` routes to appropriate backend (`F5TTSBackend.synthesize()`)
5. Backend preprocesses reference audio via `preprocess_ref_audio_text()`
6. Backend runs `infer_process()` with model and vocoder
7. Audio returned as `(np.ndarray, sample_rate)` tuple
8. Worker emits `finished` signal with audio data
9. `TTSPanel` displays waveform via `WaveformWidget` and enables playback

**Voice Cloning Flow:**

1. User drops audio file in `AudioDropZone` (`ui/cloning_panel.py`)
2. `TranscriptionWorker` runs faster-whisper in background
3. Transcript displayed for editing
4. User saves profile via `ProfileDatabase.save_profile()`
5. `VoiceFusion.extract_embedding()` processes reference audio
6. Embedding stored in SQLite database with quality score
7. Profile appears in voice combo boxes across all panels

**Batch Processing Flow:**

1. User adds jobs via `BatchPanel` or `BatchQueue.add_job()`
2. Jobs queued in `PriorityQueue` ordered by `JobPriority`
3. Background worker thread processes jobs via `_worker_loop()`
4. Each job calls `InferenceManager.synthesize()` with progress callback
5. Audio saved via `_save_audio()` using soundfile
6. Job status updated to `COMPLETED`, callbacks notified

## Key Abstractions

**TTSBackend (Abstract Base Class):**
- Purpose: Common interface for all TTS implementations
- Examples: `core/backend_base.py` lines 50-122
- Pattern: Abstract methods `load()`, `synthesize()`, `unload()` with capability metadata

```python
class TTSBackend(ABC):
    CAPABILITIES = BackendCapabilities(...)

    @abstractmethod
    def load(self, progress_callback: Optional[Callable] = None):
        """Load the model."""
        pass

    @abstractmethod
    def synthesize(self, text: str, ref_audio_path: Optional[str] = None, ...) -> Tuple[np.ndarray, int]:
        """Synthesize speech. Returns (audio_data, sample_rate)."""
        pass

    @abstractmethod
    def unload(self):
        """Unload model and free memory."""
        pass
```

**BackendCapabilities:**
- Purpose: Declare VRAM requirements, quality ratings, speed multipliers
- Location: `core/backend_base.py` lines 26-47
- Used by: `BackendAutoSelector` for hardware-aware selection

**VoiceEmbedding:**
- Purpose: Container for extracted voice embeddings with quality metadata
- Location: `core/voice_fusion.py` lines 16-24
- Pattern: Dataclass with embedding array, quality score, duration

**SpeakerProfile:**
- Purpose: Persistent voice profile with samples and embeddings
- Location: `core/profile_database.py` lines 20-64
- Pattern: Dataclass with SQLite serialization, VSP export format

## Entry Points

**GUI Entry Point:**
- Location: `main.py` lines 105-136
- Triggers: `python main.py` (default)
- Responsibilities: Initialize QApplication, create MainWindow, apply theme, run event loop

**CLI Entry Point:**
- Location: `main.py` lines 139-179
- Triggers: `python main.py --cli --text "..."`
- Responsibilities: Parse args, create InferenceManager, synthesize, save output

**Test Entry Points:**
- Location: `test_all.py`, `test_imports.py`
- Triggers: `python test_all.py`
- Responsibilities: Comprehensive test suite for all modules

## Error Handling

**Strategy:** Layered exception handling with logging and graceful degradation

**Patterns:**
- **Try-except with fallback** - Backends return placeholder audio on failure (`core/inference_manager.py` line 303)
- **Progress callback error forwarding** - Worker threads emit error signals to UI (`ui/tts_panel.py` lines 63-66)
- **Import error handling** - Optional features check availability (`utils/gpu_monitor.py` lines 18-26)
- **Model load error recovery** - Adapter unload restores base model state (`core/backend_f5.py` lines 347-353)

```python
# Error handling pattern from TTSWorker
try:
    audio, sample_rate = self.inference_manager.synthesize(...)
    self.finished.emit(audio, sample_rate)
except Exception as e:
    logger.error(f"TTS inference error: {e}", exc_info=True)
    self.error.emit(str(e))
```

## Cross-Cutting Concerns

**Logging:** Python logging module with module-level loggers, format: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`

**Validation:** Quality scoring for reference samples via `QualityScorer.score_sample()` (SNR, clipping, duration checks)

**Authentication:** Not applicable (local application)

**Memory Management:** `AdaptiveMemoryManager` with LRU eviction, VRAM budget tracking, component-level profiling (`core/memory_manager.py`)

**Quantization:** `QuantizationManager` supports FP16, INT8, INT4, GGUF formats with backend-specific application (`core/quantization.py`)

**Voice Fusion:** Multiple embedding fusion strategies (average, quality-weighted, duration-weighted) in `VoiceFusion.fuse_embeddings()` (`core/voice_fusion.py` lines 193-241)

---

*Architecture analysis: 2026-03-31*
