# Codebase Structure

**Analysis Date:** 2026-03-31

## Directory Layout

```
VoiceSynthesisLocal/
├── main.py                     # Application entry point (GUI/CLI)
├── requirements.txt            # Python dependencies
├── requirements_ultimate.txt   # Full dependency list
├── test_all.py                 # Comprehensive test suite
├── test_imports.py             # Import verification tests
├── run_f5_tts.bat              # Windows batch launcher
├── setup_backends.bat          # Backend setup script
│
├── core/                       # Core synthesis logic
│   ├── __init__.py
│   ├── backend_base.py         # TTSBackend abstract base class
│   ├── backend_f5.py           # F5-TTS implementation
│   ├── backend_gptsovits.py    # GPT-SoVITS implementation
│   ├── backend_fish.py         # Fish Speech implementation
│   ├── backend_registry.py     # Backend registry & auto-selector
│   ├── backend_vllm.py         # vLLM optimized backend
│   ├── backend_tensorrt.py     # TensorRT backend
│   ├── backend_onnx.py         # ONNX runtime backend
│   ├── backend_llamacpp.py     # llama.cpp GGUF backend
│   ├── backend_openvino.py     # OpenVINO backend
│   ├── backend_directml.py     # DirectML backend
│   ├── inference_manager.py    # Central orchestration
│   ├── memory_manager.py       # VRAM management with LRU eviction
│   ├── profile_database.py     # SQLite voice profile storage
│   ├── voice_fusion.py         # Multi-sample embedding fusion
│   ├── adapter_manager.py      # Voice adapter (LoRA) discovery
│   ├── batch_queue.py          # Priority job queue
│   ├── chunking.py             # Text chunking & audio stitching
│   ├── quantization.py         # Quantization strategies
│   └── presets.py              # Optimization presets
│
├── ui/                         # PyQt6 user interface
│   ├── __init__.py
│   ├── main_window.py          # Main application window
│   ├── tts_panel.py            # TTS synthesis panel
│   ├── cloning_panel.py        # Voice cloning panel
│   ├── batch_panel.py          # Batch processing panel
│   ├── backend_config.py       # Backend configuration panel
│   ├── quantization_panel.py   # Quantization settings
│   ├── presets_panel.py        # Preset management
│   ├── monitor_widget.py       # GPU monitor widget
│   ├── waveform_widget.py      # Waveform visualization
│   └── styles.py               # Dark theme stylesheet
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   └── gpu_monitor.py          # pynvml GPU monitoring
│
├── adapters/                   # Voice adapters (LoRA/finetunes)
│   └── romanian/               # Romanian language adapters
│
├── training_workspace/         # Training utilities
│   ├── F5-TTS/                 # F5-TTS training config
│   ├── dataset/                # Training datasets
│   ├── raw_data/               # Raw audio data
│   ├── create_dummy_data.py
│   ├── prepare_dataset.py
│   ├── setup_training.py
│   └── run_train.bat
│
├── F5-TTS/                     # F5-TTS submodule
│   └── src/f5_tts/             # F5-TTS source
│
├── GPT-SoVITS/                 # GPT-SoVITS submodule
│   └── GPT_SoVITS/             # GPT-SoVITS source
│
└── fish-speech/                # Fish Speech submodule
    └── fish_speech/            # Fish Speech source
```

## Directory Purposes

**core/:**
- Purpose: Central synthesis logic, backend implementations, orchestration
- Contains: Backend classes, managers, database, fusion algorithms
- Key files: `inference_manager.py`, `backend_registry.py`, `memory_manager.py`

**ui/:**
- Purpose: PyQt6 graphical user interface components
- Contains: Panels, widgets, main window, styles
- Key files: `main_window.py`, `tts_panel.py`, `cloning_panel.py`

**utils/:**
- Purpose: Shared utility functions and classes
- Contains: GPU monitoring, helper functions
- Key files: `gpu_monitor.py`

**adapters/:**
- Purpose: Voice adapter checkpoints (LoRA, finetunes)
- Contains: Language-specific and voice-specific adapters
- Key files: Subdirectories per language/voice

**training_workspace/:**
- Purpose: Model training and fine-tuning
- Contains: Dataset preparation, training scripts, configs
- Key files: `prepare_dataset.py`, `setup_training.py`

**F5-TTS/, GPT-SoVITS/, fish-speech/:**
- Purpose: External TTS model repositories (git submodules)
- Contains: Original model implementations
- Note: Accessed via sys.path insertion in backend modules

## Key File Locations

**Entry Points:**
- `main.py`: Application entry point with GUI/CLI mode selection
- `test_all.py`: Comprehensive test suite (56+ tests)
- `test_imports.py`: Quick import verification

**Configuration:**
- `requirements.txt`: Minimal dependencies
- `requirements_ultimate.txt`: Full dependency list
- No central config file - settings stored in SQLite profiles

**Core Logic:**
- `core/inference_manager.py`: Central orchestration for all synthesis
- `core/backend_base.py`: Abstract base class defining TTSBackend interface
- `core/backend_registry.py`: Backend registration and auto-selection
- `core/memory_manager.py`: VRAM management with LRU eviction

**Testing:**
- `test_all.py`: Full test suite with decorators
- `test_imports.py`: Import smoke tests

## Naming Conventions

**Files:**
- Snake case for modules: `inference_manager.py`, `backend_f5.py`
- Test files prefixed: `test_all.py`, `test_imports.py`
- Batch scripts: `*.bat` for Windows

**Classes:**
- PascalCase: `InferenceManager`, `TTSBackend`, `VoiceFusion`
- Abstract base classes: `TTSBackend`
- Dataclasses: `SpeakerProfile`, `VoiceEmbedding`, `BackendCapabilities`

**Functions:**
- Snake case: `synthesize()`, `load_model()`, `get_profile()`

## Where to Add New Code

**New TTS Backend:**
1. Create `core/backend_newname.py` extending `TTSBackend`
2. Implement `load()`, `synthesize()`, `unload()` methods
3. Set `CAPABILITIES` with VRAM requirements and ratings
4. Register in `core/backend_registry.py` line 78-100

**New UI Panel:**
1. Create `ui/new_panel.py` extending `QWidget`
2. Add to `ui/main_window.py` tab widget (lines 84-115)
3. Connect signals in `_connect_signals()` (lines 321-334)

**New Utility:**
1. Add to `utils/module_name.py`
2. Export via `utils/__init__.py`

**Voice Adapters:**
1. Place checkpoint in `adapters/<language>/`
2. Automatically discovered by `AdapterManager.scan_adapters()`

**Training Configurations:**
1. Add to `training_workspace/F5-TTS/` or create new subdirectory
2. Use `prepare_dataset.py` for data preparation

## Special Directories

**Submodule Directories (F5-TTS, GPT-SoVITS, fish-speech):**
- Purpose: External TTS model implementations
- Generated: No - git submodules
- Committed: No - submodule references only
- Access: Added to sys.path in backend modules

**adapters/romanian/:**
- Purpose: Romanian language voice adapters
- Generated: Via training or external download
- Committed: Checkpoint files may be large, consider LFS

**training_workspace/dataset/, training_workspace/raw_data/:**
- Purpose: Training data storage
- Generated: Via data preparation scripts
- Committed: No - data excluded via .gitignore

**~/.config/voicesynth/profiles.db:**
- Purpose: SQLite database for speaker profiles
- Location: User config directory
- Created: On first profile save via `ProfileDatabase`

**~/.cache/voicesynth/quantized/:**
- Purpose: Cache for quantized models
- Location: User cache directory
- Created: By `QuantizationManager`

---

*Structure analysis: 2026-03-31*
