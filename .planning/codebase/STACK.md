# Technology Stack

**Analysis Date:** 2026-03-31

## Languages

**Primary:**
- Python 3.x - Core application language for all TTS backends, UI, and processing pipelines

**Secondary:**
- CUDA C++ (via PyTorch) - GPU-accelerated operations for neural network inference
- ONNX IR - Model interchange format for cross-platform deployment

## Runtime

**Environment:**
- Python 3.10+ (implied by dependency constraints in `requirements_ultimate.txt`)

**Package Manager:**
- pip - Python package management
- Lockfile: Not present (using `requirements_ultimate.txt` with pinned versions)

## Frameworks

**Core:**
- PyTorch 2.5.1+cu121 (`torch`, `torchaudio`, `torchvision`) - Deep learning framework for model inference
- Transformers 4.50.0 - Hugging Face transformer models
- Accelerate 0.17.1 - Multi-device model deployment

**UI Framework:**
- PyQt6 6.10.2 (`ui/main_window.py`, `ui/tts_panel.py`, etc.) - Desktop GUI framework
  - Main window: `ui/main_window.py`
  - TTS panel: `ui/tts_panel.py`
  - Voice cloning panel: `ui/cloning_panel.py`
  - GPU monitor widget: `ui/monitor_widget.py`
  - Waveform visualization: `ui/waveform_widget.py`

**Audio Processing:**
- Librosa 0.10.2 - Audio feature extraction (mel spectrograms)
- SoundFile 0.13.1 - Audio I/O operations
- PyDub 0.25.1 - Audio manipulation
- ffmpeg-python 0.2.0 - FFmpeg integration

**Testing:**
- Custom test framework (`test_all.py`, `test_imports.py`) - Ad-hoc testing scripts

**Build/Dev:**
- Setuptools (via `pyproject.toml` in F5-TTS subproject)
- Git submodules for F5-TTS, GPT-SoVITS, Fish-Speech

## Key Dependencies

**Critical:**
- `torch>=2.5.1+cu121` - Core ML runtime with CUDA 12.1 support
- `transformers>=4.50.0` - Pretrained model architectures
- `onnxruntime>=1.23.2` - Cross-platform inference engine
- `onnxruntime-directml>=1.23.0` - Windows DirectML acceleration
- `onnxruntime-gpu>=1.23.2` - CUDA-accelerated ONNX inference
- `openvino>=2025.4.1` - Intel hardware acceleration
- `f5-tts` (editable install from SWivid/F5-TTS) - Primary TTS backend

**TTS Backends:**
- `f5-tts` - Flow matching TTS (primary backend, `core/backend_f5.py`)
- `gpt-sovits` - GPT-based voice cloning (`core/backend_gptsovits.py`)
- `fish-speech` - Autoregressive TTS with LLaMA backbone (`core/backend_fish.py`)

**Voice/Transcription:**
- `faster-whisper>=1.2.1` - Reference audio transcription
- `openai-whisper>=20250625` - Alternative Whisper implementation
- `encodec>=0.1.1` - Neural audio codec
- `vocos>=0.1.0` - Vocoder for audio synthesis

**Quantization:**
- `bitsandbytes>=0.41.1` - 4-bit/8-bit quantization (Windows wheels from jllllll)
- `ctranslate2>=4.6.3` - Efficient transformer inference

**Infrastructure:**
- `pydantic>=2.10.6` - Data validation
- `numpy>=1.26.4` - Numerical operations
- `scipy>=1.17.0` - Scientific computing
- `pandas>=2.3.3` - Data processing
- `huggingface-hub>=0.36.0` - Model distribution

**GPU Monitoring:**
- `pynvml>=13.0.1` - NVIDIA GPU monitoring
- `nvidia-ml-py>=13.590.44` - NVIDIA management library

## Configuration

**Environment:**
- PyQt6-based desktop application
- CLI mode available via `python main.py --cli`
- Models cached to `~/.cache/voicesynth/` (inferred)
- TensorRT engines cached to `~/.cache/trt_engines/` (`core/backend_tensorrt.py`)

**Build:**
- `requirements_ultimate.txt` - Complete dependency list (252 packages)
- `requirements.txt` - Empty/minimal dependencies
- `F5-TTS/pyproject.toml` - F5-TTS subproject build configuration
- `fish-speech/pyproject.toml` - Fish-Speech subproject configuration

## Platform Requirements

**Development:**
- Windows 10/11 (primary target, WSL2 compatible)
- NVIDIA GPU with CUDA 12.1 support (recommended: 6GB+ VRAM)
- AMD/Intel GPU supported via DirectML or OpenVINO
- Minimum 16GB system RAM recommended

**GPU Support Matrix:**
- **NVIDIA CUDA**: Full support via PyTorch (`backend_f5.py`, `backend_gptsovits.py`)
- **NVIDIA TensorRT**: High-performance path (`core/backend_tensorrt.py`)
- **AMD ROCm**: Limited support via ONNX Runtime ROCm EP (`core/backend_onnx.py`)
- **Intel GPU/NPU**: OpenVINO acceleration (`core/backend_openvino.py`)
- **Windows DirectML**: Universal Windows GPU via DirectX 12 (`core/backend_directml.py`)

**Production:**
- Desktop deployment (PyQt6 application)
- Local inference only (no cloud dependencies for synthesis)
- Hugging Face for model downloads
- Optional: vLLM server deployment for API access

## Memory Management

**Stack Components:**
- `core/memory_manager.py` - AdaptiveMemoryManager with LRU eviction
- `core/quantization.py` - QuantizationManager with 13 quantization types
- Target VRAM budget: 5.5GB (optimized for 6GB consumer GPUs)
- CPU offload support via `CPUOffloadManager`

**Quantization Support:**
- FP32, FP16, BF16 - Full/half/bfloat precision
- INT8 dynamic/static - PyTorch and ONNX quantization
- INT4 GPTQ/AWQ - 4-bit quantization for low VRAM
- INT4 NF4 (bitsandbytes) - NormalFloat 4-bit
- GGUF Q2-Q8 - llama.cpp format quantization
- FP8 E4M3 - Ada/Hopper GPU optimization

## Entry Points

**Main Application:**
- `main.py` - Primary entry point (GUI and CLI modes)
- `python main.py` - Launch PyQt6 GUI
- `python main.py --cli --text "..."` - CLI synthesis

**Test Scripts:**
- `test_all.py` - Comprehensive test suite
- `test_imports.py` - Import validation

**Backend-Specific:**
- `F5-TTS/src/f5_tts/infer/infer_cli.py` - F5-TTS CLI
- `F5-TTS/src/f5_tts/infer/infer_gradio.py` - F5-TTS Gradio UI
- `fish-speech/fish_speech/inference_engine/` - Fish Speech engine

---

*Stack analysis: 2026-03-31*
