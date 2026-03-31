# External Integrations

**Analysis Date:** 2026-03-31

## TTS Backends

### F5-TTS (Primary)
- **Integration:** `core/backend_f5.py` - `F5TTSBackend` class
- **Source:** SWivid/F5-TTS (git submodule, editable install)
- **Model Type:** Flow matching with DiT (Diffusion Transformer) backbone
- **Models:**
  - `F5TTS_v1_Base` - Primary model (1.25M steps)
  - `E2TTS_Base` - Alternative architecture
- **Sample Rate:** 24kHz output
- **Hugging Face Repo:** `SWivid/F5-TTS`, `SWivid/E2-TTS`
- **Key Features:**
  - Zero-shot voice cloning
  - Romanian language adapter support (`adapters/romanian/`)
  - NFE (Number of Function Evaluations) configurable (default: 32 steps)
  - Sway sampling coefficient: -1.0
  - CFG strength: 2.0

### GPT-SoVITS
- **Integration:** `core/backend_gptsovits.py` - `GPTSoVITSBackend` class
- **Source:** RVC-Boss/GPT-SoVITS (git submodule)
- **Model Type:** GPT + SoVITS architecture
- **Models:**
  - `pretrained_v2` - Latest pretrained (gsv-v2final-pretrained)
  - `pretrained_v1` - Legacy pretrained
- **Sample Rate:** 32kHz output
- **Hugging Face Repo:** `lj1995/GPT-SoVITS`
- **Key Features:**
  - Zero-shot cloning with 5-second reference
  - Fine-tuning support
  - Multi-language (zh, en, ja, ko)

### Fish Speech
- **Integration:** `core/backend_fish.py` - `FishSpeechBackend` class
- **Source:** fishaudio/fish-speech (git submodule)
- **Model Type:** Autoregressive LLaMA-based with Firefly vocoder
- **Models:**
  - `fish-speech-1.5` - 1.5B parameter model
  - `fish-speech-1.4` - 1.4B parameter model
- **Sample Rate:** 44.1kHz output
- **Hugging Face Repo:** `fishaudio/fish-speech-1.5`
- **Key Features:**
  - vLLM acceleration support (`FishSpeechVLLMBackend`)
  - INT8/INT4 quantization support
  - Multi-language support

## Hardware Acceleration

### CUDA (NVIDIA)
- **Integration:** Native PyTorch CUDA in all backends
- **Configuration:** `--device cuda` or `--device auto`
- **CUDA Version:** 12.1 (torch 2.5.1+cu121)
- **Files:** `core/memory_manager.py`, `core/backend_f5.py`
- **Features:**
  - FP16 mixed precision
  - Flash Attention 2 support
  - CUDA graph optimization

### TensorRT-LLM
- **Integration:** `core/backend_tensorrt.py` - `TensorRTBackend` class
- **Speed Multiplier:** 7x (up to 10x with FP8)
- **Min VRAM:** 2GB
- **Quantization:** FP16, INT8, FP8 (Ada+)
- **Cache Location:** `~/.cache/trt_engines/`
- **Status:** Engine compilation framework ready; model-specific implementations pending
- **Requirements:**
  - NVIDIA GPU compute capability >= 7.0
  - CUDA 11.8 or 12.x
  - cuDNN 8.x

### DirectML (Windows)
- **Integration:** `core/backend_directml.py` - `DirectMLBackend` class
- **Speed Multiplier:** 1.5x
- **Platform:** Windows 10/11 only
- **GPU Support:** Any DirectX 12 GPU (AMD, Intel, NVIDIA)
- **Min VRAM:** 2GB
- **Installation:** `pip install onnxruntime-directml`
- **Status:** Framework ready; requires ONNX model export

### OpenVINO (Intel)
- **Integration:** `core/backend_openvino.py` - `OpenVINOBackend` class
- **Speed Multiplier:** 2.5x
- **Devices:** CPU, GPU (Arc/Iris), NPU (Meteor Lake+)
- **Min VRAM:** 0GB (CPU capable)
- **Version:** 2025.4.1
- **Installation:** `pip install openvino`
- **Status:** Framework ready; requires OpenVINO IR model export

### ONNX Runtime
- **Integration:** `core/backend_onnx.py` - `ONNXBackend` class
- **Speed Multiplier:** 2x (CPU vs naive PyTorch)
- **Execution Providers:**
  - CUDAExecutionProvider (NVIDIA)
  - ROCMExecutionProvider (AMD)
  - DmlExecutionProvider (Windows DirectML)
  - CPUExecutionProvider (fallback)
- **Version:** 1.23.2 (onnxruntime, onnxruntime-gpu, onnxruntime-directml)
- **Status:** Framework ready; requires ONNX model export

### vLLM
- **Integration:** `core/backend_vllm.py` - `VLLMBackend` class
- **Speed Multiplier:** 3-4x
- **Min VRAM:** 3GB
- **Quantization:** FP16, INT8, INT4
- **Platform:** Linux preferred (limited Windows support)
- **Use Case:** High-throughput batch inference

## Model Formats

### PyTorch (.pt, .pth, .safetensors)
- **Primary Format:** Native PyTorch checkpoints
- **F5-TTS:** `.safetensors` format (e.g., `F5TTS_v1_Base/model_1250000.safetensors`)
- **GPT-SoVITS:** `.ckpt` and `.pth` formats
- **Location:** Hugging Face Hub, local `~/.cache/`

### ONNX (.onnx)
- **Usage:** Cross-platform deployment
- **Backends:** ONNX Runtime, DirectML, OpenVINO
- **Quantization:** INT8 static/dynamic via onnxruntime
- **Export:** Model-specific conversion required

### GGUF
- **Usage:** CPU inference via llama.cpp
- **Quantization Levels:** Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q8_0
- **Integration:** `core/quantization.py` - `QuantizationManager.convert_to_gguf()`
- **Backend:** `core/backend_llamacpp.py` - `LlamaCppBackend`
- **Min VRAM:** 1.5GB (Q4_K_M)

### OpenVINO IR (.xml, .bin)
- **Usage:** Intel hardware optimization
- **Components:** Model XML + binary weights
- **Backend:** OpenVINOBackend (`core/backend_openvino.py`)

## Voice Cloning Pipeline

### Reference Audio Processing
- **Input Formats:** WAV, MP3, FLAC, OGG, M4A
- **Processing:** `core/voice_fusion.py` - `VoiceFusion` class
- **Embedding Extraction:**
  - F5-TTS: Mel spectrogram mean pooling
  - GPT-SoVITS: BERT encoder
  - Fish Speech: VQGAN encoder
- **Fusion Strategies:**
  - Simple average
  - Quality-weighted (default)
  - Duration-weighted
  - Combined weighted (quality × duration)

### Auto-Transcription
- **Backend:** Faster-Whisper
- **Version:** 1.2.1
- **Models:** tiny, base, small, medium, large-v3, large-v3-turbo
- **Compute Types:** int8, float16, float32
- **Usage:** Automatic reference transcript generation

### Voice Profiles Database
- **Storage:** JSON/SQLite (inferred from `core/profile_database.py`)
- **Profile Structure:**
  ```python
  {
      "name": "ProfileName",
      "reference_samples": [
          {"path": "audio.wav", "transcript": "..."}
      ]
  }
  ```
- **Quality Scoring:** `QualityScorer` class in `core/profile_database.py`

## API/Cloud Integrations

### Hugging Face Hub
- **Integration:** `huggingface-hub>=0.36.0`
- **Usage:** Model downloads via `hf_hub_download()`
- **Repos Used:**
  - `SWivid/F5-TTS`
  - `SWivid/E2-TTS`
  - `lj1995/GPT-SoVITS`
  - `fishaudio/fish-speech-1.5`
- **Authentication:** Token-based (HF_TOKEN env var if private)

### Cloud Storage (Optional)
- **AWS S3:** `boto3>=1.42.30` in requirements
- **Google Cloud Storage:** `google-cloud-storage>=3.8.0`
- **Aliyun OSS:** `oss2>=2.19.1` (modelscope integration)

## Monitoring & Observability

### GPU Monitoring
- **Integration:** `ui/monitor_widget.py` - `GPUMonitorWidget`
- **Library:** PyNVML (`pynvml>=13.0.1`)
- **Metrics Tracked:**
  - VRAM allocation/reservation
  - GPU utilization percentage
  - Temperature
  - Power draw
- **Update Rate:** Real-time polling

### Logging
- **Framework:** Built-in `logging` module
- **Level:** INFO (configurable)
- **Format:** `%(asctime)s | %(levelname)s | %(name)s | %(message)s`
- **Additional:** `loguru>=0.7.3` available in dependencies

### Experiment Tracking
- **Weights & Biases:** `wandb>=0.24.0`
- **TensorBoard:** `tensorboard>=2.20.0`
- **Usage:** Training runs (F5-TTS fine-tuning)

## Environment Configuration

**Required Environment Variables:**
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `HF_TOKEN` - Hugging Face authentication (for gated models)
- `TRANSFORMERS_CACHE` - Model cache location override

**Optional Environment Variables:**
- `VOICESYNTH_CACHE` - Custom cache directory
- `VOICESYNTH_VRAM_LIMIT` - Override VRAM budget (default: 5.5GB)

**Secrets Location:**
- Hugging Face token: `~/.cache/huggingface/token`
- No local credential files in project

## Model Cache Structure

```
~/.cache/
├── huggingface/           # HF Hub downloads
│   └── hub/
│       ├── models--SWivid--F5-TTS/
│       ├── models--lj1995--GPT-SoVITS/
│       └── models--fishaudio--fish-speech-1.5/
├── voicesynth/            # VoiceSynth cache
│   ├── quantized/         # Quantized model variants
│   └── profiles/          # Voice profile database
└── trt_engines/           # TensorRT compiled engines
    └── {model_name}_trt.engine
```

## Webhooks & Callbacks

**Incoming:**
- None (desktop application, no server endpoints)

**Outgoing:**
- W&B experiment logging (if enabled)
- Hugging Face model downloads

---

*Integration audit: 2026-03-31*
