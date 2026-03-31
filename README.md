# Local Text to Speech + Synthesis

A unified local text-to-speech synthesis system supporting multiple TTS backends with voice cloning, batch processing, and GPU optimization.

## Features

- **Multiple TTS Backends**: F5-TTS, GPT-SoVITS, Fish Speech
- **Voice Cloning**: Clone voices from reference audio samples
- **Voice Fusion**: Blend multiple voices together
- **Batch Processing**: Queue and process multiple synthesis tasks
- **GPU Optimization**: Support for CUDA, DirectML, OpenVINO, TensorRT
- **Quantization**: FP16, INT8, INT4, GGUF Q4/Q8
- **Memory Management**: Optimized for 6GB VRAM GPUs
- **GUI & CLI**: PyQt6 GUI with CLI fallback

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for GPU acceleration)
- See `requirements_ultimate.txt` for full dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/FiredMosquito831/LOCAL-Text-to-Speech-Synthesis.git
cd LOCAL-Text-to-Speech-Synthesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements_ultimate.txt
```

## Usage

### GUI Mode

```bash
python main.py
```

### CLI Mode

```bash
# Generate speech from text
python main.py --cli --text "Hello world" --output hello.wav

# Generate from file
python main.py --cli --file input.txt --output output.wav

# Use specific voice and backend
python main.py --cli --text "Hello" --voice reference.wav --backend f5-tts

# With quantization
python main.py --cli --text "Hello" --quantization int4
```

### Backend Selection

| Backend | Command Flag | Description |
|---------|-------------|-------------|
| F5-TTS | `--backend f5-tts` | Fast, high-quality TTS |
| GPT-SoVITS | `--backend gpt-sovits` | Voice cloning focused |
| Fish Speech | `--backend fish-speech` | Multi-lingual support |

### Quantization Options

| Level | Command Flag | VRAM Usage |
|-------|-------------|------------|
| FP16 | `--quantization fp16` | ~6GB |
| INT8 | `--quantization int8` | ~3GB |
| INT4 | `--quantization int4` | ~1.5GB |
| GGUF Q4 | `--quantization gguf-q4` | ~1GB |
| GGUF Q8 | `--quantization gguf-q8` | ~2GB |

## Project Structure

```
VoiceSynthesisLocal/
├── core/                    # Core TTS backend implementations
│   ├── backend_base.py      # Base backend interface
│   ├── backend_f5.py        # F5-TTS backend
│   ├── backend_gptsovits.py # GPT-SoVITS backend
│   ├── backend_fish.py      # Fish Speech backend
│   ├── inference_manager.py # Inference orchestration
│   ├── memory_manager.py    # GPU memory management
│   └── voice_fusion.py      # Voice blending
├── ui/                      # PyQt6 GUI
│   ├── main_window.py       # Main window
│   ├── tts_panel.py         # TTS controls
│   └── cloning_panel.py     # Voice cloning UI
├── utils/                   # Utilities
├── training_workspace/      # Training utilities
├── scripts/                # Helper scripts
└── main.py                 # Entry point
```

## Training

The project includes a training workspace for fine-tuning models:

```bash
cd training_workspace
python setup_training.py
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [F5-TTS](https://github.com/SynpaTech/F5-TTS) - Fast and faithful TTS
- [GPT-SoVITS](https://github.com/jianping2022/GPT-SoVITS) - Voice cloning
- [Fish-Speech](https://github.com/fishaudio/fish-speech) - Multi-lingual TTS