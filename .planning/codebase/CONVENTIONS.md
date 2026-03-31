# Coding Conventions

**Analysis Date:** 2026-03-31

## Overview

VoiceSynthesisLocal is a Python-based voice synthesis application with multiple TTS backends (F5-TTS, GPT-SoVITS, Fish Speech). The codebase follows PEP 8 conventions with comprehensive type hints and modular architecture.

## Naming Patterns

**Files:**
- `snake_case.py` - All Python files use lowercase with underscores
- Examples: `inference_manager.py`, `backend_f5.py`, `gpu_monitor.py`, `voice_fusion.py`

**Classes:**
- `PascalCase` - All class names use PascalCase
- Examples: `InferenceManager`, `TTSBackend`, `GPUMonitor`, `VoiceFusion`, `BatchQueue`

**Functions and Methods:**
- `snake_case` - All functions and methods use snake_case
- Examples: `synthesize()`, `load_model()`, `get_memory_usage()`, `extract_embedding()`

**Variables:**
- `snake_case` for local variables: `audio_data`, `sample_rate`, `ref_text`
- Leading underscore for protected/internal: `_loaded`, `_backend_instance`, `_current_voice_path`
- Double underscore not observed (no name mangling needed)

**Constants:**
- `UPPER_SNAKE_CASE` for module-level constants
- Examples: `DARK_STYLESHEET`, `BUILTIN_PRESETS`, `F5_TTS_PATH`

**Type Variables and Enums:**
- `PascalCase` for enum classes: `BackendType`, `JobStatus`, `PresetType`, `ChunkingStrategy`
- Enum members use `UPPER_SNAKE_CASE`: `FP16`, `INT8`, `PENDING`, `COMPLETED`

## Code Style

**Formatting:**
- Line length: ~100-120 characters (observed in docstrings and comments)
- Indentation: 4 spaces (standard Python)
- Blank lines: 2 between classes/functions, 1 between methods

**Type Hints:**
- Comprehensive type hints using `typing` module
- Common imports:
  ```python
  from typing import Optional, Dict, Any, Union, Callable, Tuple, List
  from dataclasses import dataclass, field
  from enum import Enum
  from pathlib import Path
  ```

- Function signatures with full annotations:
  ```python
  def synthesize(
      self,
      text: str,
      voice: Optional[str] = None,
      nfe_steps: int = 16,
      sampler: str = "euler",
      speed: float = 1.0,
      seed: int = -1,
      language: str = "en",
      progress_callback: Optional[Callable[[int, str], None]] = None,
      **kwargs
  ) -> Tuple[np.ndarray, int]:
  ```

- Return type annotations always present:
  - `-> Tuple[np.ndarray, int]` for audio synthesis
  - `-> Dict[str, float]` for memory stats
  - `-> Optional[BackendType]` for optional returns
  - `-> None` for void methods

## Documentation Patterns

**Module Docstrings:**
- Three-line format with title and description
- Examples:
  ```python
  """
  Inference Manager
  ===================
  Central manager for TTS inference across multiple backends.
  Handles model loading, quantization, and generation.
  """
  ```

**Class Docstrings:**
- Brief description followed by feature list
- Usage examples where applicable:
  ```python
  class GPUMonitor:
      """
      Real-time GPU monitoring with callback support.

      Usage:
          monitor = GPUMonitor()
          monitor.start(callback=lambda stats: print(stats))
          ...
          monitor.stop()
      """
  ```

**Function/Method Docstrings:**
- Google-style docstrings with Args and Returns sections
- Args listed with type and description:
  ```python
  def extract_embedding(
      self,
      audio_path: Path,
      use_cache: bool = True
  ) -> VoiceEmbedding:
      """
      Extract voice embedding from an audio file.

      Args:
          audio_path: Path to audio file
          use_cache: Use cached embedding if available

      Returns:
          VoiceEmbedding object
      """
  ```

**Inline Comments:**
- Used to explain complex logic or workarounds
- Examples:
  ```python
  # INT8 dynamic quantization doesn't work on CUDA
  # Use FP16 for memory efficiency instead
  logger.info("Using FP16 precision for CUDA (INT8 not supported on CUDA)")
  ```

## Error Handling Approaches

**Exception Handling:**
- `try/except` with specific exception types
- Logging before re-raising or returning defaults:
  ```python
  try:
      import torch
      if torch.cuda.is_available():
          return "cuda"
  except ImportError:
      pass
  return "cpu"
  ```

**Error Logging:**
- Module-level logger in each file:
  ```python
  logger = logging.getLogger(__name__)
  ```
- Error messages include context:
  ```python
  logger.error(f"F5-TTS import error: {e}")
  logger.warning(f"Profile not found: {voice_reference}")
  ```

**Custom Exceptions:**
- Built-in exceptions used (`MemoryError`, `ValueError`, `ImportError`)
- No custom exception classes observed

**Graceful Degradation:**
- Optional dependencies handled with try/except:
  ```python
  try:
      import spacy
      nlp = spacy.load("en_core_web_sm")
  except ImportError:
      logger.warning("spacy not available, falling back to sentence chunking")
      return self._chunk_by_sentence(text)
  ```

**Return Values on Error:**
- Return placeholder/default values instead of crashing:
  ```python
  return np.zeros(24000 * 3, dtype=np.float32), 24000  # Placeholder audio
  return {"allocated_gb": 0, "reserved_gb": 0}  # Default memory stats
  ```

## Configuration Management

**Environment Configuration:**
- No `.env` file pattern observed
- Configuration passed via function arguments and dataclasses

**Application Settings:**
- Dataclasses for structured configuration:
  ```python
  @dataclass
  class OptimizationPreset:
      """Complete optimization settings preset."""
      name: str
      description: str
      target_vram_gb: float
      backend: str
      quantization: str
      # ...
  ```

**Path Management:**
- `pathlib.Path` used throughout for file paths:
  ```python
  from pathlib import Path
  PROJECT_ROOT = Path(__file__).parent.resolve()
  F5_TTS_PATH = Path(__file__).parent.parent / "F5-TTS" / "src"
  ```

**Logging Configuration:**
- Centralized in `main.py`:
  ```python
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
      datefmt='%H:%M:%S'
  )
  logger = logging.getLogger("VoiceSynth")
  ```

## Import Organization

**Import Order:**
1. Standard library imports (`sys`, `logging`, `time`)
2. Third-party imports (`numpy`, `PyQt6`, `torch`)
3. Local application imports (`core.*`, `ui.*`, `utils.*`)

**Examples:**
```python
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from core.backend_base import TTSBackend
from ui.styles import DARK_STYLESHEET
```

**Path Aliases:**
- No path aliases observed
- Relative imports use full module paths: `core.backend_f5`, `ui.main_window`

## Function Design

**Size Guidelines:**
- Methods typically 20-80 lines
- Complex methods broken into private helpers:
  ```python
  def load_model(self): ...
  def _load_f5_tts(self, progress_callback): ...
  def _load_gpt_sovits(self, progress_callback): ...
  ```

**Parameters:**
- Related parameters grouped in dataclasses where complex
- Optional parameters have sensible defaults
- `**kwargs` used for extensibility in synthesis methods

**Return Values:**
- Single values for simple operations
- `Tuple` for multiple return values: `(audio_data, sample_rate)`
- `Dict` for structured results: `{"allocated_gb": 1.2, "reserved_gb": 2.1}`
- `Optional[T]` when return may be `None`

## Module Design

**Exports:**
- Each module has clear public API via classes and functions
- Helper functions prefixed with underscore for internal use

**Singleton Pattern:**
- Global instance pattern for managers:
  ```python
  _memory_manager: Optional[AdaptiveMemoryManager] = None

  def get_memory_manager() -> AdaptiveMemoryManager:
      """Get the global memory manager instance."""
      global _memory_manager
      if _memory_manager is None:
          _memory_manager = AdaptiveMemoryManager()
      return _memory_manager
  ```

**Abstract Base Classes:**
- ABC used for backend interface:
  ```python
  from abc import ABC, abstractmethod

  class TTSBackend(ABC):
      @abstractmethod
      def load(self, progress_callback): ...

      @abstractmethod
      def synthesize(self, text, **kwargs) -> Tuple[np.ndarray, int]: ...
  ```

## PyQt6 Conventions

**Widget Structure:**
- Private `_setup_ui()` method for UI initialization
- Signal/slot connections in `_connect_signals()`
- Event handlers prefixed with `_on_`: `_on_voice_selected()`, `_on_preset_applied()`

**Qt Imports:**
```python
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QAction, QFont
```

**Custom Signals:**
```python
progress = pyqtSignal(int, str)  # percent, message
finished = pyqtSignal(object, int)  # audio data, sample_rate
error = pyqtSignal(str)
```

---

*Convention analysis: 2026-03-31*
