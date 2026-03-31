"""
Backend Base Class
===================
Abstract base class for all TTS backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, Dict, Any
from enum import Enum
import numpy as np


class BackendType(Enum):
    """Supported backend types."""
    PYTORCH_NATIVE = "pytorch"
    GPT_SOVITS = "gpt-sovits"
    FISH_SPEECH = "fish-speech"
    VLLM = "vllm"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    LLAMA_CPP = "llama-cpp"
    OPENVINO = "openvino"
    DIRECTML = "directml"


class BackendCapabilities:
    """Capabilities and requirements of a backend."""
    
    def __init__(
        self,
        name: str,
        min_vram_gb: float,
        supports_quantization: list,
        speed_multiplier: float,
        quality_rating: int,
        requires_compilation: bool = False,
        windows_supported: bool = True,
        linux_supported: bool = True
    ):
        self.name = name
        self.min_vram_gb = min_vram_gb
        self.supports_quantization = supports_quantization
        self.speed_multiplier = speed_multiplier
        self.quality_rating = quality_rating  # 1-5
        self.requires_compilation = requires_compilation
        self.windows_supported = windows_supported
        self.linux_supported = linux_supported


class TTSBackend(ABC):
    """
    Abstract base class for TTS backends.
    
    All backends must implement:
    - load(): Load model
    - synthesize(): Generate audio
    - unload(): Free memory
    """
    
    # Backend capabilities - override in subclass
    CAPABILITIES = BackendCapabilities(
        name="Base",
        min_vram_gb=0,
        supports_quantization=[],
        speed_multiplier=1.0,
        quality_rating=3
    )
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        self.model_name = model_name
        self.device = device
        self._loaded = False
        self._config = kwargs
    
    @abstractmethod
    def load(self, progress_callback: Optional[Callable[[int, str], None]] = None):
        """Load the model."""
        pass
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech. Returns (audio_data, sample_rate)."""
        pass
    
    @abstractmethod
    def unload(self):
        """Unload model and free memory."""
        pass
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "reserved_gb": torch.cuda.memory_reserved() / 1024**3
                }
        except:
            pass
        return {"allocated_gb": 0, "reserved_gb": 0}
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if this backend is available on the system."""
        return True  # Override in subclasses
    
    @classmethod
    def get_install_instructions(cls) -> str:
        """Return installation instructions for this backend."""
        return "No additional installation required."
