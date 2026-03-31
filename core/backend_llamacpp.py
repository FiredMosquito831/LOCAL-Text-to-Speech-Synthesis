"""
llama.cpp (GGUF) Backend
=========================
Extreme low-VRAM inference using quantized GGUF models.
Best for: Systems with <4GB VRAM, CPU-primary inference
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Callable
import numpy as np

from core.backend_base import TTSBackend, BackendCapabilities

logger = logging.getLogger(__name__)


class LlamaCppBackend(TTSBackend):
    """
    llama.cpp backend for GGUF quantized models.
    
    Provides extreme memory efficiency with:
    - 2-8 bit quantization (Q2_K to Q8_0)
    - CPU + partial GPU offload
    - Very low VRAM footprint
    """
    
    CAPABILITIES = BackendCapabilities(
        name="llama.cpp (GGUF)",
        min_vram_gb=0.5,  # Can run almost entirely on CPU
        supports_quantization=["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q8_0"],
        speed_multiplier=0.7,  # Slower than GPU but works with less VRAM
        quality_rating=3,  # Some quality loss with heavy quantization
        requires_compilation=False,
        windows_supported=True,
        linux_supported=True
    )
    
    # GGUF quantization types with descriptions
    QUANT_TYPES = {
        "q2_k": {"bits": 2, "quality": 1, "size_mult": 0.15, "desc": "Smallest, lowest quality"},
        "q3_k_m": {"bits": 3, "quality": 2, "size_mult": 0.20, "desc": "Very small, low quality"},
        "q4_k_m": {"bits": 4, "quality": 3, "size_mult": 0.25, "desc": "Small, acceptable quality"},
        "q5_k_m": {"bits": 5, "quality": 4, "size_mult": 0.35, "desc": "Medium, good quality"},
        "q8_0": {"bits": 8, "quality": 5, "size_mult": 0.50, "desc": "Larger, best quality"},
    }
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        gguf_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = 32,
        n_threads: int = 8,
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self.gguf_path = Path(gguf_path) if gguf_path else None
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers  # 0 = CPU only
        self.n_threads = n_threads
        self._model = None
    
    def load(self, progress_callback: Optional[Callable[[int, str], None]] = None):
        """Load GGUF model using llama-cpp-python."""
        if self._loaded:
            return
        
        try:
            if progress_callback:
                progress_callback(10, "Loading llama.cpp...")
            
            from llama_cpp import Llama
            
            if self.gguf_path is None or not self.gguf_path.exists():
                raise FileNotFoundError(
                    f"GGUF model not found at {self.gguf_path}. "
                    "Please download or convert a GGUF model first."
                )
            
            if progress_callback:
                progress_callback(30, f"Loading GGUF model ({self.n_gpu_layers} GPU layers)...")
            
            self._model = Llama(
                model_path=str(self.gguf_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=False
            )
            
            self._loaded = True
            
            if progress_callback:
                progress_callback(100, "GGUF model ready!")
            
            logger.info(f"llama.cpp backend loaded: {self.gguf_path}")
            
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with:\n"
                "pip install llama-cpp-python\n\n"
                "For GPU acceleration:\n"
                "CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python"
            )
    
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize using GGUF model."""
        if not self._loaded:
            self.load()
        
        raise NotImplementedError(
            "GGUF synthesis requires TTS-specific GGUF models. "
            "Conversion scripts for F5-TTS coming soon."
        )
    
    def unload(self):
        """Unload GGUF model."""
        if self._model is not None:
            del self._model
            self._model = None
        
        self._loaded = False
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        logger.info("llama.cpp backend unloaded")
    
    @classmethod
    def convert_to_gguf(
        cls,
        model_path: str,
        output_path: str,
        quantization: str = "q4_k_m",
        progress_callback: Optional[Callable] = None
    ):
        """Convert a model to GGUF format."""
        raise NotImplementedError(
            "GGUF conversion requires llama.cpp's convert scripts. "
            "See: https://github.com/ggerganov/llama.cpp#prepare-and-quantize"
        )
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if llama-cpp-python is available."""
        try:
            import llama_cpp
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        return (
            "# CPU only:\n"
            "pip install llama-cpp-python\n\n"
            "# With CUDA GPU support:\n"
            "CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python --force-reinstall --no-cache-dir\n\n"
            "# On Windows with CUDA:\n"
            'set CMAKE_ARGS="-DLLAMA_CUBLAS=on"\n'
            "pip install llama-cpp-python --force-reinstall --no-cache-dir"
        )
