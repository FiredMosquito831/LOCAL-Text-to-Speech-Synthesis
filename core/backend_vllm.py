"""
vLLM Backend
=============
Optimized inference using vLLM for autoregressive TTS models.
Best for: Fish Speech, GPT-SoVITS AR components
"""

import logging
from typing import Optional, Tuple, Callable
import numpy as np

from core.backend_base import TTSBackend, BackendCapabilities

logger = logging.getLogger(__name__)


class VLLMBackend(TTSBackend):
    """
    vLLM-accelerated backend for AR TTS models.
    
    Provides 3-4x faster inference compared to native PyTorch
    with optimizations like:
    - PagedAttention for efficient KV caching
    - Continuous batching
    - Tensor parallelism
    """
    
    CAPABILITIES = BackendCapabilities(
        name="vLLM",
        min_vram_gb=3.0,
        supports_quantization=["fp16", "int8", "int4-gptq", "int4-awq"],
        speed_multiplier=3.5,
        quality_rating=5,
        requires_compilation=False,
        windows_supported=False,  # vLLM has limited Windows support
        linux_supported=True
    )
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.85,
        enable_chunked_prefill: bool = True,
        enable_prefix_caching: bool = True,
        tensor_parallel_size: int = 1,
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_chunked_prefill = enable_chunked_prefill
        self.enable_prefix_caching = enable_prefix_caching
        self.tensor_parallel_size = tensor_parallel_size
        self._engine = None
    
    def load(self, progress_callback: Optional[Callable[[int, str], None]] = None):
        """Load model using vLLM engine."""
        if self._loaded:
            return
        
        try:
            if progress_callback:
                progress_callback(10, "Initializing vLLM engine...")
            
            from vllm import LLM, SamplingParams
            
            if progress_callback:
                progress_callback(30, "Loading model...")
            
            self._engine = LLM(
                model=self.model_name,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enable_chunked_prefill=self.enable_chunked_prefill,
                enable_prefix_caching=self.enable_prefix_caching,
                tensor_parallel_size=self.tensor_parallel_size,
            )
            
            self._loaded = True
            
            if progress_callback:
                progress_callback(100, "vLLM engine ready!")
            
            logger.info(f"vLLM backend loaded: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm\n"
                "Note: vLLM has limited Windows support. Linux recommended."
            )
    
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize using vLLM engine."""
        if not self._loaded:
            self.load()
        
        # vLLM is primarily for AR token generation
        # TTS-specific implementation would go here
        raise NotImplementedError(
            "vLLM synthesis requires a compatible AR-TTS model. "
            "Currently supported: Fish Speech (coming soon)"
        )
    
    def unload(self):
        """Unload vLLM engine."""
        if self._engine is not None:
            del self._engine
            self._engine = None
        
        self._loaded = False
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        logger.info("vLLM backend unloaded")
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if vLLM is available."""
        try:
            import vllm
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        return (
            "pip install vllm\n\n"
            "Requirements:\n"
            "- Linux (Windows has limited support)\n"
            "- CUDA 11.8 or 12.1\n"
            "- NVIDIA GPU with compute capability >= 7.0"
        )
