"""
ONNX Runtime Backend
=====================
Cross-platform inference using ONNX Runtime.
Best for: CPU fallback, AMD GPUs, portable deployment
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Callable
import numpy as np

from core.backend_base import TTSBackend, BackendCapabilities

logger = logging.getLogger(__name__)


class ONNXBackend(TTSBackend):
    """
    ONNX Runtime backend for cross-platform inference.
    
    Provides:
    - CPU fallback when no GPU available
    - AMD GPU support via ROCm
    - ~2x faster than naive PyTorch on CPU
    """
    
    CAPABILITIES = BackendCapabilities(
        name="ONNX Runtime",
        min_vram_gb=0,  # Can run on CPU
        supports_quantization=["fp32", "fp16", "int8"],
        speed_multiplier=2.0,
        quality_rating=4,
        requires_compilation=True,  # Needs ONNX export
        windows_supported=True,
        linux_supported=True
    )
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        onnx_path: Optional[str] = None,
        use_gpu: bool = True,
        num_threads: int = 4,
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        self._session = None
    
    def _get_execution_providers(self):
        """Get available execution providers."""
        providers = []
        
        if self.use_gpu:
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                elif "ROCMExecutionProvider" in available:
                    providers.append("ROCMExecutionProvider")
                elif "DmlExecutionProvider" in available:
                    providers.append("DmlExecutionProvider")
            except:
                pass
        
        providers.append("CPUExecutionProvider")
        return providers
    
    def load(self, progress_callback: Optional[Callable[[int, str], None]] = None):
        """Load ONNX model."""
        if self._loaded:
            return
        
        try:
            import onnxruntime as ort
            
            if progress_callback:
                progress_callback(10, "Initializing ONNX Runtime...")
            
            if self.onnx_path is None or not self.onnx_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {self.onnx_path}. "
                    "Please export the model to ONNX format first."
                )
            
            if progress_callback:
                progress_callback(30, "Loading ONNX model...")
            
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.num_threads
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            providers = self._get_execution_providers()
            
            self._session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_options,
                providers=providers
            )
            
            self._loaded = True
            
            if progress_callback:
                progress_callback(100, f"ONNX model ready! Using: {self._session.get_providers()[0]}")
            
            logger.info(f"ONNX backend loaded: {self.onnx_path}")
            
        except ImportError:
            raise ImportError(
                "ONNX Runtime not installed. Install with:\n"
                "pip install onnxruntime-gpu  # For GPU\n"
                "pip install onnxruntime      # For CPU only"
            )
    
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize using ONNX Runtime."""
        if not self._loaded:
            self.load()
        
        raise NotImplementedError(
            "ONNX synthesis requires model-specific input preparation. "
            "Pre-exported ONNX models for F5-TTS coming soon."
        )
    
    def unload(self):
        """Unload ONNX session."""
        if self._session is not None:
            del self._session
            self._session = None
        
        self._loaded = False
        logger.info("ONNX backend unloaded")
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if ONNX Runtime is available."""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        return (
            "# For NVIDIA GPU:\n"
            "pip install onnxruntime-gpu\n\n"
            "# For AMD GPU (ROCm):\n"
            "pip install onnxruntime-rocm\n\n"
            "# For CPU only:\n"
            "pip install onnxruntime\n\n"
            "# For DirectML (Windows, any GPU):\n"
            "pip install onnxruntime-directml"
        )
