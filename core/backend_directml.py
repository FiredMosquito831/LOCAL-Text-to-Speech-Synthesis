"""
DirectML Backend
=================
Windows GPU acceleration for AMD, Intel, and NVIDIA via DirectX 12.
Best for: Windows systems with non-NVIDIA GPUs
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Callable
import numpy as np

from core.backend_base import TTSBackend, BackendCapabilities

logger = logging.getLogger(__name__)


class DirectMLBackend(TTSBackend):
    """
    DirectML backend using ONNX Runtime DirectML EP.
    
    Provides:
    - GPU acceleration on Windows for any DirectX 12 GPU
    - Works with AMD, Intel, NVIDIA
    - No CUDA required
    """
    
    CAPABILITIES = BackendCapabilities(
        name="DirectML",
        min_vram_gb=2.0,
        supports_quantization=["fp32", "fp16"],
        speed_multiplier=1.5,
        quality_rating=4,
        requires_compilation=True,
        windows_supported=True,
        linux_supported=False  # Windows only
    )
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        onnx_path: Optional[str] = None,
        device_id: int = 0,
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.device_id = device_id
        self._session = None
    
    def load(self, progress_callback: Optional[Callable[[int, str], None]] = None):
        """Load ONNX model with DirectML execution provider."""
        if self._loaded:
            return
        
        try:
            if progress_callback:
                progress_callback(10, "Initializing DirectML...")
            
            import onnxruntime as ort
            
            available = ort.get_available_providers()
            if "DmlExecutionProvider" not in available:
                raise RuntimeError(
                    "DirectML execution provider not available. "
                    "Install with: pip install onnxruntime-directml"
                )
            
            if self.onnx_path is None or not self.onnx_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {self.onnx_path}. "
                    "Please export the model to ONNX format first."
                )
            
            if progress_callback:
                progress_callback(30, "Loading model with DirectML...")
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self._session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_options,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"]
            )
            
            self._loaded = True
            
            if progress_callback:
                progress_callback(100, "DirectML model ready!")
            
            logger.info(f"DirectML backend loaded: {self.onnx_path}")
            
        except ImportError:
            raise ImportError(
                "ONNX Runtime DirectML not installed. Install with:\n"
                "pip install onnxruntime-directml"
            )
    
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize using DirectML."""
        if not self._loaded:
            self.load()
        
        raise NotImplementedError(
            "DirectML synthesis requires model-specific implementation."
        )
    
    def unload(self):
        """Unload DirectML session."""
        if self._session is not None:
            del self._session
            self._session = None
        
        self._loaded = False
        logger.info("DirectML backend unloaded")
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if DirectML is available."""
        import sys
        if sys.platform != "win32":
            return False
        
        try:
            import onnxruntime as ort
            return "DmlExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        return (
            "pip install onnxruntime-directml\n\n"
            "Requirements:\n"
            "- Windows 10/11\n"
            "- DirectX 12 compatible GPU\n"
            "- Works with AMD, Intel, and NVIDIA GPUs"
        )
