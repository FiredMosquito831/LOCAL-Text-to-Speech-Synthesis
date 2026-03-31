"""
OpenVINO Backend
=================
Intel-optimized inference for Intel GPUs, NPUs, and CPUs.
Best for: Intel hardware acceleration
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Callable
import numpy as np

from core.backend_base import TTSBackend, BackendCapabilities

logger = logging.getLogger(__name__)


class OpenVINOBackend(TTSBackend):
    """
    OpenVINO backend for Intel hardware.
    
    Provides:
    - Intel GPU acceleration (Arc, Iris)
    - Intel NPU support (Meteor Lake+)
    - Optimized CPU inference
    """
    
    CAPABILITIES = BackendCapabilities(
        name="OpenVINO",
        min_vram_gb=0,  # Can run on CPU
        supports_quantization=["fp32", "fp16", "int8"],
        speed_multiplier=2.5,
        quality_rating=4,
        requires_compilation=True,
        windows_supported=True,
        linux_supported=True
    )
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",  # AUTO, CPU, GPU, NPU
        ir_path: Optional[str] = None,
        num_threads: int = 4,
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self.ir_path = Path(ir_path) if ir_path else None
        self.num_threads = num_threads
        self._compiled_model = None
    
    def load(self, progress_callback: Optional[Callable[[int, str], None]] = None):
        """Load OpenVINO model."""
        if self._loaded:
            return
        
        try:
            if progress_callback:
                progress_callback(10, "Initializing OpenVINO...")
            
            from openvino import Core
            
            core = Core()
            
            if progress_callback:
                progress_callback(30, f"Loading model to {self.device}...")
            
            if self.ir_path is None or not self.ir_path.exists():
                raise FileNotFoundError(
                    f"OpenVINO IR model not found at {self.ir_path}. "
                    "Please export the model to OpenVINO format first."
                )
            
            # Determine device
            device = self.device.upper() if self.device != "auto" else "AUTO"
            
            # Load and compile model
            model = core.read_model(str(self.ir_path))
            self._compiled_model = core.compile_model(model, device)
            
            self._loaded = True
            
            if progress_callback:
                progress_callback(100, f"OpenVINO model ready on {device}!")
            
            logger.info(f"OpenVINO backend loaded: {self.ir_path}")
            
        except ImportError:
            raise ImportError(
                "OpenVINO not installed. Install with:\n"
                "pip install openvino"
            )
    
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize using OpenVINO."""
        if not self._loaded:
            self.load()
        
        raise NotImplementedError(
            "OpenVINO synthesis requires model-specific implementation. "
            "OpenVINO-optimized TTS models coming soon."
        )
    
    def unload(self):
        """Unload OpenVINO model."""
        if self._compiled_model is not None:
            del self._compiled_model
            self._compiled_model = None
        
        self._loaded = False
        logger.info("OpenVINO backend unloaded")
    
    @classmethod
    def get_available_devices(cls):
        """List available OpenVINO devices."""
        try:
            from openvino import Core
            core = Core()
            return core.available_devices
        except:
            return []
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if OpenVINO is available."""
        try:
            import openvino
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        return (
            "pip install openvino\n\n"
            "For Intel GPU support, also install:\n"
            "pip install openvino-dev[extras]\n\n"
            "Compatible devices:\n"
            "- Intel CPUs (Core, Xeon)\n"
            "- Intel GPUs (Arc, Iris)\n"
            "- Intel NPUs (Meteor Lake+)"
        )
