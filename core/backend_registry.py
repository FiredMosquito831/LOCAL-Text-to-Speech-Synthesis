"""
Backend Registry & Auto-Selector
==================================
Central registry for all backends with automatic selection based on hardware.
"""

import logging
from typing import Dict, Type, Optional, List, Callable
from dataclasses import dataclass
import sys

from core.backend_base import TTSBackend, BackendType, BackendCapabilities

logger = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    """Information about a registered backend."""
    backend_type: BackendType
    backend_class: Type[TTSBackend]
    available: bool
    install_instructions: str


class BackendRegistry:
    """
    Central registry for TTS backends.
    
    Handles:
    - Backend registration and discovery
    - Availability checking
    - Auto-selection based on hardware
    """
    
    _backends: Dict[BackendType, Type[TTSBackend]] = {}
    
    @classmethod
    def register(cls, backend_type: BackendType, backend_class: Type[TTSBackend]):
        """Register a backend."""
        cls._backends[backend_type] = backend_class
        logger.debug(f"Registered backend: {backend_type.value}")
    
    @classmethod
    def get(cls, backend_type: BackendType) -> Type[TTSBackend]:
        """Get a backend class by type."""
        if backend_type not in cls._backends:
            raise ValueError(f"Unknown backend: {backend_type}")
        return cls._backends[backend_type]
    
    @classmethod
    def list_all(cls) -> Dict[BackendType, BackendInfo]:
        """List all registered backends with availability info."""
        result = {}
        for backend_type, backend_class in cls._backends.items():
            result[backend_type] = BackendInfo(
                backend_type=backend_type,
                backend_class=backend_class,
                available=backend_class.is_available(),
                install_instructions=backend_class.get_install_instructions()
            )
        return result
    
    @classmethod
    def list_available(cls) -> List[BackendType]:
        """List only available backends."""
        return [
            bt for bt, bc in cls._backends.items()
            if bc.is_available()
        ]
    
    @classmethod
    def get_capabilities(cls, backend_type: BackendType) -> BackendCapabilities:
        """Get capabilities for a backend."""
        return cls._backends[backend_type].CAPABILITIES


def _register_all_backends():
    """Register all built-in backends."""
    # Import all backends
    from core.backend_f5 import F5TTSBackend
    from core.backend_gptsovits import GPTSoVITSBackend
    from core.backend_fish import FishSpeechBackend
    from core.backend_vllm import VLLMBackend
    from core.backend_tensorrt import TensorRTBackend
    from core.backend_onnx import ONNXBackend
    from core.backend_llamacpp import LlamaCppBackend
    from core.backend_openvino import OpenVINOBackend
    from core.backend_directml import DirectMLBackend
    
    # Register each
    BackendRegistry.register(BackendType.PYTORCH_NATIVE, F5TTSBackend)
    BackendRegistry.register(BackendType.GPT_SOVITS, GPTSoVITSBackend)
    BackendRegistry.register(BackendType.FISH_SPEECH, FishSpeechBackend)
    BackendRegistry.register(BackendType.VLLM, VLLMBackend)
    BackendRegistry.register(BackendType.TENSORRT, TensorRTBackend)
    BackendRegistry.register(BackendType.ONNX, ONNXBackend)
    BackendRegistry.register(BackendType.LLAMA_CPP, LlamaCppBackend)
    BackendRegistry.register(BackendType.OPENVINO, OpenVINOBackend)
    BackendRegistry.register(BackendType.DIRECTML, DirectMLBackend)


# Register on import
try:
    _register_all_backends()
except ImportError as e:
    logger.warning(f"Some backends could not be registered: {e}")


class BackendAutoSelector:
    """
    Automatically selects the best backend based on:
    - Available VRAM
    - Installed dependencies
    - Quality/speed preferences
    """
    
    def __init__(self):
        self._vram_gb = self._get_available_vram()
    
    def _get_available_vram(self) -> float:
        """Get available VRAM in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total = props.total_memory / 1024**3
                allocated = torch.cuda.memory_allocated() / 1024**3
                return total - allocated
        except:
            pass
        return 0.0
    
    def select_backend(
        self,
        prefer_quality: bool = True,
        min_speed_multiplier: float = 1.0,
        force_available_only: bool = True
    ) -> BackendType:
        """
        Select the best backend based on current conditions.
        
        Args:
            prefer_quality: If True, prioritize quality over speed
            min_speed_multiplier: Minimum acceptable speed improvement
            force_available_only: Only consider installed backends
            
        Returns:
            Best matching BackendType
        """
        candidates = []
        
        for backend_type in BackendRegistry.list_available() if force_available_only else BackendType:
            try:
                caps = BackendRegistry.get_capabilities(backend_type)
                
                # Check VRAM requirement
                if caps.min_vram_gb > self._vram_gb + 0.5:  # 0.5GB buffer
                    continue
                
                # Check speed requirement
                if caps.speed_multiplier < min_speed_multiplier:
                    continue
                
                # Check OS compatibility
                if sys.platform == "win32" and not caps.windows_supported:
                    continue
                if sys.platform != "win32" and not caps.linux_supported:
                    continue
                
                # Calculate score
                if prefer_quality:
                    score = caps.quality_rating * 2 + caps.speed_multiplier
                else:
                    score = caps.speed_multiplier * 2 + caps.quality_rating
                
                candidates.append((backend_type, score))
                
            except Exception as e:
                logger.debug(f"Skipping {backend_type}: {e}")
        
        if not candidates:
            # Fallback to PyTorch native
            logger.warning("No optimal backend found, using PyTorch native")
            return BackendType.PYTORCH_NATIVE
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected = candidates[0][0]
        logger.info(f"Auto-selected backend: {selected.value}")
        return selected
    
    def get_recommendation(self) -> Dict:
        """
        Get a detailed recommendation with reasoning.
        
        Returns:
            Dict with recommended backend and explanation
        """
        vram = self._vram_gb
        available = BackendRegistry.list_available()
        
        # Determine recommendation based on VRAM
        if vram >= 5:
            if BackendType.TENSORRT in available:
                recommended = BackendType.TENSORRT
                reason = "TensorRT available with sufficient VRAM (5+ GB)"
            elif BackendType.VLLM in available:
                recommended = BackendType.VLLM
                reason = "vLLM available for optimized inference"
            else:
                recommended = BackendType.PYTORCH_NATIVE
                reason = "Using PyTorch with FP16 (sufficient VRAM)"
        elif vram >= 3:
            if BackendType.VLLM in available:
                recommended = BackendType.VLLM
                reason = "vLLM with INT8 for medium VRAM (3-5 GB)"
            else:
                recommended = BackendType.PYTORCH_NATIVE
                reason = "PyTorch with INT8 quantization"
        elif vram >= 1:
            if BackendType.LLAMA_CPP in available:
                recommended = BackendType.LLAMA_CPP
                reason = "GGUF Q4 for low VRAM (1-3 GB)"
            elif BackendType.ONNX in available:
                recommended = BackendType.ONNX
                reason = "ONNX with CPU offload"
            else:
                recommended = BackendType.PYTORCH_NATIVE
                reason = "PyTorch with heavy quantization"
        else:
            if BackendType.ONNX in available:
                recommended = BackendType.ONNX
                reason = "ONNX CPU mode (no VRAM available)"
            else:
                recommended = BackendType.PYTORCH_NATIVE
                reason = "PyTorch CPU fallback"
        
        return {
            "recommended": recommended,
            "reason": reason,
            "available_vram_gb": vram,
            "available_backends": [b.value for b in available],
            "all_recommendations": self._get_all_recommendations()
        }
    
    def _get_all_recommendations(self) -> List[Dict]:
        """Get recommendations for all VRAM tiers."""
        return [
            {"vram_tier": "5+ GB", "backend": "TensorRT", "quality": "★★★★★", "speed": "5-10x"},
            {"vram_tier": "3-5 GB", "backend": "vLLM + INT8", "quality": "★★★★★", "speed": "3-4x"},
            {"vram_tier": "1-3 GB", "backend": "GGUF Q4", "quality": "★★★☆☆", "speed": "Variable"},
            {"vram_tier": "<1 GB", "backend": "ONNX CPU", "quality": "★★★★☆", "speed": "2x"},
        ]


# Convenience functions
def get_best_backend(prefer_quality: bool = True) -> BackendType:
    """Get the best available backend."""
    selector = BackendAutoSelector()
    return selector.select_backend(prefer_quality=prefer_quality)


def get_backend_info() -> Dict:
    """Get detailed backend information and recommendation."""
    selector = BackendAutoSelector()
    return selector.get_recommendation()
