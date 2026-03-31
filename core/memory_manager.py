"""
Adaptive Memory Manager
========================
Memory management system with LRU eviction and profiling.
Optimized for 6GB VRAM constraint.
"""

import logging
import time
import gc
from pathlib import Path
from typing import Dict, Optional, Callable, List, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_id: str
    model: Any
    size_bytes: int
    last_used: float = field(default_factory=time.time)
    load_time_ms: float = 0
    component: str = "unknown"  # e.g., "tts", "vocoder", "whisper"


@dataclass 
class MemoryBreakdown:
    """Breakdown of memory usage by component."""
    tts_model_mb: float = 0
    vocoder_mb: float = 0
    whisper_mb: float = 0
    kv_cache_mb: float = 0
    audio_buffer_mb: float = 0
    cuda_overhead_mb: float = 500  # Approximate
    total_mb: float = 0
    available_mb: float = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "TTS Model": self.tts_model_mb,
            "Vocoder": self.vocoder_mb,
            "Whisper": self.whisper_mb,
            "KV Cache": self.kv_cache_mb,
            "Audio Buffer": self.audio_buffer_mb,
            "CUDA Overhead": self.cuda_overhead_mb,
        }


class AdaptiveMemoryManager:
    """
    Adaptive memory manager for voice synthesis.
    
    Features:
    - LRU model eviction when VRAM is low
    - Memory budget tracking
    - Automatic offloading to CPU
    - Component-level memory profiling
    """
    
    def __init__(
        self,
        vram_budget_gb: float = 5.5,
        safety_margin_gb: float = 0.5,
        enable_auto_eviction: bool = True
    ):
        self.vram_budget = vram_budget_gb * 1024**3  # Convert to bytes
        self.safety_margin = safety_margin_gb * 1024**3
        self.enable_auto_eviction = enable_auto_eviction
        
        # LRU cache of loaded models
        self._loaded_models: OrderedDict[str, ModelInfo] = OrderedDict()
        self._lock = threading.Lock()
        
        # Memory tracking
        self._component_sizes: Dict[str, int] = {}
        
        # Callbacks
        self._on_eviction_callbacks: List[Callable[[str], None]] = []
        self._on_low_memory_callbacks: List[Callable[[float], None]] = []
        
        logger.info(f"MemoryManager initialized: budget={vram_budget_gb}GB, margin={safety_margin_gb}GB")
    
    def get_vram_stats(self) -> Dict[str, float]:
        """Get current VRAM statistics."""
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                free = total - allocated
                
                return {
                    "total_gb": total / 1024**3,
                    "allocated_gb": allocated / 1024**3,
                    "reserved_gb": reserved / 1024**3,
                    "free_gb": free / 1024**3,
                    "usage_percent": (allocated / total) * 100
                }
        except:
            pass
        
        return {
            "total_gb": 0,
            "allocated_gb": 0,
            "reserved_gb": 0,
            "free_gb": 0,
            "usage_percent": 0
        }
    
    def get_free_vram(self) -> int:
        """Get available VRAM in bytes."""
        stats = self.get_vram_stats()
        return int(stats["free_gb"] * 1024**3)
    
    def can_load(self, model_size_bytes: int) -> bool:
        """Check if a model can be loaded within budget."""
        free = self.get_free_vram()
        return free > model_size_bytes + self.safety_margin
    
    def register_model(
        self,
        model_id: str,
        model: Any,
        size_bytes: int,
        component: str = "unknown"
    ):
        """Register a loaded model with the manager."""
        with self._lock:
            self._loaded_models[model_id] = ModelInfo(
                model_id=model_id,
                model=model,
                size_bytes=size_bytes,
                last_used=time.time(),
                component=component
            )
            self._loaded_models.move_to_end(model_id)
            
            # Track component size
            self._component_sizes[component] = self._component_sizes.get(component, 0) + size_bytes
            
            logger.debug(f"Registered model: {model_id} ({size_bytes / 1024**2:.1f} MB)")
    
    def unregister_model(self, model_id: str):
        """Unregister a model."""
        with self._lock:
            if model_id in self._loaded_models:
                info = self._loaded_models[model_id]
                
                # Update component tracking
                if info.component in self._component_sizes:
                    self._component_sizes[info.component] -= info.size_bytes
                
                del self._loaded_models[model_id]
                logger.debug(f"Unregistered model: {model_id}")
    
    def touch(self, model_id: str):
        """Update last-used time for a model (prevents eviction)."""
        with self._lock:
            if model_id in self._loaded_models:
                self._loaded_models[model_id].last_used = time.time()
                self._loaded_models.move_to_end(model_id)
    
    def load_model(
        self,
        model_id: str,
        loader_fn: Callable,
        size_bytes: int,
        component: str = "unknown",
        force: bool = False
    ) -> Any:
        """
        Load a model, evicting others if necessary.
        
        Args:
            model_id: Unique identifier for the model
            loader_fn: Function that loads and returns the model
            size_bytes: Expected size of the model
            component: Component type (tts, vocoder, whisper)
            force: Force load even if over budget
            
        Returns:
            The loaded model
        """
        # Check if already loaded
        with self._lock:
            if model_id in self._loaded_models:
                self.touch(model_id)
                return self._loaded_models[model_id].model
        
        # Check if we need to evict
        if not self.can_load(size_bytes):
            if self.enable_auto_eviction:
                self._evict_until_free(size_bytes)
            elif not force:
                raise MemoryError(
                    f"Not enough VRAM for {model_id}. "
                    f"Need {size_bytes / 1024**3:.2f} GB, "
                    f"have {self.get_free_vram() / 1024**3:.2f} GB free."
                )
        
        # Load the model
        start_time = time.perf_counter()
        model = loader_fn()
        load_time = (time.perf_counter() - start_time) * 1000
        
        # Register
        with self._lock:
            self._loaded_models[model_id] = ModelInfo(
                model_id=model_id,
                model=model,
                size_bytes=size_bytes,
                last_used=time.time(),
                load_time_ms=load_time,
                component=component
            )
            self._component_sizes[component] = self._component_sizes.get(component, 0) + size_bytes
        
        logger.info(f"Loaded {model_id} in {load_time:.0f}ms")
        return model
    
    def _evict_until_free(self, required_bytes: int):
        """Evict models until we have enough free space."""
        evicted = []
        
        with self._lock:
            while not self.can_load(required_bytes) and self._loaded_models:
                # Get least recently used model (first item in OrderedDict)
                # Use list() to avoid modifying dict while iterating
                model_ids = list(self._loaded_models.keys())
                if not model_ids:
                    break
                    
                model_id = model_ids[0]
                info = self._loaded_models[model_id]
                
                # Don't evict if it's the only model
                if len(self._loaded_models) == 1:
                    logger.warning("Cannot evict last remaining model")
                    break
                
                # Evict
                logger.info(f"Evicting {model_id} ({info.size_bytes / 1024**2:.1f} MB)")
                
                # Cleanup model
                del info.model
                
                # Update tracking
                if info.component in self._component_sizes:
                    self._component_sizes[info.component] -= info.size_bytes
                
                del self._loaded_models[model_id]
                evicted.append(model_id)
                
                # Clear CUDA cache
                self._clear_cuda_cache()
        
        # Notify callbacks
        for model_id in evicted:
            for callback in self._on_eviction_callbacks:
                try:
                    callback(model_id)
                except:
                    pass
    
    def evict_by_component(self, component: str):
        """Evict all models of a specific component type."""
        with self._lock:
            to_evict = [
                model_id for model_id, info in self._loaded_models.items()
                if info.component == component
            ]
        
        for model_id in to_evict:
            self.unload_model(model_id)
    
    def unload_model(self, model_id: str):
        """Explicitly unload a model."""
        with self._lock:
            if model_id in self._loaded_models:
                info = self._loaded_models[model_id]
                
                # Cleanup
                del info.model
                
                # Update tracking
                if info.component in self._component_sizes:
                    self._component_sizes[info.component] -= info.size_bytes
                
                del self._loaded_models[model_id]
                
                logger.info(f"Unloaded {model_id}")
        
        self._clear_cuda_cache()
    
    def unload_all(self):
        """Unload all models."""
        with self._lock:
            for info in self._loaded_models.values():
                del info.model
            self._loaded_models.clear()
            self._component_sizes.clear()
        
        self._clear_cuda_cache()
        gc.collect()
        
        logger.info("Unloaded all models")
    
    def _clear_cuda_cache(self):
        """Clear CUDA cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
    
    def force_garbage_collect(self):
        """Force garbage collection and CUDA cache clear."""
        gc.collect()
        self._clear_cuda_cache()
        logger.info("Forced garbage collection")
    
    def get_memory_breakdown(self) -> MemoryBreakdown:
        """Get detailed breakdown of memory usage by component."""
        breakdown = MemoryBreakdown()
        
        with self._lock:
            for component, size in self._component_sizes.items():
                size_mb = size / 1024**2
                
                if component == "tts":
                    breakdown.tts_model_mb = size_mb
                elif component == "vocoder":
                    breakdown.vocoder_mb = size_mb
                elif component == "whisper":
                    breakdown.whisper_mb = size_mb
                elif component == "kv_cache":
                    breakdown.kv_cache_mb = size_mb
                elif component == "audio":
                    breakdown.audio_buffer_mb = size_mb
        
        # Get total from GPU
        stats = self.get_vram_stats()
        breakdown.total_mb = stats["allocated_gb"] * 1024
        breakdown.available_mb = stats["free_gb"] * 1024
        
        return breakdown
    
    def get_loaded_models(self) -> List[Dict]:
        """Get list of currently loaded models."""
        with self._lock:
            return [
                {
                    "id": info.model_id,
                    "component": info.component,
                    "size_mb": info.size_bytes / 1024**2,
                    "last_used": info.last_used,
                    "age_seconds": time.time() - info.last_used
                }
                for info in self._loaded_models.values()
            ]
    
    def on_eviction(self, callback: Callable[[str], None]):
        """Register callback for model eviction events."""
        self._on_eviction_callbacks.append(callback)
    
    def on_low_memory(self, callback: Callable[[float], None]):
        """Register callback for low memory events."""
        self._on_low_memory_callbacks.append(callback)
    
    def check_memory_health(self) -> Dict[str, Any]:
        """Check memory health and return status."""
        stats = self.get_vram_stats()
        usage_percent = stats["usage_percent"]
        
        if usage_percent >= 95:
            status = "critical"
            message = "VRAM critically low! Eviction recommended."
        elif usage_percent >= 85:
            status = "warning"
            message = "VRAM usage high."
        elif usage_percent >= 70:
            status = "normal"
            message = "VRAM usage moderate."
        else:
            status = "healthy"
            message = "VRAM usage healthy."
        
        # Trigger callbacks if over threshold
        if usage_percent >= 90:
            for callback in self._on_low_memory_callbacks:
                try:
                    callback(usage_percent)
                except:
                    pass
        
        return {
            "status": status,
            "message": message,
            "usage_percent": usage_percent,
            "free_gb": stats["free_gb"],
            "loaded_models": len(self._loaded_models)
        }


# CPU Offload helpers
class CPUOffloadManager:
    """
    Manager for offloading model components to CPU.
    """
    
    def __init__(self):
        self._offloaded: Dict[str, Any] = {}
    
    def offload_to_cpu(self, model_id: str, model: Any) -> bool:
        """
        Offload a model to CPU memory.
        
        Returns True if successful.
        """
        try:
            import torch
            
            if hasattr(model, 'to'):
                model.to('cpu')
                self._offloaded[model_id] = model
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Offloaded {model_id} to CPU")
                return True
                
        except Exception as e:
            logger.error(f"Failed to offload {model_id}: {e}")
        
        return False
    
    def restore_to_gpu(self, model_id: str, device: str = "cuda") -> Optional[Any]:
        """
        Restore an offloaded model to GPU.
        
        Returns the model if successful, None otherwise.
        """
        if model_id not in self._offloaded:
            return None
        
        try:
            model = self._offloaded[model_id]
            
            if hasattr(model, 'to'):
                model.to(device)
                del self._offloaded[model_id]
                
                logger.info(f"Restored {model_id} to {device}")
                return model
                
        except Exception as e:
            logger.error(f"Failed to restore {model_id}: {e}")
        
        return None
    
    def get_offloaded_models(self) -> List[str]:
        """Get list of offloaded model IDs."""
        return list(self._offloaded.keys())


# Global instance
_memory_manager: Optional[AdaptiveMemoryManager] = None
_cpu_offload_manager: Optional[CPUOffloadManager] = None


def get_memory_manager() -> AdaptiveMemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = AdaptiveMemoryManager()
    return _memory_manager


def get_cpu_offload_manager() -> CPUOffloadManager:
    """Get the global CPU offload manager instance."""
    global _cpu_offload_manager
    if _cpu_offload_manager is None:
        _cpu_offload_manager = CPUOffloadManager()
    return _cpu_offload_manager
