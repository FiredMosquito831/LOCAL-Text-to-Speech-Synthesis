"""
GPU Monitor Utility
====================
Real-time GPU memory monitoring using pynvml.
Falls back gracefully if NVIDIA GPU not available.
"""

import threading
import time
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import pynvml (nvidia-ml-py)
# The 'pynvml' name is deprecated, but nvidia-ml-py still uses it
try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
        import pynvml  # type: ignore[import]
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml/nvidia-ml-py not installed. GPU monitoring disabled.")


@dataclass
class GPUStats:
    """Container for GPU statistics."""
    name: str = "Unknown"
    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    free_memory_mb: float = 0.0
    utilization_percent: float = 0.0
    temperature_c: int = 0
    available: bool = False
    
    @property
    def memory_percent(self) -> float:
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100


class GPUMonitor:
    """
    Real-time GPU monitoring with callback support.
    
    Usage:
        monitor = GPUMonitor()
        monitor.start(callback=lambda stats: print(stats))
        ...
        monitor.stop()
    """
    
    def __init__(self, device_index: int = 0, poll_interval: float = 1.0):
        self.device_index = device_index
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[GPUStats], None]] = None
        self._initialized = False
        
        # Initialize NVML
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._initialized = True
                logger.info("GPU monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
    
    def get_stats(self) -> GPUStats:
        """Get current GPU statistics."""
        if not self._initialized or not PYNVML_AVAILABLE:
            return GPUStats(available=False)
        
        try:
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            total_mb = mem_info.total / (1024 ** 2)
            used_mb = mem_info.used / (1024 ** 2)
            free_mb = mem_info.free / (1024 ** 2)
            
            # GPU name
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                utilization = util.gpu
            except:
                utilization = 0
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, 
                    pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                temp = 0
            
            return GPUStats(
                name=name,
                total_memory_mb=total_mb,
                used_memory_mb=used_mb,
                free_memory_mb=free_mb,
                utilization_percent=utilization,
                temperature_c=temp,
                available=True
            )
            
        except Exception as e:
            logger.error(f"Error getting GPU stats: {e}")
            return GPUStats(available=False)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            stats = self.get_stats()
            if self._callback:
                try:
                    self._callback(stats)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            time.sleep(self.poll_interval)
    
    def start(self, callback: Optional[Callable[[GPUStats], None]] = None):
        """Start background monitoring."""
        if self._running:
            return
        
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("GPU monitoring started")
    
    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("GPU monitoring stopped")
    
    def __del__(self):
        """Cleanup NVML on destruction."""
        self.stop()
        if self._initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


# Singleton instance
_monitor_instance: Optional[GPUMonitor] = None


def get_gpu_monitor() -> GPUMonitor:
    """Get the global GPU monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = GPUMonitor()
    return _monitor_instance
