"""
Quantization Manager
=====================
Central manager for model quantization with multiple methods.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Supported quantization types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8_DYNAMIC = "int8-dynamic"
    INT8_STATIC = "int8-static"
    INT4_GPTQ = "int4-gptq"
    INT4_AWQ = "int4-awq"
    INT4_BNBQ = "int4-bnb"  # bitsandbytes NF4
    GGUF_Q8_0 = "gguf-q8_0"
    GGUF_Q5_K_M = "gguf-q5_k_m"
    GGUF_Q4_K_M = "gguf-q4_k_m"
    GGUF_Q3_K_M = "gguf-q3_k_m"
    GGUF_Q2_K = "gguf-q2_k"
    FP8_E4M3 = "fp8-e4m3"


@dataclass
class QuantizationInfo:
    """Information about a quantization method."""
    name: str
    bits: int
    vram_reduction: float  # Percentage reduction vs FP32
    quality_rating: int  # 1-5
    speed_impact: float  # Multiplier (> 1 = faster)
    requires_calibration: bool
    supported_backends: list
    description: str


# Quantization method specifications
QUANT_SPECS: Dict[QuantizationType, QuantizationInfo] = {
    QuantizationType.FP32: QuantizationInfo(
        name="FP32 (Full Precision)",
        bits=32,
        vram_reduction=0,
        quality_rating=5,
        speed_impact=1.0,
        requires_calibration=False,
        supported_backends=["pytorch", "onnx", "openvino"],
        description="Full 32-bit precision. Maximum quality, highest memory usage."
    ),
    QuantizationType.FP16: QuantizationInfo(
        name="FP16 (Half Precision)",
        bits=16,
        vram_reduction=50,
        quality_rating=5,
        speed_impact=1.5,
        requires_calibration=False,
        supported_backends=["pytorch", "tensorrt", "vllm", "onnx"],
        description="16-bit floating point. Near-lossless quality with 50% memory savings."
    ),
    QuantizationType.BF16: QuantizationInfo(
        name="BF16 (Brain Float)",
        bits=16,
        vram_reduction=50,
        quality_rating=5,
        speed_impact=1.5,
        requires_calibration=False,
        supported_backends=["pytorch", "tensorrt"],
        description="BFloat16 format. Better numerical range than FP16, requires Ampere+."
    ),
    QuantizationType.INT8_DYNAMIC: QuantizationInfo(
        name="INT8 Dynamic",
        bits=8,
        vram_reduction=60,
        quality_rating=4,
        speed_impact=1.8,
        requires_calibration=False,
        supported_backends=["pytorch", "onnx"],
        description="Dynamic INT8 quantization. No calibration needed, ~60% memory savings."
    ),
    QuantizationType.INT8_STATIC: QuantizationInfo(
        name="INT8 Static (PTQ)",
        bits=8,
        vram_reduction=65,
        quality_rating=5,
        speed_impact=2.0,
        requires_calibration=True,
        supported_backends=["tensorrt", "onnx", "openvino"],
        description="Static INT8 with calibration. Best INT8 quality, requires ~100 samples."
    ),
    QuantizationType.INT4_GPTQ: QuantizationInfo(
        name="INT4 GPTQ",
        bits=4,
        vram_reduction=75,
        quality_rating=3,
        speed_impact=2.5,
        requires_calibration=True,
        supported_backends=["pytorch", "vllm"],
        description="4-bit GPTQ quantization. Significant compression with moderate quality loss."
    ),
    QuantizationType.INT4_AWQ: QuantizationInfo(
        name="INT4 AWQ",
        bits=4,
        vram_reduction=75,
        quality_rating=4,
        speed_impact=2.5,
        requires_calibration=True,
        supported_backends=["pytorch", "vllm"],
        description="4-bit AWQ quantization. Better quality than GPTQ for same size."
    ),
    QuantizationType.INT4_BNBQ: QuantizationInfo(
        name="INT4 NF4 (bitsandbytes)",
        bits=4,
        vram_reduction=75,
        quality_rating=4,
        speed_impact=1.5,
        requires_calibration=False,
        supported_backends=["pytorch"],
        description="4-bit NormalFloat via bitsandbytes. Easy to use, good quality."
    ),
    QuantizationType.GGUF_Q8_0: QuantizationInfo(
        name="GGUF Q8_0",
        bits=8,
        vram_reduction=50,
        quality_rating=5,
        speed_impact=1.2,
        requires_calibration=False,
        supported_backends=["llama-cpp"],
        description="GGUF 8-bit. Best GGUF quality, larger file size."
    ),
    QuantizationType.GGUF_Q5_K_M: QuantizationInfo(
        name="GGUF Q5_K_M",
        bits=5,
        vram_reduction=65,
        quality_rating=4,
        speed_impact=1.3,
        requires_calibration=False,
        supported_backends=["llama-cpp"],
        description="GGUF 5-bit mixed. Good balance of size and quality."
    ),
    QuantizationType.GGUF_Q4_K_M: QuantizationInfo(
        name="GGUF Q4_K_M",
        bits=4,
        vram_reduction=75,
        quality_rating=3,
        speed_impact=1.4,
        requires_calibration=False,
        supported_backends=["llama-cpp"],
        description="GGUF 4-bit mixed. Popular choice for low VRAM."
    ),
    QuantizationType.GGUF_Q3_K_M: QuantizationInfo(
        name="GGUF Q3_K_M",
        bits=3,
        vram_reduction=80,
        quality_rating=2,
        speed_impact=1.5,
        requires_calibration=False,
        supported_backends=["llama-cpp"],
        description="GGUF 3-bit. Very small, noticeable quality loss."
    ),
    QuantizationType.GGUF_Q2_K: QuantizationInfo(
        name="GGUF Q2_K",
        bits=2,
        vram_reduction=85,
        quality_rating=1,
        speed_impact=1.6,
        requires_calibration=False,
        supported_backends=["llama-cpp"],
        description="GGUF 2-bit. Smallest possible, significant quality loss."
    ),
    QuantizationType.FP8_E4M3: QuantizationInfo(
        name="FP8 E4M3 (Ada+)",
        bits=8,
        vram_reduction=50,
        quality_rating=5,
        speed_impact=2.0,
        requires_calibration=False,
        supported_backends=["tensorrt"],
        description="8-bit float for Ada/Hopper GPUs. Fast with minimal quality loss."
    ),
}


class QuantizationManager:
    """
    Central manager for model quantization.
    
    Supports:
    - Runtime quantization (bitsandbytes)
    - Static quantization (GPTQ, AWQ)
    - GGUF conversion
    - FP8 for Ada+ GPUs
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or "~/.cache/voicesynth/quantized").expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_available_methods(self) -> Dict[QuantizationType, QuantizationInfo]:
        """Get all available quantization methods based on installed packages."""
        available = {}
        
        for qtype, info in QUANT_SPECS.items():
            if self._check_method_available(qtype):
                available[qtype] = info
        
        return available
    
    def _check_method_available(self, qtype: QuantizationType) -> bool:
        """Check if a quantization method is available."""
        try:
            if qtype in [QuantizationType.FP32, QuantizationType.FP16, QuantizationType.BF16]:
                import torch
                return True
            
            if qtype == QuantizationType.INT8_DYNAMIC:
                import torch
                return hasattr(torch, 'quantization')
            
            if qtype == QuantizationType.INT4_BNBQ:
                import bitsandbytes
                return True
            
            if qtype in [QuantizationType.INT4_GPTQ]:
                import auto_gptq
                return True
            
            if qtype in [QuantizationType.INT4_AWQ]:
                import awq
                return True
            
            if qtype.value.startswith("gguf"):
                import llama_cpp
                return True
            
            if qtype == QuantizationType.FP8_E4M3:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    return props.major >= 9  # Hopper or Ada
                return False
            
            return True  # Assume available for unknown types
            
        except ImportError:
            return False
    
    def get_recommendation(self, vram_gb: float, prefer_quality: bool = True) -> QuantizationType:
        """
        Get recommended quantization based on available VRAM.
        
        Args:
            vram_gb: Available VRAM in GB
            prefer_quality: If True, prefer quality over memory savings
        """
        if vram_gb >= 8:
            return QuantizationType.FP16
        elif vram_gb >= 5:
            return QuantizationType.INT8_DYNAMIC
        elif vram_gb >= 3:
            if self._check_method_available(QuantizationType.INT4_AWQ):
                return QuantizationType.INT4_AWQ
            return QuantizationType.INT4_BNBQ
        elif vram_gb >= 1.5:
            if self._check_method_available(QuantizationType.GGUF_Q4_K_M):
                return QuantizationType.GGUF_Q4_K_M
            return QuantizationType.INT4_BNBQ
        else:
            if self._check_method_available(QuantizationType.GGUF_Q2_K):
                return QuantizationType.GGUF_Q2_K
            return QuantizationType.INT4_BNBQ
    
    def apply_bitsandbytes_quantization(
        self,
        model,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True
    ):
        """
        Apply bitsandbytes quantization to a model.
        
        Args:
            model: The model to quantize
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            bnb_4bit_compute_dtype: Compute dtype for 4-bit
            bnb_4bit_quant_type: Quantization type (nf4 or fp4)
            bnb_4bit_use_double_quant: Use double quantization for more savings
        """
        try:
            import torch
            from transformers import BitsAndBytesConfig
            
            if load_in_4bit:
                compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant
                )
                
                logger.info(f"Applied 4-bit quantization ({bnb_4bit_quant_type})")
                return bnb_config
            
            elif load_in_8bit:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Applied 8-bit quantization")
                return bnb_config
            
            return None
            
        except ImportError:
            logger.warning("bitsandbytes not installed for quantization")
            return None
    
    def convert_to_gguf(
        self,
        model_path: Path,
        output_path: Path,
        quantization: str = "q4_k_m",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> bool:
        """
        Convert a model to GGUF format.
        
        Note: This requires llama.cpp's convert scripts.
        """
        logger.info(f"Converting to GGUF {quantization}: {model_path} -> {output_path}")
        
        if progress_callback:
            progress_callback(10, "Preparing conversion...")
        
        # This is a placeholder - actual conversion requires llama.cpp scripts
        # or gguf-py package
        try:
            # Try using gguf package if available
            import gguf
            
            if progress_callback:
                progress_callback(50, "Converting model...")
            
            # Conversion logic would go here
            # For now, just log the intent
            logger.warning(
                "GGUF conversion requires manual steps. "
                "See: https://github.com/ggerganov/llama.cpp#prepare-and-quantize"
            )
            
            if progress_callback:
                progress_callback(100, "Conversion complete (placeholder)")
            
            return False
            
        except ImportError:
            logger.warning("gguf package not installed. Install with: pip install gguf")
            return False
    
    def benchmark_quantization(
        self,
        model,
        test_text: str = "Hello, this is a test for quality benchmarking.",
        reference_audio: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark the quality of quantized model output.
        
        Returns metrics like:
        - Inference time
        - Memory usage
        - (Optionally) quality metrics vs reference
        """
        import time
        
        results = {
            "inference_time_ms": 0,
            "memory_allocated_mb": 0,
            "memory_reserved_mb": 0,
        }
        
        try:
            import torch
            
            # Clear cache before benchmark
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            if progress_callback:
                progress_callback(20, "Running inference benchmark...")
            
            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Run inference
            start_time = time.perf_counter()
            
            # Placeholder - actual inference would go here
            # output = model.synthesize(test_text)
            
            end_time = time.perf_counter()
            
            results["inference_time_ms"] = (end_time - start_time) * 1000
            
            # Measure memory
            if torch.cuda.is_available():
                results["memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
                results["memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
                results["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            
            if progress_callback:
                progress_callback(100, "Benchmark complete!")
            
        except Exception as e:
            logger.error(f"Benchmark error: {e}")
            results["error"] = str(e)
        
        return results
    
    def estimate_memory_usage(self, model_size_gb: float, qtype: QuantizationType) -> float:
        """Estimate memory usage after quantization."""
        info = QUANT_SPECS.get(qtype)
        if info:
            reduction = info.vram_reduction / 100
            return model_size_gb * (1 - reduction)
        return model_size_gb
