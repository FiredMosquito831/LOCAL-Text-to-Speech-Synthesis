"""
Optimization Presets
=====================
Pre-configured optimization settings for different VRAM budgets.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class PresetType(Enum):
    """Built-in preset types."""
    ULTRA_LOW = "ultra_low"
    LOW = "low"
    BALANCED = "balanced"
    QUALITY = "quality"
    CUSTOM = "custom"


@dataclass
class OptimizationPreset:
    """Complete optimization settings preset."""
    
    name: str
    description: str
    target_vram_gb: float
    
    # Backend settings
    backend: str  # pytorch, vllm, tensorrt, onnx, llama-cpp
    quantization: str  # fp32, fp16, int8, int4, gguf-q4, etc.
    
    # Whisper settings
    whisper_model: str  # tiny, base, small, medium, large-v3, large-v3-turbo
    whisper_compute_type: str  # int8, float16, float32
    
    # Memory settings
    enable_flash_attention: bool
    enable_torch_compile: bool
    enable_attention_slicing: bool
    enable_cpu_offload: bool
    max_batch_size: int
    
    # Quality settings
    nfe_steps: int
    expected_quality: int  # 1-5
    expected_rtf: float  # Real-time factor
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "OptimizationPreset":
        return cls(**data)


# Built-in presets
BUILTIN_PRESETS: Dict[PresetType, OptimizationPreset] = {
    PresetType.ULTRA_LOW: OptimizationPreset(
        name="Ultra Low VRAM",
        description="For systems with 3GB or less VRAM. Maximum compression with quality tradeoffs.",
        target_vram_gb=3.0,
        backend="llama-cpp",
        quantization="gguf-q4_k_m",
        whisper_model="tiny",
        whisper_compute_type="int8",
        enable_flash_attention=False,
        enable_torch_compile=False,
        enable_attention_slicing=True,
        enable_cpu_offload=True,
        max_batch_size=1,
        nfe_steps=8,
        expected_quality=2,
        expected_rtf=0.5
    ),
    PresetType.LOW: OptimizationPreset(
        name="Low VRAM",
        description="For systems with 4GB VRAM. Good balance of quality and memory usage.",
        target_vram_gb=4.0,
        backend="pytorch",
        quantization="int8-dynamic",
        whisper_model="small",
        whisper_compute_type="int8",
        enable_flash_attention=True,
        enable_torch_compile=False,
        enable_attention_slicing=True,
        enable_cpu_offload=False,
        max_batch_size=1,
        nfe_steps=16,
        expected_quality=3,
        expected_rtf=0.4
    ),
    PresetType.BALANCED: OptimizationPreset(
        name="Balanced",
        description="For systems with 5-6GB VRAM. Default recommended settings.",
        target_vram_gb=5.0,
        backend="pytorch",
        quantization="int8-dynamic",
        whisper_model="small",
        whisper_compute_type="int8",
        enable_flash_attention=True,
        enable_torch_compile=True,
        enable_attention_slicing=False,
        enable_cpu_offload=False,
        max_batch_size=2,
        nfe_steps=16,
        expected_quality=4,
        expected_rtf=0.3
    ),
    PresetType.QUALITY: OptimizationPreset(
        name="Quality",
        description="For systems with 6GB+ VRAM. Maximum quality, higher memory usage.",
        target_vram_gb=6.0,
        backend="pytorch",
        quantization="fp16",
        whisper_model="medium",
        whisper_compute_type="float16",
        enable_flash_attention=True,
        enable_torch_compile=True,
        enable_attention_slicing=False,
        enable_cpu_offload=False,
        max_batch_size=4,
        nfe_steps=32,
        expected_quality=5,
        expected_rtf=0.25
    ),
}


class PresetManager:
    """
    Manager for optimization presets.
    
    Handles:
    - Built-in preset access
    - Custom preset creation/save/load
    - Preset recommendation based on hardware
    """
    
    def __init__(self, presets_dir: Optional[Path] = None):
        self.presets_dir = Path(presets_dir or "~/.config/voicesynth/presets").expanduser()
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        self._custom_presets: Dict[str, OptimizationPreset] = {}
        self._load_custom_presets()
    
    def get_preset(self, preset_type: PresetType) -> OptimizationPreset:
        """Get a built-in preset."""
        return BUILTIN_PRESETS.get(preset_type)
    
    def get_custom_preset(self, name: str) -> Optional[OptimizationPreset]:
        """Get a custom preset by name."""
        return self._custom_presets.get(name)
    
    def list_all_presets(self) -> Dict[str, OptimizationPreset]:
        """List all available presets (built-in + custom)."""
        all_presets = {}
        
        # Add built-in presets
        for preset_type, preset in BUILTIN_PRESETS.items():
            all_presets[preset_type.value] = preset
        
        # Add custom presets
        for name, preset in self._custom_presets.items():
            all_presets[f"custom_{name}"] = preset
        
        return all_presets
    
    def save_custom_preset(self, preset: OptimizationPreset):
        """Save a custom preset."""
        self._custom_presets[preset.name] = preset
        
        # Save to file
        preset_file = self.presets_dir / f"{preset.name}.json"
        with open(preset_file, "w") as f:
            json.dump(preset.to_dict(), f, indent=2)
        
        logger.info(f"Saved custom preset: {preset.name}")
    
    def delete_custom_preset(self, name: str):
        """Delete a custom preset."""
        if name in self._custom_presets:
            del self._custom_presets[name]
            
            preset_file = self.presets_dir / f"{name}.json"
            if preset_file.exists():
                preset_file.unlink()
            
            logger.info(f"Deleted custom preset: {name}")
    
    def _load_custom_presets(self):
        """Load custom presets from disk."""
        for preset_file in self.presets_dir.glob("*.json"):
            try:
                with open(preset_file) as f:
                    data = json.load(f)
                preset = OptimizationPreset.from_dict(data)
                self._custom_presets[preset.name] = preset
            except Exception as e:
                logger.warning(f"Failed to load preset {preset_file}: {e}")
    
    def recommend_preset(self, available_vram_gb: float) -> PresetType:
        """
        Recommend a preset based on available VRAM.
        
        Args:
            available_vram_gb: Available VRAM in GB
            
        Returns:
            Recommended preset type
        """
        if available_vram_gb >= 6:
            return PresetType.QUALITY
        elif available_vram_gb >= 5:
            return PresetType.BALANCED
        elif available_vram_gb >= 4:
            return PresetType.LOW
        else:
            return PresetType.ULTRA_LOW
    
    def apply_preset(self, preset: OptimizationPreset) -> Dict[str, Any]:
        """
        Convert a preset to a configuration dictionary that can be applied.
        
        Returns settings dict for use by InferenceManager and other components.
        """
        return {
            "backend": preset.backend,
            "quantization": preset.quantization,
            "whisper": {
                "model": preset.whisper_model,
                "compute_type": preset.whisper_compute_type,
            },
            "memory": {
                "flash_attention": preset.enable_flash_attention,
                "torch_compile": preset.enable_torch_compile,
                "attention_slicing": preset.enable_attention_slicing,
                "cpu_offload": preset.enable_cpu_offload,
            },
            "inference": {
                "max_batch_size": preset.max_batch_size,
                "nfe_steps": preset.nfe_steps,
            }
        }
    
    def get_preset_summary(self, preset: OptimizationPreset) -> str:
        """Get a human-readable summary of a preset."""
        quality_stars = "★" * preset.expected_quality + "☆" * (5 - preset.expected_quality)
        
        return (
            f"{preset.name}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Target VRAM: {preset.target_vram_gb} GB\n"
            f"Backend: {preset.backend}\n"
            f"Quantization: {preset.quantization}\n"
            f"Whisper: {preset.whisper_model}\n"
            f"Quality: {quality_stars}\n"
            f"Speed: ~{preset.expected_rtf}x realtime\n"
        )


# Convenience function
def get_recommended_preset() -> OptimizationPreset:
    """Get the recommended preset for the current system."""
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            manager = PresetManager()
            preset_type = manager.recommend_preset(vram)
            return manager.get_preset(preset_type)
    except:
        pass
    
    # Default to balanced
    return BUILTIN_PRESETS[PresetType.BALANCED]
