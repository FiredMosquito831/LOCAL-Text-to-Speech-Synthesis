"""
F5-TTS Backend
===============
Dedicated backend integration for F5-TTS model.
Handles model loading, inference, and optimization.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Callable
import numpy as np

from core.backend_base import TTSBackend, BackendCapabilities
from core.adapter_manager import get_adapter_manager

logger = logging.getLogger(__name__)

# Add F5-TTS to path
F5_TTS_PATH = Path(__file__).parent.parent / "F5-TTS" / "src"
if F5_TTS_PATH.exists() and str(F5_TTS_PATH) not in sys.path:
    sys.path.insert(0, str(F5_TTS_PATH))


class F5TTSBackend(TTSBackend):
    """
    F5-TTS Backend for voice synthesis.
    
    Supports:
    - Model loading with quantization options
    - Zero-shot voice cloning
    - Configurable NFE steps and samplers
    """
    
    CAPABILITIES = BackendCapabilities(
        name="F5-TTS (PyTorch)",
        min_vram_gb=1.5,  # With INT8
        supports_quantization=["fp32", "fp16", "int8"],
        speed_multiplier=1.0,  # Baseline
        quality_rating=5,
        requires_compilation=False,
        windows_supported=True,
        linux_supported=True
    )
    
    # Default model configurations
    MODEL_CONFIGS = {
        "F5TTS_v1_Base": {
            "repo": "SWivid/F5-TTS",
            "ckpt": "F5TTS_v1_Base/model_1250000.safetensors",
            "config": {
                "dim": 1024,
                "depth": 22,
                "heads": 16,
                "ff_mult": 2,
                "text_dim": 512,
                "conv_layers": 4,
            }
        },
        "E2TTS_Base": {
            "repo": "SWivid/E2-TTS", 
            "ckpt": "E2TTS_Base/model_1200000.safetensors",
            "config": {
                "dim": 1024,
                "depth": 24,
                "heads": 16,
                "ff_mult": 4,
            }
        }
    }
    
    def __init__(
        self,
        model_name: str = "F5TTS_v1_Base",
        device: str = "cuda",
        quantization: str = "fp16",
        use_flash_attn: bool = True,
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self.quantization = quantization
        self.use_flash_attn = use_flash_attn
        
        self._model = None
        self._vocoder = None
        self._current_adapter = None
        self._base_model_state = None  # To restore after adapter use
        
        # Inference settings
        self.target_sample_rate = 24000
        self.nfe_step = 32
        self.cfg_strength = 2.0
        self.sway_sampling_coef = -1.0
        self.cross_fade_duration = 0.15
        
        logger.info(f"F5TTSBackend initialized: model={model_name}, device={device}, quant={quantization}")
    
    def load(self, progress_callback: Optional[Callable[[int, str], None]] = None):
        """Load the F5-TTS model and vocoder."""
        if self._loaded:
            logger.info("Model already loaded")
            return
        
        try:
            import torch
            from huggingface_hub import hf_hub_download
            
            if progress_callback:
                progress_callback(10, "Loading F5-TTS components...")
            
            # Import F5-TTS modules
            from f5_tts.model import DiT
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            
            if progress_callback:
                progress_callback(30, "Downloading model weights...")
            
            # Get model config
            config = self.MODEL_CONFIGS.get(self.model_name)
            if not config:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            # Download checkpoint
            ckpt_path = hf_hub_download(
                repo_id=config["repo"],
                filename=config["ckpt"]
            )
            
            if progress_callback:
                progress_callback(50, "Loading model to GPU...")
            
            # Load model
            self._model = load_model(
                DiT,
                config["config"],
                ckpt_path,
                device=self.device
            )
            
            # Apply quantization if requested
            if self.quantization == "int8":
                self._apply_int8_quantization()
            
            if progress_callback:
                progress_callback(80, "Loading vocoder...")
            
            # Load vocoder
            self._vocoder = load_vocoder(device=self.device)
            
            self._loaded = True
            
            if progress_callback:
                progress_callback(100, "Model loaded successfully!")
            
            logger.info("F5-TTS model loaded successfully")
            
        except ImportError as e:
            logger.error(f"F5-TTS import error: {e}")
            raise ImportError(
                "F5-TTS not properly installed. Run: pip install -e ./F5-TTS"
            ) from e
    
    def _apply_int8_quantization(self):
        """Apply memory-efficient precision.
        
        Note: PyTorch dynamic INT8 quantization only works on CPU, not CUDA.
        bitsandbytes INT8 requires special model loading at init time.
        For CUDA models, we use FP16 which provides good memory savings.
        """
        try:
            import torch
            
            if self.device == "cuda":
                # INT8 dynamic quantization doesn't work on CUDA
                # Use FP16 for memory efficiency instead
                logger.info("Using FP16 precision for CUDA (INT8 not supported on CUDA)")
                self._model = self._model.half()
            else:
                # CPU: can use dynamic quantization
                logger.info("Applying INT8 dynamic quantization for CPU...")
                self._model = torch.quantization.quantize_dynamic(
                    self._model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("INT8 dynamic quantization applied")
            
        except Exception as e:
            logger.warning(f"Precision optimization failed: {e}")
    
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        nfe_steps: int = 32,
        speed: float = 1.0,
        language: str = "en",
        progress_callback: Optional[Callable[[int, str], None]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio for voice cloning
            ref_text: Transcript of reference audio (auto-transcribed if not provided)
            nfe_steps: Number of function evaluations
            speed: Speech speed multiplier
            language: Target language (en, ro, zh)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not self._loaded:
            self.load(progress_callback)
        
        try:
            from f5_tts.infer.utils_infer import (
                preprocess_ref_audio_text,
                infer_process
            )
            
            if progress_callback:
                progress_callback(10, f"Synthesizing ({language.upper()})...")
            
            # Preprocess text for Romanian if needed
            # Legacy text normalization removed in favor of adapter
            if language == "ro" and not self._current_adapter:
                logger.warning("Romanian language selected but no adapter loaded. Quality may be suboptimal.")
            
            if progress_callback:
                progress_callback(20, "Preprocessing reference audio...")
            
            # Use default reference if none provided
            if ref_audio_path is None:
                # Use built-in example
                from importlib.resources import files
                ref_audio_path = str(
                    files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")
                )
                ref_text = "Some call me nature, others call me mother nature."
            
            # Preprocess reference
            ref_audio, ref_text = preprocess_ref_audio_text(
                ref_audio_path,
                ref_text or "",
                show_info=lambda x: logger.debug(x)
            )
            
            if progress_callback:
                progress_callback(40, "Generating audio...")
            
            # Run inference
            audio, sr, _ = infer_process(
                ref_audio,
                ref_text,
                text,
                self._model,
                self._vocoder,
                nfe_step=nfe_steps,
                speed=speed,
                device=self.device,
                show_info=lambda x: logger.debug(x),
                progress=None  # Disable tqdm
            )
            
            if progress_callback:
                progress_callback(100, "Generation complete!")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            raise
    
    def load_adapter(self, adapter_path: str):
        """
        Load a voice adapter (LoRA/finetune).
        
        Args:
            adapter_path: Path to the adapter checkpoint (.pt/.safetensors)
                          or name of the adapter.
        """
        if not self._loaded:
            logger.warning("Cannot load adapter: Model not loaded yet.")
            return

        try:
            import torch
            from f5_tts.infer.utils_infer import load_checkpoint
            
            # Resolve path via manager if it's a name
            manager = get_adapter_manager()
            adapter_config = manager.get_adapter(adapter_path)
            if adapter_config:
                path = adapter_config.path
                logger.info(f"Resolved adapter '{adapter_path}' to {path}")
            else:
                path = Path(adapter_path)
                
            if not path.exists():
                logger.error(f"Adapter file not found: {path}")
                return

            logger.info(f"Loading adapter from {path}...")
            
            # TODO: Implement proper LoRA loading if supported by F5 codebase.
            # Currently F5-TTS fine-tuning usually results in a full checkpoint.
            # We will assume it's a full checkpoint for now and replace weights.
            # Ideally we should merge or use LoRA if architecture supports it.
            
            # Checkpoint loading logic (simplified)
            # If it's a full checkpoint, we might need to reload the whole model?
            # Or use load_checkpoint util.
            
            checkpoint = torch.load(path, map_location=self.device)
            
            # If we haven't saved base state, do it now
            if self._base_model_state is None:
                # Deep copy state dict to memory (expensive but safe)
                import copy
                self._base_model_state = copy.deepcopy(self._model.state_dict())
                
            # Load new weights
            # Note: keys might need matching depending on how it was saved (EMA vs non-EMA)
            if "ema_model_state_dict" in checkpoint:
                state_dict = checkpoint["ema_model_state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
                
            # Filter and load
            self._model.load_state_dict(state_dict, strict=False)
            self._current_adapter = adapter_path
            logger.info(f"Adapter '{adapter_path}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            # Restore base model if partial failure?
            if self._base_model_state:
                self.unload_adapter()

    def unload_adapter(self):
        """Restore base model weights."""
        if self._base_model_state and self._model:
            logger.info("Restoring base model...")
            self._model.load_state_dict(self._base_model_state, strict=False)
            self._current_adapter = None
            logger.info("Adapter unloaded")
    
    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._vocoder is not None:
            del self._vocoder
            self._vocoder = None
        
        self._loaded = False
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        logger.info("F5-TTS model unloaded")
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if F5-TTS is available."""
        try:
            from f5_tts.model import DiT
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        return (
            "cd VoiceSynthesisLocal\n"
            "git clone https://github.com/SWivid/F5-TTS\n"
            "pip install -e ./F5-TTS"
        )
