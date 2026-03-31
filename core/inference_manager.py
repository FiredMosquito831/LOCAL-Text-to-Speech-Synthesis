"""
Inference Manager
==================
Central manager for TTS inference across multiple backends.
Handles model loading, quantization, and generation.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class Backend(Enum):
    """Supported TTS backends."""
    F5_TTS = "f5-tts"
    GPT_SOVITS = "gpt-sovits"
    FISH_SPEECH = "fish-speech"


class QuantizationType(Enum):
    """Supported quantization types."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    GGUF_Q8 = "gguf-q8"
    GGUF_Q4 = "gguf-q4"


class InferenceManager:
    """
    Central inference manager for voice synthesis.
    
    Supports multiple backends with automatic fallback
    and memory-aware model loading.
    """
    
    def __init__(
        self,
        backend: str = "f5-tts",
        quantization: str = "int8",
        device: str = "auto"
    ):
        self.backend_name = backend
        self.quantization = quantization
        self.device = self._resolve_device(device)
        
        self._backend_instance = None
        self._loaded = False
        
        # Current voice profile (for cloning)
        self._current_voice_path: Optional[str] = None
        self._current_voice_text: Optional[str] = None
        
        logger.info(f"InferenceManager: backend={backend}, quant={quantization}, device={self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device
    
    def load_model(self, progress_callback: Optional[Callable[[int, str], None]] = None):
        """Load the TTS model with specified quantization."""
        if self._loaded:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading model: {self.backend_name}")
        
        if self.backend_name == "f5-tts":
            self._load_f5_tts(progress_callback)
        elif self.backend_name == "gpt-sovits":
            self._load_gpt_sovits(progress_callback)
        elif self.backend_name == "fish-speech":
            self._load_fish_speech(progress_callback)
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")
        
        self._loaded = True
        logger.info("Model loaded successfully")
    
    def _load_f5_tts(self, progress_callback: Optional[Callable] = None):
        """Load F5-TTS backend."""
        from core.backend_f5 import F5TTSBackend
        
        self._backend_instance = F5TTSBackend(
            model_name="F5TTS_v1_Base",
            device=self.device,
            quantization=self.quantization
        )
        self._backend_instance.load(progress_callback)
    
    def _load_gpt_sovits(self, progress_callback: Optional[Callable] = None):
        """Load GPT-SoVITS backend."""
        from core.backend_gptsovits import GPTSoVITSBackend
        
        self._backend_instance = GPTSoVITSBackend(
            model_name="pretrained_v2",
            device=self.device,
            quantization=self.quantization
        )
        self._backend_instance.load(progress_callback)
    
    def _load_fish_speech(self, progress_callback: Optional[Callable] = None):
        """Load Fish Speech backend."""
        from core.backend_fish import FishSpeechBackend
        
        self._backend_instance = FishSpeechBackend(
            model_name="fish-speech-1.5",
            device=self.device,
            quantization=self.quantization
        )
        self._backend_instance.load(progress_callback)
    
    def set_voice(self, voice_reference: str, text: Optional[str] = None):
        """
        Set the voice profile for cloning.
        
        Args:
            voice_reference: Can be:
                - Path to reference audio file (e.g., "C:/audio/ref.wav")
                - Profile name (e.g., "Andrei") - will be resolved from database
                - Profile ID
            text: Transcript of reference audio (optional, will be auto-transcribed)
        """
        from pathlib import Path
        
        audio_path = None
        transcript = text
        
        # Check if it's a file path (contains path separators or ends with audio extension)
        path_check = Path(voice_reference)
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.mpeg'}
        
        if path_check.suffix.lower() in audio_extensions and path_check.exists():
            # It's a direct file path
            audio_path = str(path_check)
            logger.info(f"Voice set from file: {audio_path}")
        else:
            # Try to resolve from profile database
            try:
                from core.profile_database import get_profile_database
                db = get_profile_database()
                
                # First try to get by ID
                profile = db.get_profile(voice_reference)
                
                # If not found, search by name
                if not profile:
                    profiles = db.list_profiles(search=voice_reference)
                    if profiles:
                        # Find exact name match first
                        for p in profiles:
                            if p.name.lower() == voice_reference.lower():
                                profile = p
                                break
                        # Otherwise use first match
                        if not profile:
                            profile = profiles[0]
                
                if profile and profile.reference_samples:
                    # Get the first (or best) sample
                    best_sample = profile.reference_samples[0]
                    audio_path = best_sample.get("path", "")
                    transcript = best_sample.get("transcript", "") or text
                    
                    if audio_path and Path(audio_path).exists():
                        logger.info(f"Voice set from profile '{profile.name}': {audio_path}")
                    else:
                        logger.warning(f"Profile '{profile.name}' audio not found: {audio_path}")
                        audio_path = None
                else:
                    logger.warning(f"Profile not found: {voice_reference}")
                    
            except Exception as e:
                logger.error(f"Failed to resolve voice profile: {e}")
        
        # Set the resolved path
        self._current_voice_path = audio_path
        self._current_voice_text = transcript
        
        if audio_path:
            logger.info(f"Voice configured: {audio_path}")
        else:
            logger.warning(f"Could not resolve voice: {voice_reference}")

    def set_adapter(self, adapter: Optional[str]):
        """
        Set the voice adapter (for F5-TTS).
        
        Args:
            adapter: Name or path of the adapter, or None to unload.
        """
        if not self._backend_instance:
            logger.warning("Cannot set adapter: No backend loaded")
            return
            
        if self.backend_name != "f5-tts":
            logger.warning(f"Adapters not supported for backend: {self.backend_name}")
            return
            
        try:
            if adapter:
                logger.info(f"Setting adapter: {adapter}")
                # Assuming backend has load_adapter method (we added it to F5TTSBackend)
                if hasattr(self._backend_instance, "load_adapter"):
                    self._backend_instance.load_adapter(adapter)
                else:
                    logger.warning("Backend does not support 'load_adapter'")
            else:
                logger.info("Unloading adapter...")
                if hasattr(self._backend_instance, "unload_adapter"):
                    self._backend_instance.unload_adapter()
        except Exception as e:
            logger.error(f"Error setting adapter: {e}")
    
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        nfe_steps: int = 16,
        sampler: str = "euler",
        speed: float = 1.0,
        seed: int = -1,
        language: str = "en",
        progress_callback: Optional[Callable[[int, str], None]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice: Voice profile name or path to reference audio
            nfe_steps: Number of function evaluations (quality/speed tradeoff)
            sampler: ODE sampler (euler, midpoint, rk4)
            speed: Speech speed multiplier
            seed: Random seed (-1 for random)
            language: Target language code (en, ro, zh, auto)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not self._loaded:
            self.load_model(progress_callback)
        
        # Auto-detect language if requested
        if language == "auto":
            language = self._detect_language(text)
        
        logger.info(f"Synthesizing: {len(text)} chars, lang={language}, NFE={nfe_steps}, speed={speed}")
        
        # Determine reference audio
        ref_audio = voice or self._current_voice_path
        ref_text = self._current_voice_text
        
        # Set seed if specified
        if seed >= 0:
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            except ImportError:
                pass
        
        # Route to backend synthesize method
        if self._backend_instance is not None:
            # Filter kwargs based on backend capabilities
            backend_kwargs = {
                "text": text,
                "ref_audio_path": ref_audio,
                "ref_text": ref_text,
                "speed": speed,
                "language": language,
                "progress_callback": progress_callback,
            }
            
            # Only pass nfe_steps to backends that support it (F5-TTS)
            if self.backend_name == "f5-tts":
                backend_kwargs["nfe_steps"] = nfe_steps
            
            # Add any extra kwargs that aren't None
            for k, v in kwargs.items():
                if v is not None:
                    backend_kwargs[k] = v
            
            return self._backend_instance.synthesize(**backend_kwargs)
        else:
            logger.error("No backend instance available")
            return np.zeros(24000 * 3, dtype=np.float32), 24000
    
    def _detect_language(self, text: str) -> str:
        """Auto-detect language from text."""
        # Simple heuristic detection
        romanian_chars = set('ăîâșțĂÎÂȘȚ')
        chinese_range = range(0x4E00, 0x9FFF)
        
        text_set = set(text)
        
        # Check for Romanian diacritics
        if text_set & romanian_chars:
            return "ro"
        
        # Check for Chinese characters
        for char in text:
            if ord(char) in chinese_range:
                return "zh"
        
        # Default to English
        return "en"
    
    def save_audio(
        self, 
        audio: np.ndarray, 
        path: Union[str, Path], 
        sample_rate: int = 24000
    ):
        """Save audio to file."""
        import soundfile as sf
        sf.write(str(path), audio, sample_rate)
        logger.info(f"Audio saved: {path}")
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self._backend_instance is not None:
            self._backend_instance.unload()
            self._backend_instance = None
        
        self._loaded = False
        
        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("Model unloaded")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if self._backend_instance:
            return self._backend_instance.get_memory_usage()
        
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "reserved_gb": torch.cuda.memory_reserved() / 1024**3
                }
        except ImportError:
            pass
        return {"allocated_gb": 0, "reserved_gb": 0}
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def switch_backend(self, backend: str):
        """Switch to a different backend."""
        if backend == self.backend_name and self._loaded:
            return
        
        # Unload current
        self.unload_model()
        
        # Switch
        self.backend_name = backend
        logger.info(f"Switched to backend: {backend}")
