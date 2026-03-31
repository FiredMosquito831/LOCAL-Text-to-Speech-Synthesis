"""
GPT-SoVITS Backend
===================
Backend integration for GPT-SoVITS voice synthesis.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Callable, List
import numpy as np

from core.backend_base import TTSBackend, BackendCapabilities

logger = logging.getLogger(__name__)


class GPTSoVITSBackend(TTSBackend):
    """
    GPT-SoVITS Backend for voice synthesis.
    
    Supports:
    - Zero-shot voice cloning
    - Fine-tuned models
    - Multiple languages
    """
    
    CAPABILITIES = BackendCapabilities(
        name="GPT-SoVITS",
        min_vram_gb=2.0,
        supports_quantization=["fp32", "fp16", "int8"],
        speed_multiplier=1.2,
        quality_rating=5,
        requires_compilation=False,
        windows_supported=True,
        linux_supported=True
    )
    
    # Model configurations
    MODEL_CONFIGS = {
        "pretrained_v2": {
            "gpt_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "sovits_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
        },
        "pretrained_v1": {
            "gpt_path": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "sovits_path": "GPT_SoVITS/pretrained_models/s2G488k.pth",
        }
    }
    
    def __init__(
        self,
        model_name: str = "pretrained_v2",
        device: str = "cuda",
        quantization: str = "fp16",
        gpt_path: Optional[str] = None,
        sovits_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self.quantization = quantization
        self.gpt_path = gpt_path
        self.sovits_path = sovits_path
        
        self._gpt_model = None
        self._sovits_model = None
        self._ssl_model = None
        
        # Inference settings
        self.target_sample_rate = 32000
        self.top_k = 5
        self.top_p = 1.0
        self.temperature = 1.0
        self.ref_free = False
        
        logger.info(f"GPTSoVITSBackend initialized: model={model_name}, device={device}")
    
    def load(self, progress_callback: Optional[Callable[[int, str], None]] = None):
        """Load GPT-SoVITS models."""
        if self._loaded:
            return
        
        try:
            if progress_callback:
                progress_callback(10, "Loading GPT-SoVITS components...")
            
            # Determine model paths
            config = self.MODEL_CONFIGS.get(self.model_name, {})
            gpt_path = self.gpt_path or config.get("gpt_path")
            sovits_path = self.sovits_path or config.get("sovits_path")
            
            if progress_callback:
                progress_callback(30, "Loading GPT model...")
            
            # Import GPT-SoVITS modules
            try:
                from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
                
                # Create TTS config
                tts_config = TTS_Config({
                    "device": self.device,
                    "is_half": self.quantization == "fp16",
                    "t2s_weights_path": gpt_path,
                    "vits_weights_path": sovits_path
                })
                
                if progress_callback:
                    progress_callback(60, "Initializing TTS engine...")
                
                self._tts = TTS(tts_config)
                
            except ImportError:
                # Fallback: basic loading without full GPT-SoVITS package
                logger.warning("GPT-SoVITS package not found, using stub implementation")
                self._tts = None
            
            self._loaded = True
            
            if progress_callback:
                progress_callback(100, "GPT-SoVITS loaded!")
            
            logger.info("GPT-SoVITS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load GPT-SoVITS: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "auto",
        speed: float = 1.0,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech using GPT-SoVITS.
        
        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio for voice cloning
            ref_text: Transcript of reference audio
            language: Target language (auto, zh, en, ja, ko, etc.)
            speed: Speech speed multiplier
            progress_callback: Progress callback
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not self._loaded:
            self.load(progress_callback)
        
        if self._tts is None:
            # Return placeholder audio when package not installed
            logger.warning(
                "GPT-SoVITS package not installed. Returning placeholder audio. "
                "Install from: https://github.com/RVC-Boss/GPT-SoVITS"
            )
            if progress_callback:
                progress_callback(100, "GPT-SoVITS not installed - placeholder audio")
            
            # Generate simple placeholder tone
            duration = max(1.0, len(text) * 0.08)  # rough estimate
            t = np.linspace(0, duration, int(self.target_sample_rate * duration))
            audio = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3 tone
            return audio.astype(np.float32), self.target_sample_rate
        
        try:
            if progress_callback:
                progress_callback(20, "Preparing input...")
            
            # Prepare inputs
            inputs = {
                "text": text,
                "text_lang": language if language != "auto" else "zh",
                "ref_audio_path": ref_audio_path,
                "prompt_text": ref_text,
                "prompt_lang": language if language != "auto" else "zh",
                "top_k": self.top_k,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "speed_factor": speed,
            }
            
            if progress_callback:
                progress_callback(50, "Generating audio...")
            
            # Run inference
            result = self._tts.run(inputs)
            
            # Collect audio chunks
            audio_chunks = []
            for chunk in result:
                if chunk is not None:
                    sample_rate, audio_data = chunk
                    audio_chunks.append(audio_data)
            
            if audio_chunks:
                audio = np.concatenate(audio_chunks)
            else:
                raise RuntimeError("No audio generated")
            
            if progress_callback:
                progress_callback(100, "Generation complete!")
            
            return audio, sample_rate
            
        except Exception as e:
            logger.error(f"GPT-SoVITS synthesis error: {e}")
            raise
    
    def unload(self):
        """Unload GPT-SoVITS models."""
        if self._tts is not None:
            del self._tts
            self._tts = None
        
        if self._gpt_model is not None:
            del self._gpt_model
            self._gpt_model = None
        
        if self._sovits_model is not None:
            del self._sovits_model
            self._sovits_model = None
        
        self._loaded = False
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        logger.info("GPT-SoVITS model unloaded")
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if GPT-SoVITS is available."""
        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        return (
            "git clone https://github.com/RVC-Boss/GPT-SoVITS\n"
            "cd GPT-SoVITS\n"
            "pip install -r requirements.txt\n\n"
            "Download pretrained models from:\n"
            "https://huggingface.co/lj1995/GPT-SoVITS"
        )
