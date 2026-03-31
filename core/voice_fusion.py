"""
Voice Fusion
==============
Multi-sample voice embedding fusion for improved voice cloning.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VoiceEmbedding:
    """A voice embedding from a single sample."""
    sample_path: str
    embedding: np.ndarray
    quality_score: float
    duration_s: float
    speaker_id: Optional[str] = None


class VoiceFusion:
    """
    Fuse multiple voice samples into a single speaker representation.
    
    Features:
    - Extract embeddings from multiple samples
    - Weight by quality and duration
    - Multiple fusion strategies
    - Speaker verification
    """
    
    def __init__(
        self,
        backend_type: str = "f5-tts",
        device: str = "cuda"
    ):
        self.backend_type = backend_type
        self.device = device
        self._encoder = None
        self._embeddings_cache: Dict[str, VoiceEmbedding] = {}
    
    def extract_embedding(
        self,
        audio_path: Path,
        use_cache: bool = True
    ) -> VoiceEmbedding:
        """
        Extract voice embedding from an audio file.
        
        Args:
            audio_path: Path to audio file
            use_cache: Use cached embedding if available
            
        Returns:
            VoiceEmbedding object
        """
        cache_key = str(audio_path)
        
        if use_cache and cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]
        
        try:
            # Load audio
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Mono
            
            duration = len(audio) / sr
            
            # Extract embedding based on backend
            if self.backend_type == "f5-tts":
                embedding = self._extract_f5_embedding(audio, sr)
            elif self.backend_type == "gpt-sovits":
                embedding = self._extract_sovits_embedding(audio, sr)
            elif self.backend_type == "fish-speech":
                embedding = self._extract_fish_embedding(audio, sr)
            else:
                embedding = self._extract_generic_embedding(audio, sr)
            
            # Score the sample
            from core.profile_database import QualityScorer
            scorer = QualityScorer()
            score_result = scorer.score_sample(audio_path)
            quality_score = score_result.get('overall', 0.5)
            
            voice_embedding = VoiceEmbedding(
                sample_path=str(audio_path),
                embedding=embedding,
                quality_score=quality_score,
                duration_s=duration
            )
            
            if use_cache:
                self._embeddings_cache[cache_key] = voice_embedding
            
            return voice_embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            raise
    
    def _extract_f5_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract embedding using F5-TTS style."""
        try:
            import torch
            import torchaudio
            
            # Resample if needed
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                audio_tensor = torch.from_numpy(audio).float()
                audio = resampler(audio_tensor).numpy()
                sr = 24000
            
            # Simple mel-based embedding
            mel = self._compute_mel(audio, sr)
            
            # Mean pooling over time
            embedding = mel.mean(axis=1)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"F5 embedding failed, using generic: {e}")
            return self._extract_generic_embedding(audio, sr)
    
    def _extract_sovits_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract embedding using GPT-SoVITS style."""
        # Placeholder - would use BERT encoder
        return self._extract_generic_embedding(audio, sr)
    
    def _extract_fish_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract embedding using Fish Speech style (VQGAN)."""
        # Placeholder - would use VQGAN encoder
        return self._extract_generic_embedding(audio, sr)
    
    def _extract_generic_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Generic embedding using mel spectrogram statistics."""
        mel = self._compute_mel(audio, sr, n_mels=80)
        
        # Statistics over time
        mean = mel.mean(axis=1)
        std = mel.std(axis=1)
        max_val = mel.max(axis=1)
        
        # Concatenate statistics
        embedding = np.concatenate([mean, std, max_val])
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _compute_mel(
        self,
        audio: np.ndarray,
        sr: int,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256
    ) -> np.ndarray:
        """Compute mel spectrogram."""
        try:
            import librosa
            return librosa.feature.melspectrogram(
                y=audio.astype(np.float32),
                sr=sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length
            )
        except ImportError:
            # Fallback to simple FFT
            from scipy import signal
            
            f, t, Sxx = signal.spectrogram(audio, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
            
            # Rough mel binning
            mel_bins = np.linspace(0, len(f)-1, n_mels, dtype=int)
            mel = Sxx[mel_bins, :]
            
            return mel
    
    def fuse_embeddings(
        self,
        embeddings: List[VoiceEmbedding],
        strategy: str = "quality_weighted"
    ) -> np.ndarray:
        """
        Fuse multiple voice embeddings into one.
        
        Args:
            embeddings: List of VoiceEmbedding objects
            strategy: Fusion strategy
                - "average": Simple mean
                - "quality_weighted": Weight by quality score
                - "duration_weighted": Weight by duration
                - "combined_weighted": Weight by quality * duration
                
        Returns:
            Fused embedding array
        """
        if not embeddings:
            raise ValueError("No embeddings to fuse")
        
        if len(embeddings) == 1:
            return embeddings[0].embedding
        
        # Stack embeddings
        stacked = np.stack([e.embedding for e in embeddings])
        
        if strategy == "average":
            return stacked.mean(axis=0)
        
        elif strategy == "quality_weighted":
            weights = np.array([e.quality_score for e in embeddings])
            weights = weights / weights.sum()
            return np.average(stacked, axis=0, weights=weights)
        
        elif strategy == "duration_weighted":
            weights = np.array([e.duration_s for e in embeddings])
            weights = weights / weights.sum()
            return np.average(stacked, axis=0, weights=weights)
        
        elif strategy == "combined_weighted":
            weights = np.array([e.quality_score * e.duration_s for e in embeddings])
            weights = weights / weights.sum()
            return np.average(stacked, axis=0, weights=weights)
        
        else:
            logger.warning(f"Unknown strategy '{strategy}', using average")
            return stacked.mean(axis=0)
    
    def fuse_from_paths(
        self,
        audio_paths: List[Path],
        strategy: str = "quality_weighted"
    ) -> Tuple[np.ndarray, List[VoiceEmbedding]]:
        """
        Convenience method to extract and fuse from paths.
        
        Returns:
            Tuple of (fused_embedding, individual_embeddings)
        """
        embeddings = []
        
        for path in audio_paths:
            try:
                emb = self.extract_embedding(path)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")
        
        if not embeddings:
            raise ValueError("No valid embeddings extracted")
        
        fused = self.fuse_embeddings(embeddings, strategy)
        
        return fused, embeddings
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Returns:
            Similarity score (0 to 1)
        """
        # Normalize
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        emb1_norm = emb1 / norm1
        emb2_norm = emb2 / norm2
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(np.clip(similarity, 0, 1))
    
    def verify_speaker(
        self,
        embeddings: List[VoiceEmbedding],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Verify that all samples are from the same speaker.
        
        Returns:
            Dict with verification results
        """
        if len(embeddings) < 2:
            return {
                "verified": True,
                "min_similarity": 1.0,
                "avg_similarity": 1.0,
                "inconsistent_pairs": []
            }
        
        similarities = []
        inconsistent = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self.compute_similarity(
                    embeddings[i].embedding,
                    embeddings[j].embedding
                )
                similarities.append(sim)
                
                if sim < threshold:
                    inconsistent.append({
                        "sample1": embeddings[i].sample_path,
                        "sample2": embeddings[j].sample_path,
                        "similarity": sim
                    })
        
        return {
            "verified": len(inconsistent) == 0,
            "min_similarity": min(similarities) if similarities else 1.0,
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 1.0,
            "inconsistent_pairs": inconsistent
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embeddings_cache.clear()


# Convenience function
def fuse_voice_samples(
    audio_paths: List[Path],
    backend: str = "f5-tts",
    strategy: str = "quality_weighted"
) -> np.ndarray:
    """
    Quickly fuse multiple audio samples into one embedding.
    
    Args:
        audio_paths: List of audio file paths
        backend: TTS backend type
        strategy: Fusion strategy
        
    Returns:
        Fused embedding array
    """
    fusion = VoiceFusion(backend_type=backend)
    fused, _ = fusion.fuse_from_paths(audio_paths, strategy)
    return fused
