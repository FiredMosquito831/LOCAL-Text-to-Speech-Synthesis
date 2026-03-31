"""
Text Chunking and Audio Stitching
===================================
Strategies for processing long text and stitching audio chunks.
"""

import logging
import re
from typing import List, Tuple, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Text chunking strategies."""
    SENTENCE = "sentence"
    FIXED_CHAR = "fixed_char"
    FIXED_TOKEN = "fixed_token"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    NONE = "none"


@dataclass
class TextChunk:
    """A chunk of text for synthesis."""
    text: str
    index: int
    start_pos: int  # Character position in original text
    end_pos: int
    
    @property
    def length(self) -> int:
        return len(self.text)


class TextChunker:
    """
    Split long text into chunks for synthesis.
    
    Strategies:
    - Sentence: Split on sentence boundaries
    - Fixed Character: Fixed character count with overlap
    - Fixed Token: Fixed token count (for LLM models)
    - Paragraph: Split on paragraph breaks
    - Semantic: Use NLP for semantic chunking
    """
    
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
        max_chars: int = 200,
        overlap_chars: int = 20,
        max_tokens: int = 100,
        overlap_tokens: int = 10
    ):
        self.strategy = strategy
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        # Sentence splitting patterns
        self._sentence_pattern = re.compile(
            r'(?<=[.!?。！？])\s+|(?<=[.!?。！？])(?=[A-Z\u4e00-\u9fff])'
        )
    
    def chunk(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks based on the selected strategy.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of TextChunk objects
        """
        text = text.strip()
        
        if not text:
            return []
        
        if self.strategy == ChunkingStrategy.NONE:
            return [TextChunk(text=text, index=0, start_pos=0, end_pos=len(text))]
        
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(text)
        
        elif self.strategy == ChunkingStrategy.FIXED_CHAR:
            return self._chunk_by_fixed_chars(text)
        
        elif self.strategy == ChunkingStrategy.FIXED_TOKEN:
            return self._chunk_by_tokens(text)
        
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(text)
        
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(text)
        
        else:
            return [TextChunk(text=text, index=0, start_pos=0, end_pos=len(text))]
    
    def _chunk_by_sentence(self, text: str) -> List[TextChunk]:
        """Split by sentence boundaries."""
        sentences = self._sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        start_pos = 0
        
        for sentence in sentences:
            # Would adding this sentence exceed max?
            if len(current_chunk) + len(sentence) + 1 > self.max_chars and current_chunk:
                # Save current chunk
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    index=len(chunks),
                    start_pos=start_pos,
                    end_pos=start_pos + len(current_chunk)
                ))
                start_pos += len(current_chunk) + 1
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add remaining
        if current_chunk:
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                index=len(chunks),
                start_pos=start_pos,
                end_pos=start_pos + len(current_chunk)
            ))
        
        return chunks
    
    def _chunk_by_fixed_chars(self, text: str) -> List[TextChunk]:
        """Split by fixed character count with overlap."""
        chunks = []
        pos = 0
        
        while pos < len(text):
            end = min(pos + self.max_chars, len(text))
            
            # Try to break at word boundary
            if end < len(text):
                # Look for last space within range
                last_space = text.rfind(' ', pos, end)
                if last_space > pos:
                    end = last_space
            
            chunk_text = text[pos:end].strip()
            
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_pos=pos,
                    end_pos=end
                ))
            
            # Move forward, accounting for overlap
            pos = end - self.overlap_chars if end < len(text) else end
        
        return chunks
    
    def _chunk_by_tokens(self, text: str) -> List[TextChunk]:
        """Split by token count (simple word-based tokenization)."""
        words = text.split()
        chunks = []
        pos = 0
        char_pos = 0
        
        while pos < len(words):
            end = min(pos + self.max_tokens, len(words))
            chunk_words = words[pos:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(TextChunk(
                text=chunk_text,
                index=len(chunks),
                start_pos=char_pos,
                end_pos=char_pos + len(chunk_text)
            ))
            
            char_pos += len(chunk_text) + 1
            pos = end - self.overlap_tokens if end < len(words) else end
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[TextChunk]:
        """Split by paragraph boundaries."""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        char_pos = 0
        
        for para in paragraphs:
            # If paragraph is too long, chunk it further
            if len(para) > self.max_chars:
                sub_chunks = self._chunk_by_sentence(para)
                for sub in sub_chunks:
                    sub.start_pos += char_pos
                    sub.end_pos += char_pos
                    sub.index = len(chunks)
                    chunks.append(sub)
            else:
                chunks.append(TextChunk(
                    text=para,
                    index=len(chunks),
                    start_pos=char_pos,
                    end_pos=char_pos + len(para)
                ))
            
            char_pos += len(para) + 2  # +2 for paragraph break
        
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[TextChunk]:
        """Semantic chunking using NLP (falls back to sentence if unavailable)."""
        try:
            # Try using spacy for semantic chunking
            import spacy
            
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            
            chunks = []
            current_chunk = ""
            start_pos = 0
            
            for sent in doc.sents:
                sent_text = sent.text.strip()
                
                if len(current_chunk) + len(sent_text) > self.max_chars and current_chunk:
                    chunks.append(TextChunk(
                        text=current_chunk.strip(),
                        index=len(chunks),
                        start_pos=start_pos,
                        end_pos=start_pos + len(current_chunk)
                    ))
                    start_pos += len(current_chunk)
                    current_chunk = sent_text
                else:
                    current_chunk += " " + sent_text if current_chunk else sent_text
            
            if current_chunk:
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    index=len(chunks),
                    start_pos=start_pos,
                    end_pos=start_pos + len(current_chunk)
                ))
            
            return chunks
            
        except ImportError:
            logger.warning("spacy not available, falling back to sentence chunking")
            return self._chunk_by_sentence(text)


class AudioStitcher:
    """
    Stitch multiple audio chunks together with cross-fade.
    
    Features:
    - Linear cross-fade
    - Cosine cross-fade
    - Overlap-add
    - Normalize output
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        crossfade_ms: float = 50,
        fade_type: str = "linear"
    ):
        self.sample_rate = sample_rate
        self.crossfade_ms = crossfade_ms
        self.fade_type = fade_type
    
    @property
    def crossfade_samples(self) -> int:
        return int(self.crossfade_ms * self.sample_rate / 1000)
    
    def stitch(
        self,
        chunks: List[np.ndarray],
        normalize: bool = True,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> np.ndarray:
        """
        Stitch audio chunks together with cross-fade.
        
        Args:
            chunks: List of audio arrays
            normalize: Whether to normalize the output
            progress_callback: Optional progress callback
            
        Returns:
            Stitched audio array
        """
        if not chunks:
            return np.array([], dtype=np.float32)
        
        if len(chunks) == 1:
            return chunks[0]
        
        fade_samples = self.crossfade_samples
        
        # Calculate output length
        total_length = sum(len(c) for c in chunks) - fade_samples * (len(chunks) - 1)
        output = np.zeros(total_length, dtype=np.float32)
        
        pos = 0
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(
                    int((i + 1) / len(chunks) * 100),
                    f"Stitching chunk {i + 1}/{len(chunks)}"
                )
            
            if i == 0:
                # First chunk: no fade-in
                output[:len(chunk)] = chunk
                pos = len(chunk) - fade_samples
            else:
                # Apply cross-fade
                fade_out = self._get_fade_curve(fade_samples, fade_out=True)
                fade_in = self._get_fade_curve(fade_samples, fade_out=False)
                
                # Cross-fade region
                output[pos:pos + fade_samples] *= fade_out
                output[pos:pos + fade_samples] += chunk[:fade_samples] * fade_in
                
                # Non-overlapping region
                output[pos + fade_samples:pos + len(chunk)] = chunk[fade_samples:]
                
                pos += len(chunk) - fade_samples
        
        # Normalize if requested
        if normalize:
            max_val = np.max(np.abs(output))
            if max_val > 0:
                output = output / max_val * 0.95
        
        return output
    
    def _get_fade_curve(self, length: int, fade_out: bool = True) -> np.ndarray:
        """Generate fade curve."""
        if self.fade_type == "linear":
            if fade_out:
                return np.linspace(1, 0, length)
            else:
                return np.linspace(0, 1, length)
        
        elif self.fade_type == "cosine":
            if fade_out:
                return np.cos(np.linspace(0, np.pi / 2, length))
            else:
                return np.sin(np.linspace(0, np.pi / 2, length))
        
        elif self.fade_type == "exponential":
            if fade_out:
                return np.exp(-np.linspace(0, 5, length))
            else:
                return 1 - np.exp(-np.linspace(0, 5, length))
        
        else:
            # Default to linear
            if fade_out:
                return np.linspace(1, 0, length)
            else:
                return np.linspace(0, 1, length)
    
    def add_silence(
        self,
        audio: np.ndarray,
        before_ms: float = 0,
        after_ms: float = 0
    ) -> np.ndarray:
        """Add silence before/after audio."""
        before_samples = int(before_ms * self.sample_rate / 1000)
        after_samples = int(after_ms * self.sample_rate / 1000)
        
        silence_before = np.zeros(before_samples, dtype=audio.dtype)
        silence_after = np.zeros(after_samples, dtype=audio.dtype)
        
        return np.concatenate([silence_before, audio, silence_after])
    
    def trim_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        min_silence_ms: float = 100
    ) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        min_samples = int(min_silence_ms * self.sample_rate / 1000)
        
        # Find first non-silent sample
        start = 0
        for i in range(len(audio)):
            if abs(audio[i]) > threshold:
                start = max(0, i - min_samples)
                break
        
        # Find last non-silent sample
        end = len(audio)
        for i in range(len(audio) - 1, -1, -1):
            if abs(audio[i]) > threshold:
                end = min(len(audio), i + min_samples)
                break
        
        return audio[start:end]


class LongTextSynthesizer:
    """
    High-level synthesizer for long text.
    
    Combines chunking, synthesis, and stitching.
    """
    
    def __init__(
        self,
        backend,  # TTSBackend instance
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
        max_chunk_chars: int = 200,
        crossfade_ms: float = 50
    ):
        self.backend = backend
        self.chunker = TextChunker(
            strategy=chunking_strategy,
            max_chars=max_chunk_chars
        )
        self.stitcher = AudioStitcher(crossfade_ms=crossfade_ms)
    
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize long text with chunking and stitching.
        
        Args:
            text: Long text input
            ref_audio_path: Reference audio for voice cloning
            ref_text: Reference text
            progress_callback: Progress callback
            **kwargs: Additional synthesis parameters
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Chunk the text
        chunks = self.chunker.chunk(text)
        
        if not chunks:
            return np.array([], dtype=np.float32), self.stitcher.sample_rate
        
        logger.info(f"Synthesizing {len(chunks)} chunks")
        
        # Synthesize each chunk
        audio_chunks = []
        sample_rate = self.stitcher.sample_rate
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                base_progress = int(i / len(chunks) * 80)
                progress_callback(
                    base_progress,
                    f"Synthesizing chunk {i + 1}/{len(chunks)}: {chunk.text[:30]}..."
                )
            
            # Synthesize this chunk
            audio, sr = self.backend.synthesize(
                text=chunk.text,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                **kwargs
            )
            
            audio_chunks.append(audio)
            sample_rate = sr
        
        # Update stitcher sample rate
        self.stitcher.sample_rate = sample_rate
        
        if progress_callback:
            progress_callback(80, "Stitching audio chunks...")
        
        # Stitch chunks together
        output = self.stitcher.stitch(audio_chunks, normalize=True)
        
        if progress_callback:
            progress_callback(100, "Complete!")
        
        return output, sample_rate
