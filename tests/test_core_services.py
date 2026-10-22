"""Model-free tests for chunking, profiles, fusion and inference helpers."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from core.chunking import AudioStitcher, ChunkingStrategy, LongTextSynthesizer, TextChunker
from core.inference_manager import InferenceManager
from core.profile_database import ProfileDatabase, QualityScorer, SpeakerProfile
from core.voice_fusion import VoiceFusion


def test_chunker_strategies_cover_the_text():
    text = "First sentence. Second sentence! Third one?"
    for strategy in (
        ChunkingStrategy.SENTENCE,
        ChunkingStrategy.FIXED_CHAR,
        ChunkingStrategy.PARAGRAPH,
        ChunkingStrategy.NONE,
    ):
        chunker = TextChunker(strategy=strategy, max_chars=40)
        chunks = chunker.chunk(text)
        assert chunks, strategy
        assert "".join(chunk.text for chunk in chunks).replace(" ", "") != ""
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_chunker_returns_empty_for_blank_text():
    assert TextChunker().chunk("   \n ") == []


def test_audio_stitcher_crossfades_arrays():
    stitcher = AudioStitcher(crossfade_ms=10, sample_rate=1000)
    first = np.ones(100, dtype=np.float32)
    second = np.ones(100, dtype=np.float32)
    output = stitcher.stitch([first, second], normalize=False)
    assert output.dtype == np.float32
    assert len(output) == 190


class FakeBackend:
    def synthesize(self, text, **kwargs):
        return np.zeros(100, dtype=np.float32), 1000


def test_long_text_synthesizer_stitches_chunks():
    synth = LongTextSynthesizer(FakeBackend(), max_chunk_chars=30)
    audio, sample_rate = synth.synthesize("One. Two. Three. Four. Five.")
    assert sample_rate == 1000
    assert len(audio) > 0


def test_profile_database_roundtrip(tmp_path):
    database = ProfileDatabase(db_path=tmp_path / "profiles.db")
    profile = SpeakerProfile(
        id="p-1",
        name="Narrator",
        tags=["en", "warm"],
        description="Test voice",
        reference_samples=[{"path": "ref.wav", "transcript": "hello", "duration_s": 4.0}],
    )
    assert database.save_profile(profile)
    loaded = database.get_profile("p-1")
    assert loaded is not None
    assert loaded.name == "Narrator"
    assert database.list_profiles(search="Narr")[0].id == "p-1"

    export_path = tmp_path / "narrator.vsp"
    assert database.export_profile("p-1", export_path)
    assert export_path.read_text(encoding="utf-8").startswith("{")

    assert database.delete_profile("p-1")
    assert database.get_profile("p-1") is None


def test_quality_scorer_scores_generated_audio(tmp_path):
    path = tmp_path / "tone.wav"
    sample_rate = 16000
    t = np.linspace(0, 4, sample_rate * 4, endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t)
    sf.write(path, audio.astype(np.float32), sample_rate)

    score = QualityScorer().score_sample(path)
    assert 0.0 <= score["overall"] <= 1.0
    assert score["duration_s"] == pytest.approx(4.0, abs=0.01)


def test_voice_fusion_generic_embedding_and_weighting(tmp_path):
    path = tmp_path / "voice.wav"
    t = np.linspace(0, 3, 24000 * 3, endpoint=False)
    sf.write(path, (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 24000)

    fusion = VoiceFusion(backend_type="gpt-sovits", device="cpu")
    embedding = fusion.extract_embedding(path)
    assert embedding.embedding.ndim == 1
    fused = fusion.fuse_embeddings([embedding, embedding], strategy="quality_weighted")
    assert fused.shape == embedding.embedding.shape


def test_inference_manager_language_heuristics():
    manager = InferenceManager(device="cpu")
    assert manager._detect_language("Bună ziua, ce mai faci?") == "ro"
    assert manager._detect_language("你好，世界") == "zh"
    assert manager._detect_language("Hello there") == "en"


def test_save_audio_preserves_sample_rate(tmp_path):
    manager = InferenceManager(device="cpu")
    path = tmp_path / "out.wav"
    manager.save_audio(np.zeros(3200, dtype=np.float32), path, sample_rate=32000)
    audio, sample_rate = sf.read(path)
    assert sample_rate == 32000
    assert len(audio) == 3200
