"""Backend declarations stay honest and parseable by the site build."""

from __future__ import annotations

from core.backend_base import BackendType, TTSBackend
from core.backend_f5 import F5TTSBackend
from core.backend_gptsovits import GPTSoVITSBackend
from core.backend_fish import FishSpeechBackend
from core.backend_registry import BackendRegistry
from scripts.repo_data import build_data


def test_capabilities_are_declared():
    for backend in (F5TTSBackend, GPTSoVITSBackend, FishSpeechBackend):
        assert issubclass(backend, TTSBackend)
        assert backend.CAPABILITIES.name
        assert backend.CAPABILITIES.quality_rating in range(1, 6)


def test_registry_contains_the_model_backends():
    registered = BackendRegistry.list_all()
    assert BackendType.PYTORCH_NATIVE in registered
    assert BackendType.GPT_SOVITS in registered
    assert BackendType.FISH_SPEECH in registered


def test_site_data_matches_source_declarations():
    data = build_data()
    assert data["version"] == "0.1.1"
    model_names = {backend.name for backend in data["model_backends"]}
    assert "F5-TTS (PyTorch)" in model_names
    assert "GPT-SoVITS" in model_names
    assert "Fish Speech" in model_names
    assert len(data["backends"]) == 9


def test_unimplemented_runtimes_are_labelled():
    data = build_data()
    runtimes = {backend.name: backend for backend in data["runtimes"]}
    for name in ("ONNX Runtime", "OpenVINO", "TensorRT-LLM", "DirectML", "llama.cpp (GGUF)", "vLLM"):
        backend = runtimes[name]
        assert backend.supports_synthesis is False
        assert backend.status == "interface"


def test_cli_options_are_extracted_from_argparse():
    data = build_data()
    flags = {flag for option in data["cli"]["options"] for flag in option.flags}
    assert {"--text", "--file", "--voice", "--output", "--backend", "--quantization", "--device"} <= flags
