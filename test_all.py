"""
Comprehensive Test Suite for VoiceSynth Studio
===============================================
Tests all major components and identifies bugs.
"""

import sys
import traceback
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

results = []

def test(name):
    """Decorator to register and run tests."""
    def decorator(func):
        def wrapper():
            try:
                func()
                results.append((name, "PASS", None))
                print(f"✓ {name}")
            except Exception as e:
                results.append((name, "FAIL", str(e)))
                print(f"✗ {name}: {e}")
        return wrapper
    return decorator

# ============== CORE MODULE TESTS ==============

@test("Import core.backend_base")
def test_backend_base():
    from core.backend_base import TTSBackend, BackendType, BackendCapabilities
    assert len(BackendType) >= 9, f"Expected 9+ backend types, got {len(BackendType)}"

@test("Import core.backend_f5")
def test_backend_f5():
    from core.backend_f5 import F5TTSBackend
    assert F5TTSBackend.CAPABILITIES is not None

@test("Import core.backend_gptsovits")
def test_backend_gptsovits():
    from core.backend_gptsovits import GPTSoVITSBackend
    assert GPTSoVITSBackend.CAPABILITIES is not None

@test("Import core.backend_fish")
def test_backend_fish():
    from core.backend_fish import FishSpeechBackend
    assert FishSpeechBackend.CAPABILITIES is not None

@test("Import core.inference_manager")
def test_inference_manager():
    from core.inference_manager import InferenceManager
    manager = InferenceManager(backend="f5-tts", quantization="fp16", device="cpu")
    assert manager.device == "cpu"

@test("Import core.backend_registry")
def test_backend_registry():
    from core.backend_registry import BackendRegistry, BackendAutoSelector
    from core.backend_base import BackendType
    # Check GPT-SoVITS and Fish Speech are registered
    all_backends = BackendRegistry.list_all()
    assert BackendType.GPT_SOVITS in all_backends, "GPT_SOVITS not registered"
    assert BackendType.FISH_SPEECH in all_backends, "FISH_SPEECH not registered"

@test("Import core.memory_manager")
def test_memory_manager():
    from core.memory_manager import AdaptiveMemoryManager, get_memory_manager
    manager = get_memory_manager()
    assert manager is not None
    stats = manager.get_vram_stats()
    assert "total_gb" in stats

@test("Import core.batch_queue")
def test_batch_queue():
    from core.batch_queue import BatchQueue, TTSJob, JobStatus
    queue = BatchQueue()
    job_id = queue.add_job("Test text")
    assert job_id is not None
    assert queue.get_pending_count() == 1

@test("Import core.chunking")
def test_chunking():
    from core.chunking import TextChunker, AudioStitcher
    chunker = TextChunker()  # Uses default strategy
    chunks = chunker.chunk("This is a test. Another sentence.")
    assert len(chunks) >= 1

@test("Import core.profile_database")
def test_profile_database():
    from core.profile_database import ProfileDatabase, SpeakerProfile
    # Just import test
    assert SpeakerProfile is not None

@test("Import core.quantization")
def test_quantization():
    from core.quantization import QuantizationManager, QuantizationType
    manager = QuantizationManager()
    assert manager is not None

@test("Import core.voice_fusion")
def test_voice_fusion():
    from core.voice_fusion import VoiceFusion
    fuser = VoiceFusion()
    assert fuser is not None

@test("Import core.presets")
def test_presets():
    from core.presets import PresetManager, OptimizationPreset
    manager = PresetManager()
    presets = manager.list_all_presets()  # Correct method name
    assert len(presets) >= 3

# ============== UI MODULE TESTS ==============

@test("Import ui.styles")
def test_styles():
    from ui.styles import DARK_STYLESHEET, apply_dark_theme
    assert DARK_STYLESHEET is not None

@test("Import ui.tts_panel (no Qt app)")
def test_tts_panel_import():
    # Just test import works
    import ui.tts_panel
    assert ui.tts_panel.TTSPanel is not None

@test("Import ui.cloning_panel (no Qt app)")
def test_cloning_panel_import():
    import ui.cloning_panel
    assert ui.cloning_panel.VoiceCloningPanel is not None

@test("Import ui.monitor_widget (no Qt app)")
def test_monitor_widget_import():
    import ui.monitor_widget
    assert ui.monitor_widget.GPUMonitorWidget is not None

@test("Import ui.batch_panel (no Qt app)")
def test_batch_panel_import():
    import ui.batch_panel
    assert ui.batch_panel.BatchPanel is not None

# ============== UTILS TESTS ==============

@test("Import utils.gpu_monitor")
def test_gpu_monitor():
    from utils.gpu_monitor import GPUMonitor, GPUStats
    monitor = GPUMonitor()
    stats = monitor.get_stats()
    assert isinstance(stats, GPUStats)

# ============== INTEGRATION TESTS ==============

@test("InferenceManager synthesize parameter filtering")
def test_synth_params():
    from core.inference_manager import InferenceManager
    import numpy as np
    
    # Create manager but don't load model
    manager = InferenceManager(backend="f5-tts", device="cpu")
    
    # Check it doesn't crash when we try to synthesize without loading
    # (should return placeholder audio)
    try:
        # This will fail but shouldn't crash with parameter errors
        audio, sr = manager.synthesize("Test", nfe_steps=16, speed=1.0)
    except Exception as e:
        # Expected - model not loaded
        if "nfe_steps" in str(e) or "sampler" in str(e):
            raise AssertionError(f"Parameter filtering failed: {e}")

@test("Backend synthesize signatures")
def test_backend_signatures():
    import inspect
    from core.backend_f5 import F5TTSBackend
    from core.backend_gptsovits import GPTSoVITSBackend
    from core.backend_fish import FishSpeechBackend
    
    # Check all backends accept **kwargs
    for backend_cls in [F5TTSBackend, GPTSoVITSBackend, FishSpeechBackend]:
        sig = inspect.signature(backend_cls.synthesize)
        params = list(sig.parameters.keys())
        assert "kwargs" in params or any(p.startswith("**") for p in params), \
            f"{backend_cls.__name__}.synthesize missing **kwargs"

# ============== RUN ALL TESTS ==============

if __name__ == "__main__":
    print("=" * 60)
    print("VoiceSynth Studio - Comprehensive Test Suite")
    print("=" * 60)
    print()
    
    # Get all test functions
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    
    for test_fn in tests:
        test_fn()
    
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r[1] == "PASS")
    failed = sum(1 for r in results if r[1] == "FAIL")
    
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print()
        print("FAILURES:")
        for name, status, error in results:
            if status == "FAIL":
                print(f"  - {name}: {error}")
    
    print()
