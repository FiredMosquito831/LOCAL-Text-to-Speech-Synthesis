# Testing Patterns

**Analysis Date:** 2026-03-31

## Test Framework

**Runner:**
- Custom test framework (no pytest/unittest dependency)
- Home-grown decorator-based test registration
- Located in `test_all.py` and `test_imports.py`

**Test Execution:**
```bash
python test_all.py          # Run comprehensive test suite
python test_imports.py      # Quick import verification
```

**No Configuration Files:**
- No `pytest.ini`, `pyproject.toml`, or `setup.cfg` for test configuration
- Tests run directly via Python interpreter

## Test File Organization

**Location:**
- Tests are in project root (not a separate `tests/` directory)
- `tests/` directory exists but is empty
- Co-location pattern: tests alongside main entry point

**Naming:**
- `test_*.py` pattern
- Files: `test_all.py`, `test_imports.py`

**Structure:**
```
VoiceSynthesisLocal/
├── main.py
├── test_all.py           # Comprehensive test suite
├── test_imports.py       # Quick import check
├── core/
├── ui/
└── tests/                # Empty directory
```

## Test Structure

**Test Registration Pattern:**
```python
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
```

**Test Function Naming:**
- `test_*` prefix for all test functions
- Descriptive names indicating what is tested:
  - `test_backend_base()`
  - `test_inference_manager()`
  - `test_memory_manager()`

**Test Execution:**
```python
if __name__ == "__main__":
    print("=" * 60)
    print("VoiceSynth Studio - Comprehensive Test Suite")
    print("=" * 60)

    # Get all test functions
    tests = [v for k, v in list(globals().items())
             if k.startswith("test_") and callable(v)]

    for test_fn in tests:
        test_fn()

    # Print summary
    passed = sum(1 for r in results if r[1] == "PASS")
    print(f"Passed: {passed}/{len(results)}")
```

## Testing Patterns

**Import Verification Tests:**
```python
@test("Import core.backend_base")
def test_backend_base():
    from core.backend_base import TTSBackend, BackendType, BackendCapabilities
    assert len(BackendType) >= 9, f"Expected 9+ backend types, got {len(BackendType)}"
```

**Class Instantiation Tests:**
```python
@test("Import core.inference_manager")
def test_inference_manager():
    from core.inference_manager import InferenceManager
    manager = InferenceManager(backend="f5-tts", quantization="fp16", device="cpu")
    assert manager.device == "cpu"
```

**Functionality Tests:**
```python
@test("Import core.batch_queue")
def test_batch_queue():
    from core.batch_queue import BatchQueue, TTSJob, JobStatus
    queue = BatchQueue()
    job_id = queue.add_job("Test text")
    assert job_id is not None
    assert queue.get_pending_count() == 1
```

**Integration Tests:**
```python
@test("InferenceManager synthesize parameter filtering")
def test_synth_params():
    from core.inference_manager import InferenceManager
    import numpy as np

    manager = InferenceManager(backend="f5-tts", device="cpu")

    try:
        audio, sr = manager.synthesize("Test", nfe_steps=16, speed=1.0)
    except Exception as e:
        if "nfe_steps" in str(e) or "sampler" in str(e):
            raise AssertionError(f"Parameter filtering failed: {e}")
```

**Signature Verification:**
```python
@test("Backend synthesize signatures")
def test_backend_signatures():
    import inspect
    from core.backend_f5 import F5TTSBackend
    from core.backend_gptsovits import GPTSoVITSBackend

    for backend_cls in [F5TTSBackend, GPTSoVITSBackend]:
        sig = inspect.signature(backend_cls.synthesize)
        params = list(sig.parameters.keys())
        assert "kwargs" in params, f"{backend_cls.__name__} missing **kwargs"
```

## Mocking

**No Formal Mocking Framework:**
- No unittest.mock or pytest.mock observed
- Tests use direct instantiation with controlled parameters

**Inline Mocking Pattern:**
```python
# Tests create instances with mock-friendly parameters
manager = InferenceManager(backend="f5-tts", device="cpu")
# Using device="cpu" avoids GPU dependency in tests
```

**Dependency Isolation:**
```python
@test("Import ui.tts_panel (no Qt app)")
def test_tts_panel_import():
    # Just test import works - don't instantiate without Qt app
    import ui.tts_panel
    assert ui.tts_panel.TTSPanel is not None
```

## Test Categories

**Unit Tests:**
- Individual module imports verified
- Class instantiation tested
- Method signatures checked

**Integration Tests:**
- Multi-component interactions tested
- Parameter passing between layers verified
- Backend synthesize signatures validated

**Smoke Tests:**
- `test_imports.py` provides quick verification
- Tests all major imports without deep functionality checks

## Test Coverage

**Coverage Tooling:**
- No coverage.py or pytest-cov configured
- No coverage reporting enforced

**Coverage Gaps:**
- UI widgets not instantiated in tests (require Qt app)
- GPU-dependent tests use CPU fallback
- Backend loading tests skip actual model download

**Modules with Tests:**
| Module | Test Coverage |
|--------|---------------|
| `core/backend_base.py` | Import + class attributes |
| `core/backend_f5.py` | Import + capabilities |
| `core/backend_gptsovits.py` | Import + capabilities |
| `core/backend_fish.py` | Import + capabilities |
| `core/inference_manager.py` | Import + instantiation + synthesis |
| `core/backend_registry.py` | Import + registry check |
| `core/memory_manager.py` | Import + singleton + stats |
| `core/batch_queue.py` | Import + job addition |
| `core/chunking.py` | Import + instantiation |
| `core/profile_database.py` | Import only |
| `core/quantization.py` | Import + instantiation |
| `core/voice_fusion.py` | Import + instantiation |
| `core/presets.py` | Import + instantiation |
| `ui/styles.py` | Import + constant check |
| `ui/tts_panel.py` | Import only (Qt dependency) |
| `ui/cloning_panel.py` | Import only (Qt dependency) |
| `ui/monitor_widget.py` | Import only (Qt dependency) |
| `ui/batch_panel.py` | Import only (Qt dependency) |
| `utils/gpu_monitor.py` | Import + instantiation + stats |

## Quality Assurance Practices

**Pre-commit Checks:**
- No pre-commit hooks configured
- No automated linting (flake8, pylint, ruff)

**Manual Testing:**
- Run `test_imports.py` for quick verification
- Run `test_all.py` for comprehensive testing

**Error Reporting:**
```python
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

passed = sum(1 for r in results if r[1] == "PASS")
failed = sum(1 for r in results if r[1] == "FAIL")

print(f"Passed: {passed}/{len(results)}")
print(f"Failed: {failed}/{len(results)}")

if failed > 0:
    print("\nFAILURES:")
    for name, status, error in results:
        if status == "FAIL":
            print(f"  - {name}: {error}")
```

## Test Utilities

**No Dedicated Test Utilities:**
- No fixtures module
- No conftest.py
- Helper functions defined inline

**Reuse Patterns:**
- Test decorator `@test(name)` provides common error handling
- Results list tracks all test outcomes

## Recommendations for Test Improvements

**Suggested Additions:**
1. Move to pytest for better fixture support
2. Add `tests/` directory with proper test modules
3. Add pytest.ini configuration
4. Implement mocking for external dependencies
5. Add coverage.py integration
6. Create test fixtures for common test data
7. Add integration tests with real model loading (CI/CD)

**Current Test Commands:**
```bash
# Quick import check (1-2 seconds)
python test_imports.py

# Full test suite (5-10 seconds)
python test_all.py
```

---

*Testing analysis: 2026-03-31*
