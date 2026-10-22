# Contributing

Thanks for taking a look at VoiceSynth Studio. This is a local-first TTS
orchestration project; the most valuable contributions make the existing
synthesis paths more reliable or make the incomplete ones honest.

## Good first contributions

- Implement a real synthesis path for a runtime target that currently raises
  `NotImplementedError` (ONNX, OpenVINO, TensorRT, DirectML, llama.cpp, vLLM).
- Add a reproducible benchmark harness with documented hardware. Do not add
  synthetic numbers to the README or site.
- Add tests that run without downloading model weights.
- Improve the Fish Speech and GPT-SoVITS integrations.
- Fix PII/path leakage or Windows/Linux portability issues.

## Development setup

```bash
git clone https://github.com/FiredMosquito831/LOCAL-Text-to-Speech-Synthesis.git
cd LOCAL-Text-to-Speech-Synthesis
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# Model-free test + docs dependencies
pip install -r requirements-dev.txt
python -m pytest

# Optional: full runtime environment (large, Windows/CUDA 12.1 lockfile)
pip install -r requirements_ultimate.txt
```

Model weights are not required for the test suite. The full runtime can be
several gigabytes and needs Python 3.10–3.12.

## Tests

```bash
python -m pytest                      # focused, model-free tests
python test_all.py                    # legacy smoke suite (imports PyQt6)
python scripts/build_site.py --check  # docs must be committed and valid
```

If you change `main.py`'s argparse options, `core/*/CAPABILITIES`, or
`BackendRegistry` registrations, rebuild the site:

```bash
python scripts/build_site.py
# optionally refresh images/screenshots:
python scripts/make_og.py
python scripts/capture_screenshots.py
```

The CI workflow runs the pytest suite, the legacy test suite on a Windows runner,
and the site build check. It does not download model weights.

## Adding or changing a backend

1. Implement `TTSBackend` in `core/backend_<name>.py` with a real `load()`,
   `synthesize()` and `unload()`.
2. Declare honest `CAPABILITIES` (VRAM floor, precision, platform support).
3. Register it in `core/backend_registry.py` if it should appear in the app.
4. If `synthesize()` is not implemented, leave it raising `NotImplementedError`.
   Do not advertise the backend as usable in the README or site; the generator
   marks it as `interface`.
5. Add a model-free test for the capability declaration and status extraction.
6. Rebuild the site so the matrix reflects the change.

## Code style

- Python 3.10+ syntax, 4-space indentation, type hints where practical.
- Keep heavy imports (`torch`, model packages, PyQt6) inside functions when
  possible so the model-free test suite stays fast.
- Log clearly and do not silently return fabricated audio. If a fallback tone is
  used, log a warning and document it as a limitation.
- Update `README.md`, `CHANGELOG.md` and the relevant site page for user-facing
  changes.

## Pull requests

- Keep changes focused and describe what was tested.
- Do not commit model weights, virtual environments, caches, `.env` files or
  generated screenshots that were not produced from the built site.
- Do not add telemetry, analytics, tracking pixels or external fonts/scripts to
  the documentation site.
- Preserve the MIT license and existing third-party acknowledgements.

## License

By contributing, you agree that your contribution is licensed under the MIT
License in this repository.
