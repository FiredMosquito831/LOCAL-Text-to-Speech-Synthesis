# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| < 0.1 | No |

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting:

**https://github.com/FiredMosquito831/LOCAL-Text-to-Speech-Synthesis/security/advisories/new**

Do not open a public issue for a suspected vulnerability before a fix is
available. Include:

- A description of the issue and its impact.
- Reproduction steps and the affected commit/version.
- Your environment (OS, Python version, backend, whether CUDA is present).
- Any relevant logs with secrets and personal paths removed.

You can expect an initial acknowledgement within 7 days. This is a
single-maintainer hobby project, so timelines are best effort.

## Scope

VoiceSynth Studio is a local desktop application. In scope:

- The Python orchestration code (`main.py`, `core/`, `ui/`, `utils/`, `scripts/`).
- Unsafe file handling, path traversal or code execution triggered by inputs
  (text files, reference audio, profiles, adapters, model paths).
- Secrets or private data written to logs or bundled into the repository.
- Supply-chain issues in the repository's own scripts.
- The generated documentation site under `docs/`.

Out of scope:

- Vulnerabilities in upstream model code or weights (F5-TTS, GPT-SoVITS, Fish
  Speech, vocoders) — report them upstream.
- Vulnerabilities in PyTorch, PyQt6, ONNX Runtime, OpenVINO, TensorRT,
  llama.cpp or vLLM — report them to those projects.
- The security of any voice you choose to clone, or the ethics of synthetic
  speech generated outside applicable law and consent.
- Model output quality, hallucinations or transcription errors.

## Security model

- The application opens **no listening network socket** and ships no HTTP API.
- There is **no authentication layer**, because there is no remote service.
- Text, reference audio and generated audio are processed locally.
- Model weights are downloaded from Hugging Face on first use. This is the only
  automatic network access.
- Speaker profiles are stored in SQLite under
  `~/.config/voicesynth/profiles.db`. Exported `.vsp` profiles are plain JSON.
- Paths for reference audio, text files, adapters, ONNX/OpenVINO models and
  output WAVs are user-supplied and are not sandboxed. Run the app with the
  privileges you would give any local Python tool.
- Generated audio is plain WAV with no watermark or provenance metadata.

## Known non-security limitations

- Fish Speech and missing GPT-SoVITS dependencies return placeholder tones
  instead of raising. This is a usability behaviour, not a security boundary.
- Several accelerator backends are scaffolds whose `synthesize()` methods raise
  `NotImplementedError`.
- No benchmarks are published; capability metadata is not a measurement.
