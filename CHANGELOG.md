# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.1] — 2026-10-22

### Added

- Generated static documentation site in `docs/` (GitHub Pages) with an
  audio-studio identity: overview, quickstart, CLI/GUI reference, backend
  matrix, performance/limits, FAQ, security/about and a custom 404.
- Deterministic site generator (`scripts/build_site.py`) and data extraction
  from `main.py`, `core/backend_*.py`, `core/presets.py` and
  `core/backend_registry.py`.
- Single `SITE_URL` constant (`scripts/site_config.py`), per-page canonical,
  Open Graph/Twitter metadata, JSON-LD, `sitemap.xml`, `robots.txt`,
  `CNAME` (`tts.danubelabs.net`) and a real 1200×630 `og.png`.
- Playwright screenshot tooling and committed screenshots under
  `site/static/screenshots/`.
- Model-free pytest suite (`tests/`) covering batch jobs, chunking, profiles,
  voice fusion, CLI flags, backend status extraction and site generation.
- CI workflow: pytest + site build check on Linux, legacy smoke suite on
  Windows.
- `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE` (MIT),
  `requirements-site.txt` and `requirements-dev.txt`.
- `--version` CLI flag and a shared `core.__version__` constant.

### Fixed

- **CLI output regression:** `run_cli()` passed the `(audio, sample_rate)` tuple
  to `save_audio()` and assumed 24 kHz. It now unpacks the tuple and writes the
  backend's actual sample rate (e.g. 32 kHz for GPT-SoVITS).
- **Batch queue was unusable:** `BatchJob` was declared without `@dataclass`,
  so constructing any job raised `BatchJob() takes no arguments`. Added the
  dataclass decorator and the missing `ref_audio_path`, `output_path` and
  `settings` fields.
- **Batch jobs silently "completed" without a manager.** The worker now marks
  the job failed with an explicit error, passes adapter and reference audio, and
  writes per-job output files.
- Legacy smoke test updated to the corrected `BatchJob` class.
- Mobile navigation no longer clips the last item; the docs site has no
  horizontal overflow at 390 px.

### Changed

- README rewritten around verified capabilities, an honest backend table,
  real screenshots, installation, quickstart and known limitations.
- Backend status is now explicit: F5-TTS is implemented, GPT-SoVITS is
  implemented with a stub fallback, Fish Speech is a scaffold, and the six
  runtime targets are interfaces only.
- Version bumped from 0.1.0 to 0.1.1.

### Security

- Documented the local-only data flow, the absence of a network listener and
  telemetry, the model-weight download path, profile storage and the
  consent/watermarking limitations.

## [0.1.0] — 2026-03-31

### Added

- Initial multi-backend TTS application: F5-TTS, GPT-SoVITS and Fish Speech
  integrations behind a `TTSBackend` contract.
- PyQt6 studio with TTS, voice cloning, backend configuration, quantization,
  presets, batch processing, waveform display and GPU monitor.
- CLI entry point, inference manager, backend registry/auto-selector, adaptive
  memory manager, SQLite speaker profiles, chunking/stitching, voice fusion,
  quantization catalogue and VRAM presets.
- Training workspace for F5-TTS dataset preparation.
