"""CLI contract tests that do not import torch or load models."""

from __future__ import annotations

import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        timeout=60,
    )


def test_version_flag():
    result = run_cli("--version")
    assert result.returncode == 0
    assert "VoiceSynth Studio 0.1.1" in result.stdout


def test_help_lists_real_options():
    result = run_cli("--help")
    assert result.returncode == 0
    for flag in (
        "--cli",
        "--text",
        "--file",
        "--voice",
        "--output",
        "--backend",
        "--quantization",
        "--device",
    ):
        assert flag in result.stdout


def test_help_lists_backend_and_device_choices():
    result = run_cli("--help")
    assert "f5-tts" in result.stdout
    assert "gpt-sovits" in result.stdout
    assert "fish-speech" in result.stdout
    assert "cuda" in result.stdout
    assert "cpu" in result.stdout


def test_missing_input_exits_non_zero():
    result = run_cli("--cli")
    assert result.returncode != 0
    assert "No input text provided" in (result.stderr + result.stdout)


def test_missing_file_exits_non_zero():
    result = run_cli("--cli", "--file", "does-not-exist.txt")
    assert result.returncode != 0
    assert "File not found" in (result.stderr + result.stdout)
