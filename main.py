"""
VoiceSynth Studio - Main Entry Point
=====================================
A state-of-the-art local voice synthesis application with:
- Multiple TTS backends (F5-TTS, GPT-SoVITS, Fish Speech)
- Voice cloning with faster-whisper transcription
- Memory-optimized for 6GB VRAM GPUs
- PyQt6 GUI with CLI fallback

Usage:
    python main.py              # Launch GUI
    python main.py --cli        # CLI mode
    python main.py --help       # Show help
"""

import sys
import argparse
import logging
from pathlib import Path

from core import __version__ as APP_VERSION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("VoiceSynth")

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VoiceSynth Studio - Local Voice Synthesis & Cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Launch GUI
  python main.py --cli --text "Hello"   # Generate speech via CLI
  python main.py --cli --file input.txt # Generate from file
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode instead of GUI"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"VoiceSynth Studio {APP_VERSION}"
    )
    
    # CLI-specific options
    cli_group = parser.add_argument_group("CLI Options")
    cli_group.add_argument(
        "--text", "-t",
        type=str,
        help="Text to synthesize"
    )
    cli_group.add_argument(
        "--file", "-f",
        type=Path,
        help="Text file to synthesize"
    )
    cli_group.add_argument(
        "--voice", "-v",
        type=str,
        default=None,
        help="Voice profile to use (name or path to reference audio)"
    )
    cli_group.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output.wav"),
        help="Output audio file path (default: output.wav)"
    )
    cli_group.add_argument(
        "--backend", "-b",
        type=str,
        choices=["f5-tts", "gpt-sovits", "fish-speech"],
        default="f5-tts",
        help="TTS backend to use (default: f5-tts)"
    )
    
    # Performance options
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--quantization", "-q",
        type=str,
        choices=["fp16", "int8", "int4", "gguf-q4", "gguf-q8"],
        default="int8",
        help="Quantization level (default: int8)"
    )
    perf_group.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (default: auto)"
    )
    
    return parser.parse_args()


def run_gui():
    """Launch the PyQt6 GUI application."""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from ui.main_window import MainWindow
    except ImportError as e:
        logger.error(f"PyQt6 not installed: {e}")
        logger.info("Install with: pip install PyQt6")
        sys.exit(1)
    
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("VoiceSynth Studio")
    app.setOrganizationName("VoiceSynth")
    app.setApplicationVersion(APP_VERSION)
    
    # Apply dark theme
    try:
        from ui.styles import apply_dark_theme
        apply_dark_theme(app)
    except ImportError:
        logger.warning("Could not load custom theme")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


def run_cli(args):
    """Run in CLI mode for batch processing."""
    from core.inference_manager import InferenceManager
    
    # Get text to synthesize
    if args.text:
        text = args.text
    elif args.file:
        if not args.file.exists():
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        text = args.file.read_text(encoding="utf-8")
    else:
        logger.error("No input text provided. Use --text or --file")
        sys.exit(1)
    
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Text length: {len(text)} characters")
    
    # Initialize inference manager
    manager = InferenceManager(
        backend=args.backend,
        quantization=args.quantization,
        device=args.device
    )
    
    # Generate audio
    try:
        audio, sample_rate = manager.synthesize(
            text=text,
            voice=args.voice,
        )

        # Save output
        manager.save_audio(audio, args.output, sample_rate)
        logger.info(f"Audio saved to: {args.output} ({sample_rate} Hz)")
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("=" * 50)
    logger.info(f"VoiceSynth Studio v{APP_VERSION}")
    logger.info("=" * 50)
    
    if args.cli:
        logger.info("Running in CLI mode")
        run_cli(args)
    else:
        logger.info("Launching GUI...")
        run_gui()


if __name__ == "__main__":
    main()
