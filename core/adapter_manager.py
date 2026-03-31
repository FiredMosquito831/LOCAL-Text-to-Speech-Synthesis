"""
Adapter Manager
===============
Manages discovery and loading of voice adapters (LoRA, fine-tunes).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AdapterConfig:
    """Configuration for a voice adapter."""
    name: str
    path: Path
    type: str = "f5-tts"  # f5-tts, gpt-sovits, etc.
    description: str = ""

class AdapterManager:
    """
    Singleton manager for voice adapters.
    Scans `adapters/` directory and other locations.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.adapters_dir = self.project_root / "adapters"
        self._adapters: Dict[str, AdapterConfig] = {}
        self.scan_adapters()
        
    def scan_adapters(self):
        """Scan for available adapters."""
        self._adapters.clear()
        
        # 1. Scan default adapters directory
        if self.adapters_dir.exists():
            self._scan_dir(self.adapters_dir)
            
        # 2. Check for Ro-F5TTS specifically (legacy/specific integration)
        ro_path = self.adapters_dir / "romanian"
        if ro_path.exists():
            self._scan_dir(ro_path, prefix="Ro-F5TTS")
            
        logger.info(f"Found {len(self._adapters)} adapters: {list(self._adapters.keys())}")

    def _scan_dir(self, path: Path, prefix: str = ""):
        """Helper to scan a directory for checkpoint files."""
        # Look for .pt, .pth, .safetensors
        extensions = ["*.pt", "*.pth", "*.safetensors"]
        files = []
        for ext in extensions:
            files.extend(path.glob(f"**/{ext}"))
            
        for file_path in files:
            # Create a clean name
            name = file_path.stem
            if prefix:
                name = f"{prefix} ({name})"
            
            # Use parent folder name if filename is generic 'model' or 'checkpoint'
            if name.lower() in ["model", "checkpoint", "model_last", "best_model"]:
                name = f"{prefix} {file_path.parent.name}" if prefix else file_path.parent.name
                
            self._adapters[name] = AdapterConfig(
                name=name,
                path=file_path,
                type="f5-tts" # Assuming F5 for now, could detect from header
            )

    def get_adapter(self, name: str) -> Optional[AdapterConfig]:
        """Get adapter configuration by name."""
        return self._adapters.get(name)

    def get_all_adapters(self) -> List[str]:
        """Get list of all adapter names."""
        return list(self._adapters.keys())

# Global instance
_instance = None

def get_adapter_manager() -> AdapterManager:
    global _instance
    if _instance is None:
        _instance = AdapterManager()
    return _instance
