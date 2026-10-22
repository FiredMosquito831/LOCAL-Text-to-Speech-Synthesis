"""Extract real product data from the repository for the static site.

Everything here is read from source with :mod:`ast` and plain file reads so the
site build does not need torch, PyQt6 or any model dependency installed. The
generator is deterministic: no timestamps, no network access, no randomness.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.site_config import ROOT  # noqa: E402


# ---------------------------------------------------------------------------
# Small AST helpers
# ---------------------------------------------------------------------------


def _constant(node: ast.AST, default: Any = None) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _constant(node.operand)
        return -value if isinstance(value, (int, float)) else default
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_constant(elt) for elt in node.elts]
    if isinstance(node, ast.Call):
        # Path("output.wav") -> "output.wav"; preserve scalar args.
        args = [_constant(arg) for arg in node.args]
        if len(args) == 1 and isinstance(args[0], str):
            return args[0]
        return None
    return default


def _kw(call: ast.Call, name: str, default: Any = None) -> Any:
    for keyword in call.keywords:
        if keyword.arg == name:
            return _constant(keyword.value, default)
    return default


def _doc_first_line(node: ast.AST) -> str:
    doc = ast.get_docstring(node) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line and set(line) != {"="}:
            return line
    return ""


def parse_source(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ---------------------------------------------------------------------------
# Version & CLI
# ---------------------------------------------------------------------------


def read_version() -> str:
    text = (ROOT / "core" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise ValueError("Could not find __version__ in core/__init__.py")
    return match.group(1)


@dataclass
class CliOption:
    flags: List[str]
    dest: str
    help: str
    group: str
    choices: Optional[List[str]] = None
    default: Any = None
    type: Optional[str] = None
    action: Optional[str] = None


def parse_cli() -> Dict[str, Any]:
    """Return the real argparse options declared in ``main.py``."""
    tree = parse_source(ROOT / "main.py")

    groups: Dict[str, str] = {"parser": "General"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument_group":
            title = _constant(call.args[0]) if call.args else None
            if title:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        groups[target.id] = title

    options: List[CliOption] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue

        flags = [_constant(arg) for arg in node.args]
        flags = [flag for flag in flags if isinstance(flag, str)]
        if not flags:
            continue

        group = "General"
        if isinstance(func.value, ast.Name):
            group = groups.get(func.value.id, "General")

        help_text = _kw(node, "help", "") or ""
        action = _kw(node, "action")
        if not help_text and action == "version":
            help_text = "Show program's version number and exit"

        options.append(
            CliOption(
                flags=flags,
                dest=flags[0].lstrip("-").replace("-", "_"),
                help=help_text,
                group=group,
                choices=_kw(node, "choices"),
                default=_kw(node, "default"),
                type=_kw(node, "type"),
                action=action,
            )
        )

    order = ["General", "CLI Options", "Performance Options"]
    grouped: Dict[str, List[CliOption]] = {}
    for option in options:
        grouped.setdefault(option.group, []).append(option)
    ordered = {name: grouped[name] for name in order if name in grouped}
    for name, items in grouped.items():
        if name not in ordered:
            ordered[name] = items

    return {
        "options": options,
        "groups": ordered,
        "prog": "python main.py",
    }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    module: str
    class_name: str
    name: str
    backend_type: str = ""
    kind: str = "runtime"
    min_vram_gb: float = 0
    supports_quantization: List[str] = field(default_factory=list)
    speed_multiplier: float = 1.0
    quality_rating: int = 3
    requires_compilation: bool = False
    windows_supported: bool = True
    linux_supported: bool = True
    target_sample_rate: Optional[int] = None
    supports_synthesis: bool = True
    stub_fallback: bool = False
    status: str = "interface"
    summary: str = ""

    @property
    def vram_label(self) -> str:
        if self.min_vram_gb <= 0:
            return "CPU capable"
        return f"{self.min_vram_gb:g} GB min"


def _parse_enum(tree: ast.Module, class_name: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.Assign) and len(item.targets) == 1:
                    target = item.targets[0]
                    if isinstance(target, ast.Name) and isinstance(item.value, ast.Constant):
                        values[target.id] = str(item.value.value)
    return values


def _parse_registry() -> Dict[str, str]:
    """Map backend class name -> registered BackendType value."""
    path = ROOT / "core" / "backend_registry.py"
    tree = parse_source(path)
    enum_values = _parse_enum(parse_source(ROOT / "core" / "backend_base.py"), "BackendType")
    mapping: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "register"):
            continue
        if len(node.args) < 2:
            continue
        type_node, class_node = node.args[0], node.args[1]
        if isinstance(type_node, ast.Attribute) and isinstance(class_node, ast.Name):
            mapping[class_node.id] = enum_values.get(type_node.attr, type_node.attr.lower())
    return mapping


def _has_not_implemented(class_node: ast.ClassDef) -> bool:
    for item in class_node.body:
        if not isinstance(item, ast.FunctionDef) or item.name != "synthesize":
            continue
        for child in ast.walk(item):
            if isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call):
                func = child.exc.func
                if isinstance(func, ast.Name) and func.id == "NotImplementedError":
                    return True
            if isinstance(child, ast.Name) and child.id == "NotImplementedError":
                return True
    return False


def _target_sample_rate(class_node: ast.ClassDef) -> Optional[int]:
    for child in ast.walk(class_node):
        if isinstance(child, ast.Assign) and len(child.targets) == 1:
            target = child.targets[0]
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "target_sample_rate"
                and isinstance(child.value, ast.Constant)
            ):
                return int(child.value.value)
    return None


def parse_backends() -> List[Backend]:
    registry = _parse_registry()
    backends: List[Backend] = []

    for path in sorted((ROOT / "core").glob("backend_*.py")):
        if path.stem in {"backend_base", "backend_registry"}:
            continue
        tree = parse_source(path)
        source = path.read_text(encoding="utf-8")
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            caps_call = None
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and item.targets[0].id == "CAPABILITIES"
                    and isinstance(item.value, ast.Call)
                    and isinstance(item.value.func, ast.Name)
                    and item.value.func.id == "BackendCapabilities"
                ):
                    caps_call = item.value
                    break
            if caps_call is None:
                continue

            backend_type = registry.get(node.name, "")
            if not backend_type:
                # Classes that are not registered in BackendRegistry are internal
                # helpers (for example the Fish Speech + vLLM subclass) and are
                # not selectable by the application.
                continue
            kind = (
                "model"
                if backend_type in {"pytorch", "gpt-sovits", "fish-speech"}
                else "runtime"
            )
            supports_synthesis = not _has_not_implemented(node)
            stub_fallback = "placeholder" in source.lower()

            if not supports_synthesis:
                status = "interface"
            elif path.stem == "backend_fish":
                status = "scaffold"
            else:
                status = "implemented"

            backend = Backend(
                module=path.stem,
                class_name=node.name,
                name=str(_kw(caps_call, "name", node.name)),
                backend_type=backend_type,
                kind=kind,
                min_vram_gb=float(_kw(caps_call, "min_vram_gb", 0) or 0),
                supports_quantization=list(_kw(caps_call, "supports_quantization", []) or []),
                speed_multiplier=float(_kw(caps_call, "speed_multiplier", 1) or 1),
                quality_rating=int(_kw(caps_call, "quality_rating", 3) or 3),
                requires_compilation=bool(_kw(caps_call, "requires_compilation", False)),
                windows_supported=bool(_kw(caps_call, "windows_supported", True)),
                linux_supported=bool(_kw(caps_call, "linux_supported", True)),
                target_sample_rate=_target_sample_rate(node),
                supports_synthesis=supports_synthesis,
                stub_fallback=stub_fallback,
                status=status,
                summary=_doc_first_line(node),
            )
            backends.append(backend)

    return backends


# ---------------------------------------------------------------------------
# Presets & repo statistics
# ---------------------------------------------------------------------------


def parse_presets() -> List[Dict[str, Any]]:
    tree = parse_source(ROOT / "core" / "presets.py")
    presets: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target
            value = node.value
        else:
            continue
        if not (isinstance(target, ast.Name) and target.id == "BUILTIN_PRESETS"):
            continue
        if not isinstance(value, ast.Dict):
            continue
        for entry in value.values:
            if not isinstance(entry, ast.Call):
                continue
            presets.append(
                {
                    "name": _kw(entry, "name", "Preset"),
                    "description": _kw(entry, "description", ""),
                    "target_vram_gb": _kw(entry, "target_vram_gb"),
                    "backend": _kw(entry, "backend"),
                    "quantization": _kw(entry, "quantization"),
                    "nfe_steps": _kw(entry, "nfe_steps"),
                    "expected_quality": _kw(entry, "expected_quality"),
                    "expected_rtf": _kw(entry, "expected_rtf"),
                }
            )
    return presets


def repo_stats() -> Dict[str, int]:
    roots = ["core", "ui", "utils", "scripts", "training_workspace", "tests"]
    files = [ROOT / "main.py", ROOT / "test_all.py", ROOT / "test_imports.py"]
    for directory in roots:
        base = ROOT / directory
        if base.is_dir():
            files.extend(
                p
                for p in base.rglob("*.py")
                if "__pycache__" not in p.parts and "F5-TTS" not in p.parts
            )
    total_lines = 0
    for path in files:
        if path.exists():
            total_lines += len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    return {
        "python_files": len(files),
        "python_lines": total_lines,
        "core_modules": len(list((ROOT / "core").glob("*.py"))),
        "ui_modules": len(list((ROOT / "ui").glob("*.py"))),
    }


def build_data() -> Dict[str, Any]:
    backends = parse_backends()
    model_backends = [b for b in backends if b.kind == "model"]
    runtimes = [b for b in backends if b.kind == "runtime"]
    return {
        "version": read_version(),
        "cli": parse_cli(),
        "backends": backends,
        "model_backends": model_backends,
        "runtimes": runtimes,
        "presets": parse_presets(),
        "stats": repo_stats(),
        "quantization_levels": [
            "fp32",
            "fp16",
            "bf16",
            "int8",
            "int4",
            "gguf-q8",
            "gguf-q4",
        ],
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import json

    print(json.dumps(build_data(), indent=2, ensure_ascii=False, default=vars))
