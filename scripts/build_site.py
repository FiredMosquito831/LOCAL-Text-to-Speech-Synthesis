#!/usr/bin/env python3
"""Build the static documentation site into ``docs/``.

Usage:
    python scripts/build_site.py            # build docs/
    python scripts/build_site.py --check    # verify docs/ is current + valid

The build is deterministic and offline: data comes from the repository source,
templates live in ``site/templates`` and static files in ``site/static``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: E402

from scripts.repo_data import build_data  # noqa: E402
from scripts.site_config import (  # noqa: E402
    DOCS_DIR,
    REPO_URL,
    SITE_DESCRIPTION,
    SITE_DIR,
    SITE_NAME,
    SITE_TAGLINE,
    SITE_URL,
    STATIC_DIR,
    TEMPLATES_DIR,
)

PAGES: List[Dict[str, str]] = [
    {
        "slug": "index",
        "url": "/",
        "template": "home.html",
        "nav": "overview",
        "title": "VoiceSynth Studio — Local text-to-speech, without the cloud",
        "description": SITE_DESCRIPTION,
    },
    {
        "slug": "quickstart",
        "url": "/quickstart.html",
        "template": "quickstart.html",
        "nav": "quickstart",
        "title": "Install and run — VoiceSynth Studio",
        "description": (
            "Install VoiceSynth Studio, add an F5-TTS environment, fetch model weights "
            "and synthesize your first local WAV file."
        ),
    },
    {
        "slug": "usage",
        "url": "/usage.html",
        "template": "usage.html",
        "nav": "usage",
        "title": "CLI reference and studio guide — VoiceSynth Studio",
        "description": (
            "Every command-line flag, generated from main.py, plus the PyQt6 studio "
            "workflow for synthesis, cloning, profiles and batch jobs."
        ),
    },
    {
        "slug": "backends",
        "url": "/backends.html",
        "template": "backends.html",
        "nav": "backends",
        "title": "Backends and capability matrix — VoiceSynth Studio",
        "description": (
            "Model backends and runtime targets with the real implementation status, "
            "VRAM floors, precision support and platform notes read from the source."
        ),
    },
    {
        "slug": "performance",
        "url": "/performance.html",
        "template": "performance.html",
        "nav": "performance",
        "title": "Performance, memory and limits — VoiceSynth Studio",
        "description": (
            "What the code declares about memory, precision, sample rates, long-text "
            "chunking and the honest limits of the current release."
        ),
    },
    {
        "slug": "faq",
        "url": "/faq.html",
        "template": "faq.html",
        "nav": "faq",
        "title": "FAQ — VoiceSynth Studio",
        "description": (
            "Answers about model weights, backends, audio quality, GPU support, "
            "voice cloning and why some accelerators are interfaces only."
        ),
    },
    {
        "slug": "about",
        "url": "/about.html",
        "template": "about.html",
        "nav": "about",
        "title": "About, security and license — VoiceSynth Studio",
        "description": (
            "How VoiceSynth Studio handles your text and audio, what leaves your "
            "machine (nothing by default), and how the project is licensed."
        ),
    },
]

NOT_FOUND = {
    "slug": "404",
    "url": "/404.html",
    "template": "404.html",
    "nav": "",
    "title": "Signal lost — VoiceSynth Studio",
    "description": "The requested page was not found.",
}


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append(value)


def json_ld(data: Dict, page: Dict, canonical: str) -> str:
    payload = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": SITE_NAME,
        "applicationCategory": "MultimediaApplication",
        "applicationSubCategory": "Text-to-speech synthesis",
        "operatingSystem": "Windows, Linux, macOS",
        "softwareVersion": data["version"],
        "description": SITE_DESCRIPTION,
        "url": canonical,
        "downloadUrl": REPO_URL,
        "codeRepository": REPO_URL,
        "license": f"{REPO_URL}/blob/master/LICENSE",
        "isAccessibleForFree": True,
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
        "featureList": [
            "Local text-to-speech synthesis",
            "F5-TTS, GPT-SoVITS and Fish Speech backend integrations",
            "Zero-shot voice cloning from reference audio",
            "Batch synthesis queue",
            "PyQt6 studio interface and command-line mode",
        ],
    }
    if page["url"] != "/":
        payload["mainEntityOfPage"] = canonical
    return json.dumps(payload, ensure_ascii=False, indent=2)


def make_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals.update(
        site_url=SITE_URL,
        site_host=urlsplit(SITE_URL).netloc,
        site_name=SITE_NAME,
        site_tagline=SITE_TAGLINE,
        repo_url=REPO_URL,
        current_version="",
    )
    return env


def render_site(output: Path) -> Dict[str, str]:
    data = build_data()
    env = make_env()
    env.globals["current_version"] = data["version"]  # type: ignore[index]

    output.mkdir(parents=True, exist_ok=True)
    rendered: Dict[str, str] = {}

    for page in PAGES + [NOT_FOUND]:
        canonical = SITE_URL + page["url"]
        html = env.get_template(page["template"]).render(
            data=data,
            page=page,
            canonical=canonical,
            json_ld=json_ld(data, page, canonical),
            noindex=page["slug"] == "404",
            active=page["nav"],
        )
        name = "404.html" if page["slug"] == "404" else (
            "index.html" if page["slug"] == "index" else f'{page["slug"]}.html'
        )
        (output / name).write_text(html, encoding="utf-8", newline="\n")
        rendered[name] = html

    # Assets
    assets = output / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    shutil.copytree(STATIC_DIR, assets)

    # Open Graph image is a committed 1200x630 PNG.
    og_source = STATIC_DIR / "og.png"
    if og_source.exists():
        shutil.copyfile(og_source, output / "og.png")
    else:
        raise FileNotFoundError(
            "site/static/og.png is required (1200x630). "
            "Run: python scripts/make_og.py"
        )

    # CNAME / robots / sitemap
    site_host = urlsplit(SITE_URL).netloc
    (output / "CNAME").write_text(f"{site_host}\n", encoding="utf-8", newline="\n")
    (output / ".nojekyll").write_text("", encoding="utf-8", newline="\n")
    (output / "robots.txt").write_text(
        "User-agent: *\nAllow: /\n\n"
        f"Sitemap: {SITE_URL}/sitemap.xml\n",
        encoding="utf-8",
        newline="\n",
    )

    sitemap_entries = "".join(
        f"  <url><loc>{SITE_URL}{page['url']}</loc></url>\n" for page in PAGES
    )
    sitemap = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{sitemap_entries}"
        "</urlset>\n"
    )
    (output / "sitemap.xml").write_text(sitemap, encoding="utf-8", newline="\n")

    return rendered


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _png_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return width, height


def validate(output: Path, rendered: Dict[str, str]) -> List[str]:
    problems: List[str] = []
    required = [
        "index.html",
        "quickstart.html",
        "usage.html",
        "backends.html",
        "performance.html",
        "faq.html",
        "about.html",
        "404.html",
        "sitemap.xml",
        "robots.txt",
        "CNAME",
        ".nojekyll",
        "og.png",
        "assets/style.css",
    ]
    for name in required:
        if not (output / name).is_file():
            problems.append(f"missing required file: {name}")

    og = output / "og.png"
    if og.is_file():
        try:
            width, height = _png_size(og)
            if (width, height) != (1200, 630):
                problems.append(f"og.png must be 1200x630, got {width}x{height}")
        except ValueError as exc:
            problems.append(str(exc))

    # Internal link check (only for built HTML files).
    for name, html in rendered.items():
        collector = LinkCollector()
        collector.feed(html)
        for link in collector.links:
            if link.startswith(("http://", "https://", "mailto:", "#", "data:")):
                continue
            target = link.split("#", 1)[0]
            if not target:
                continue
            if target.startswith("/"):
                # Root-relative links must map inside the docs root.
                relative = target.lstrip("/")
                if relative == "" or relative.endswith("/"):
                    relative = relative + "index.html"
                candidate = output / relative
            else:
                candidate = (output / name).parent / target
            try:
                candidate.resolve().relative_to(output.resolve())
            except ValueError:
                problems.append(f"{name}: link escapes site root: {link}")
                continue
            if not candidate.exists():
                problems.append(f"{name}: broken link: {link}")

    # Basic SEO assertions.
    for name, html in rendered.items():
        if name == "404.html":
            continue
        if f'<link rel="canonical" href="{SITE_URL}' not in html:
            problems.append(f"{name}: missing canonical URL")
        if 'property="og:image"' not in html:
            problems.append(f"{name}: missing og:image")
        if "application/ld+json" not in html:
            problems.append(f"{name}: missing JSON-LD")

    return problems


def tree_hashes(root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def check() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "docs"
        rendered = render_site(output)
        problems = validate(output, rendered)
        fresh = tree_hashes(output)
        current = tree_hashes(DOCS_DIR) if DOCS_DIR.exists() else {}

        if fresh != current:
            problems.append(
                "docs/ is out of date with site/ — run `python scripts/build_site.py`"
            )
        if problems:
            print("Site check failed:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print(f"Site check passed: {len(fresh)} files, {len(rendered)} HTML pages.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the VoiceSynth Studio docs site")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that docs/ is current and internally valid (does not write docs/)",
    )
    args = parser.parse_args()

    if args.check:
        return check()

    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    rendered = render_site(DOCS_DIR)
    problems = validate(DOCS_DIR, rendered)
    if problems:
        print("Site built with problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    files = sum(1 for p in DOCS_DIR.rglob("*") if p.is_file())
    print(f"Built docs/ ({len(rendered)} pages, {files} files) at {SITE_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
