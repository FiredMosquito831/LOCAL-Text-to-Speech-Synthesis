"""Single source of truth for the generated documentation site.

Change ``SITE_URL`` here (and only here) when the production host changes.
"""

from pathlib import Path

# Public site URL (no trailing slash). Used for canonical links, Open Graph,
# JSON-LD, sitemap.xml and robots.txt.
SITE_URL = "https://tts.danubelabs.net"

SITE_NAME = "VoiceSynth Studio"
SITE_TAGLINE = "Local text-to-speech, without the cloud."
SITE_DESCRIPTION = (
    "A local-first text-to-speech workstation with a unified backend layer, "
    "voice cloning, batch synthesis and a PyQt6 studio interface."
)

REPO_OWNER = "FiredMosquito831"
REPO_NAME = "LOCAL-Text-to-Speech-Synthesis"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
ISSUES_URL = f"{REPO_URL}/issues"

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT / "site"
TEMPLATES_DIR = SITE_DIR / "templates"
STATIC_DIR = SITE_DIR / "static"
DOCS_DIR = ROOT / "docs"
