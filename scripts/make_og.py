#!/usr/bin/env python3
"""Render the 1200x630 Open Graph image used by the docs site.

Run this only when the OG artwork changes; the resulting PNG is committed at
``site/static/og.png`` and copied into ``docs/og.png`` by the site build.

    pip install playwright
    python -m playwright install chromium
    python scripts/make_og.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.repo_data import read_version  # noqa: E402
from scripts.site_config import SITE_URL, STATIC_DIR  # noqa: E402

WIDTH, HEIGHT = 1200, 630
BARS = [
    18, 42, 65, 34, 80, 54, 26, 72, 92, 60, 38, 84, 48, 22, 58, 76, 36, 68,
    44, 90, 52, 30, 64, 40, 74, 28, 56, 82, 46, 20, 62, 50, 70, 32, 86, 42,
    24, 66, 48, 80, 36, 54, 28, 72, 44, 60, 34, 50, 78, 40,
]


def html(version: str) -> str:
    bars = "".join(f'<i style="height:{height}%"></i>' for height in BARS)
    site_host = urlsplit(SITE_URL).netloc
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  * {{ box-sizing: border-box; margin: 0; }}
  body {{
    width: {WIDTH}px; height: {HEIGHT}px; overflow: hidden;
    background: #080b0f;
    background-image:
      radial-gradient(760px 420px at 88% -10%, rgba(255,154,60,.16), transparent 65%),
      linear-gradient(rgba(255,255,255,.022) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,.022) 1px, transparent 1px);
    background-size: auto, 72px 72px, 72px 72px;
    color: #edf2f5; font-family: "Segoe UI", Arial, sans-serif;
    padding: 64px 72px; position: relative;
  }}
  .top {{ display:flex; align-items:center; gap:14px; }}
  .mark {{ width:38px; height:38px; }}
  .mark rect {{ fill:#ffb454; }}
  .brand {{ font-size: 22px; font-weight: 650; letter-spacing:-.02em; }}
  .brand span {{ color:#8b9ba8; font-weight:400; }}
  .badge {{
    margin-left:auto; font-family: Consolas, monospace; font-size: 13px;
    color:#ffb454; border:1px solid rgba(255,180,84,.35); border-radius:20px;
    padding: 4px 12px; letter-spacing:.1em;
  }}
  h1 {{
    margin-top: 52px; font-size: 74px; line-height: 1.02; letter-spacing: -0.045em;
    font-weight: 700; max-width: 900px;
  }}
  h1 em {{ color:#ffb454; font-style: normal; }}
  .sub {{
    margin-top: 24px; font-size: 25px; color:#b7c4ce; max-width: 760px; line-height:1.45;
  }}
  .mono {{
    font-family: Consolas, monospace; font-size: 15px; color:#64737f;
    letter-spacing: .14em; text-transform: uppercase;
  }}
  .wave {{
    position:absolute; left:72px; right:72px; bottom:96px; height:92px;
    display:flex; align-items:center; gap:4px;
    border-top:1px solid #1a2631; border-bottom:1px solid #1a2631;
    padding: 8px 0;
  }}
  .wave i {{
    flex:1; border-radius:1px; min-width:3px;
    background: linear-gradient(180deg, #ffb454, rgba(255,154,60,.22));
  }}
  .bottom {{
    position:absolute; left:72px; right:72px; bottom:42px;
    display:flex; justify-content:space-between; align-items:center;
  }}
  .url {{ color:#ffb454; font-family: Consolas, monospace; font-size:16px; letter-spacing:.05em; }}
</style></head>
<body>
  <div class="top">
    <svg class="mark" viewBox="0 0 32 32"><rect x="1" y="12" width="2.4" height="8" rx="1.2"/><rect x="5.9" y="8" width="2.4" height="16" rx="1.2"/><rect x="10.8" y="4" width="2.4" height="24" rx="1.2"/><rect x="15.7" y="9.5" width="2.4" height="13" rx="1.2"/><rect x="20.6" y="5.5" width="2.4" height="21" rx="1.2"/><rect x="25.5" y="11" width="2.4" height="10" rx="1.2"/><rect x="30" y="13.5" width="1.4" height="5" rx=".7"/></svg>
    <div class="brand">VoiceSynth<span> Studio</span></div>
    <div class="badge">LOCAL · v{version}</div>
  </div>
  <h1>Local text-to-speech,<br><em>without the cloud.</em></h1>
  <p class="sub">F5-TTS, GPT-SoVITS and Fish Speech behind one auditable local studio — CLI, PyQt6 GUI, voice profiles and batch synthesis.</p>
  <div class="wave">{bars}</div>
  <div class="bottom">
    <span class="mono">No API keys · No telemetry · MIT</span>
    <span class="url">{site_host}</span>
  </div>
</body></html>"""


def main() -> int:
    from playwright.sync_api import sync_playwright

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    target = STATIC_DIR / "og.png"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
        page.set_content(html(read_version()), wait_until="load")
        page.screenshot(path=str(target), clip={"x": 0, "y": 0, "width": WIDTH, "height": HEIGHT})
        browser.close()
    print(f"Wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
