#!/usr/bin/env python3
"""Capture real screenshots of the generated docs site.

Starts a local HTTP server for docs/ and uses Playwright Chromium. Screenshots
are written to site/static/screenshots/ and are copied into docs/assets by the
site build.

    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import functools
import http.server
import socket
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.site_config import DOCS_DIR, STATIC_DIR  # noqa: E402

SHOTS = [
    ("home-desktop", "/", 1440, 900),
    ("quickstart-desktop", "/quickstart.html", 1440, 900),
    ("backends-desktop", "/backends.html", 1440, 900),
    ("usage-desktop", "/usage.html", 1440, 900),
    ("home-mobile", "/", 390, 844),
    ("backends-mobile", "/backends.html", 390, 844),
]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> int:
    from playwright.sync_api import sync_playwright

    if not (DOCS_DIR / "index.html").exists():
        print("docs/ is empty — run `python scripts/build_site.py` first.")
        return 1

    target = STATIC_DIR / "screenshots"
    target.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(DOCS_DIR)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            for name, path, width, height in SHOTS:
                page = browser.new_page(
                    viewport={"width": width, "height": height},
                    device_scale_factor=1,
                    reduced_motion="reduce",
                )
                page.goto(base + path, wait_until="networkidle")
                page.screenshot(path=str(target / f"{name}.png"))
                page.close()
                print(f"captured {name}.png ({width}x{height})")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
