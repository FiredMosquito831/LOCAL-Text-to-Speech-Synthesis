"""The generated site is valid, deterministic and deployable as-is."""

from __future__ import annotations

import json
import re

import pytest

jinja2 = pytest.importorskip("jinja2")

from scripts.build_site import render_site, validate  # noqa: E402
from scripts.site_config import SITE_URL  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    output = tmp_path_factory.mktemp("docs")
    rendered = render_site(output)
    return output, rendered


def test_site_url_constant_is_single_source():
    assert SITE_URL == "https://tts.danubelabs.net"


def test_required_pages_and_assets_exist(built):
    output, rendered = built
    assert len(rendered) == 8
    for name in (
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
        "og.png",
        "assets/style.css",
    ):
        assert (output / name).is_file(), name


def test_generated_site_has_no_broken_internal_links(built):
    output, rendered = built
    assert validate(output, rendered) == []


def test_canonical_and_og_tags(built):
    output, _ = built
    home = (output / "index.html").read_text(encoding="utf-8")
    assert f'<link rel="canonical" href="{SITE_URL}/">' in home
    assert f'<meta property="og:image" content="{SITE_URL}/og.png">' in home
    assert "summary_large_image" in home
    assert '"@type": "SoftwareApplication"' in home


def test_json_ld_is_valid_json(built):
    output, _ = built
    html = (output / "backends.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>', html, re.S
    )
    assert match
    payload = json.loads(match.group(1))
    assert payload["name"] == "VoiceSynth Studio"
    assert payload["softwareVersion"] == "0.1.1"


def test_cname_robots_and_sitemap(built):
    output, _ = built
    assert (output / "CNAME").read_text(encoding="utf-8").strip() == "tts.danubelabs.net"
    robots = (output / "robots.txt").read_text(encoding="utf-8")
    assert f"Sitemap: {SITE_URL}/sitemap.xml" in robots
    sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
    for page in ("quickstart.html", "usage.html", "backends.html", "faq.html", "about.html"):
        assert f"{SITE_URL}/{page}" in sitemap
    assert "404.html" not in sitemap


def test_og_image_is_1200x630(built):
    from scripts.build_site import _png_size

    output, _ = built
    assert _png_size(output / "og.png") == (1200, 630)


def test_build_is_deterministic(built, tmp_path):
    from scripts.build_site import tree_hashes

    output, _ = built
    render_site(tmp_path / "second")
    assert tree_hashes(output) == tree_hashes(tmp_path / "second")


def test_404_page_is_noindex(built):
    output, _ = built
    html = (output / "404.html").read_text(encoding="utf-8")
    assert 'name="robots" content="noindex' in html
