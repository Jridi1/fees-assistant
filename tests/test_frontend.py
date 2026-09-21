import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from feesbot.api import STATIC_DIR, create_app, page_headers
from feesbot.settings import Settings

ROOT = Path(__file__).resolve().parent.parent
ORIGIN = "https://me.pages.dev"


class NoService:
    pass


def client(origins=(ORIGIN,)):
    settings = Settings(_env_file=None, groq_api_key="x", cors_origins=list(origins))
    return TestClient(create_app(service=NoService(), settings=settings))


def test_the_chat_page_is_served_at_the_root():
    response = client().get("/")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    assert "Fees assistant" in response.text


def test_page_security_headers_only_allow_our_own_scripts_and_the_portfolio_as_a_frame_parent():
    headers = client().get("/").headers
    csp = headers["content-security-policy"]
    assert "script-src 'self'" in csp and "connect-src 'self'" in csp and "default-src 'none'" in csp
    assert f"frame-ancestors 'self' {ORIGIN}" in csp
    assert headers["x-content-type-options"] == "nosniff" and headers["referrer-policy"] == "no-referrer"


def test_without_configured_origins_only_the_site_itself_may_frame_the_page():
    csp = client(origins=()).get("/").headers["content-security-policy"]
    assert csp.endswith("frame-ancestors 'self'")


def test_malformed_origins_cannot_inject_csp_directives():
    csp = page_headers(["https://ok.example", "https://x.example; script-src *", "javascript:alert(1)", "https://a b"])[
        "Content-Security-Policy"
    ]
    assert csp.endswith("frame-ancestors 'self' https://ok.example")
    assert "script-src *" not in csp


def test_every_asset_the_page_references_is_served():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assets = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
    assert {"/static/style.css", "/static/sse.js", "/static/markdown.js", "/static/app.js"} <= set(assets)
    c = client()
    for path in assets:
        assert c.get(path).status_code == 200, path


def test_the_page_works_under_its_own_content_security_policy():
    """The CSP forbids inline scripts, inline handlers and style attributes; the HTML must not use any."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)", html), "inline <script> would be blocked"
    assert not re.search(r"\son[a-z]+\s*=", html, re.I), "inline event handler would be blocked"
    assert not re.search(r"\sstyle\s*=", html, re.I), "inline style attribute would be blocked"
    assert "<style" not in html.lower()


def test_the_api_docs_do_not_list_the_page_route():
    schema = client().get("/openapi.json").json()
    assert "/" not in schema["paths"] and "/chat/stream" in schema["paths"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_javascript_unit_tests_pass():
    """Runs the Node tests for the page's Markdown renderer and SSE parser."""
    result = subprocess.run(
        ["node", "--test", "tests/js/markdown.test.js", "tests/js/sse.test.js"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
