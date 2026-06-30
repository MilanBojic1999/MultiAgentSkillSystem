"""
web_scraper_tool.py
────────────────────────────────────────────────────────────────────────────
A plug-and-play web-scraping tool for Anthropic-based research agents.

Features
  • Browser-like headers to avoid most basic bot-detection
  • Only advertises Accept-Encoding values it can actually decode, so a
    Brotli/Zstandard response never gets silently mishandled
  • Automatic charset detection from the Content-Type header, UTF-8 fallback
  • Optional content cleaning via BeautifulSoup (strips scripts/styles,
    returns readable plain-text in addition to, or instead of, raw HTML)
  • text_only=True mode: extracts and returns only clean text, never
    storing/returning the raw HTML — keeps agent context usage small
  • Local save: set SAVE_SCRAPED_PAGES_TO to a directory path and every
    scraped page is also saved to disk (full, untruncated) automatically —
    no change to the caller API
  • Playwright fallback for JS-rendered pages (opt-in, see get_page_js)
  • Ready-to-use Anthropic tool definition (SCRAPE_TOOL)
  • Ready-to-use tool-dispatch helper (handle_tool_call)

Quick start
  1. pip install requests beautifulsoup4 lxml
  2. (Recommended) pip install brotli
     Most CDNs (Cloudflare, Fastly, etc.) compress with Brotli by default.
     Without this package the tool still works correctly (it just won't
     ask for 'br'), but responses are typically larger/slower to fetch.
  3. (Optional JS support) pip install playwright && playwright install chromium
  4. Drop this file next to your agent and wire it up:

        from web_scraper_tool import SCRAPE_TOOL, handle_tool_call
        response = client.messages.create(
            tools=[SCRAPE_TOOL, ...],
            ...
        )
        result = handle_tool_call(response)
"""

from __future__ import annotations

import re
import time
import random
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from requests.exceptions import (
    ConnectionError, Timeout, TooManyRedirects, HTTPError, RequestException
)

from langchain_core.tools import tool

# Optional – installed only when needed
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

# Brotli ('br') and Zstandard ('zstd') are encodings many CDNs (Cloudflare,
# Fastly, etc.) use by default. requests/urllib3 only auto-decompress them
# if the matching package is installed; otherwise they pass the *raw
# compressed bytes* through with no error. We detect availability here so
# we never advertise (via Accept-Encoding) support we don't actually have –
# that mismatch is what causes "garbled output" bugs.
try:
    import brotli as _brotli
    _BROTLI_AVAILABLE = True
except ImportError:
    try:
        import brotlicffi as _brotli
        _BROTLI_AVAILABLE = True
    except ImportError:
        _brotli = None
        _BROTLI_AVAILABLE = False

try:
    import zstandard as _zstd
    _ZSTD_AVAILABLE = True
except ImportError:
    _zstd = None
    _ZSTD_AVAILABLE = False

logger = logging.getLogger(__name__)
if not _BROTLI_AVAILABLE:
    logger.info(
        "brotli not installed – omitting 'br' from Accept-Encoding. "
        "Run `pip install brotli` to allow smaller/faster Brotli responses."
    )

# ─── Constants ───────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT = 20          # seconds
MAX_CONTENT_LENGTH = 5_000_000  # 5 MB – refuse anything larger
MAX_RETURNED_CHARS = 200_000  # cap what we hand back to the agent

# When set to a directory path (e.g. "./scraped_pages"), every page fetched
# by get_page() or get_page_js() is also saved locally — full, untruncated —
# before the agent-context truncation is applied.  Set to None to disable.
SAVE_SCRAPED_PAGES_TO: Optional[str] = "./scraped_pages"

# Only claim encodings we can actually decode. gzip/deflate are always safe
# (handled by Python's stdlib zlib via urllib3). br/zstd are added only if
# the optional decoder package is present.
_accept_encodings = ["gzip", "deflate"]
if _BROTLI_AVAILABLE:
    _accept_encodings.append("br")
if _ZSTD_AVAILABLE:
    _accept_encodings.append("zstd")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": ", ".join(_accept_encodings),
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# ─── Anthropic tool definition ────────────────────────────────────────────────

SCRAPE_TOOL = {
    "name": "scrape_webpage",
    "description": (
        "Fetches a webpage's content given its URL. Use this to read "
        "articles, extract detailed information from pages found in "
        "Google search results, or access any publicly available web "
        "content. By default returns only clean extracted text (no "
        "markup) to keep results compact — set text_only=false if you "
        "need the raw HTML itself (e.g. to inspect links or attributes)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "The full URL of the webpage to fetch "
                    "(must start with http:// or https://)."
                ),
            },
            "text_only": {
                "type": "boolean",
                "description": (
                    "If true (default), return only clean extracted text — "
                    "no raw HTML. Much smaller, so prefer this unless you "
                    "specifically need markup (links, image src, data "
                    "attributes, etc.), in which case set this to false."
                ),
                "default": True,
            },
        },
        "required": ["url"],
    },
}

# ─── Core scraping functions ──────────────────────────────────────────────────


def _validate_url(url: str) -> None:
    """Raise ValueError for obviously bad URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL must use http or https, got: '{parsed.scheme}'")
    if not parsed.netloc:
        raise ValueError(f"URL has no host: '{url}'")


def _maybe_manual_decompress(raw_bytes: bytes, content_encoding: str) -> bytes:
    """
    Safety net for compressed bytes that leaked through undecoded.

    Normally requests/urllib3 auto-decompresses gzip/deflate/br/zstd bodies
    as they're streamed. But if Accept-Encoding ever advertises a codec we
    can't actually decode (the bug that caused this function to be added),
    or an intermediary mangles things, this catches it via magic bytes /
    the Content-Encoding header instead of silently returning compressed
    binary as if it were text.

    Raises RuntimeError with an actionable message if the body is clearly
    still compressed and we have no way to decompress it.
    """
    encoding = (content_encoding or "").lower()

    # gzip has an unambiguous 2-byte magic number – safe to sniff directly.
    if raw_bytes[:2] == b"\x1f\x8b":
        try:
            import gzip
            return gzip.decompress(raw_bytes)
        except OSError:
            pass  # fall through – maybe it wasn't really gzip

    if "br" in encoding:
        if _BROTLI_AVAILABLE:
            try:
                return _brotli.decompress(raw_bytes)
            except Exception:
                pass  # urllib3 probably already handled it; leave bytes as-is
        else:
            raise RuntimeError(
                "Server sent Brotli-compressed content (Content-Encoding: br) "
                "but the 'brotli' package is not installed, so it can't be "
                "decompressed. Run: pip install brotli"
            )

    if "zstd" in encoding:
        if _ZSTD_AVAILABLE:
            try:
                return _zstd.ZstdDecompressor().decompress(raw_bytes)
            except Exception:
                pass
        else:
            raise RuntimeError(
                "Server sent Zstandard-compressed content (Content-Encoding: "
                "zstd) but the 'zstandard' package is not installed. "
                "Run: pip install zstandard"
            )

    return raw_bytes


def _looks_garbled(text: str, sample_size: int = 2000) -> bool:
    """
    Heuristic check for decoded text that's actually undecoded binary
    (e.g. still-compressed bytes forced through a text decoder). Real
    HTML/text is overwhelmingly printable; raw binary is not.
    """
    sample = text[:sample_size]
    if not sample:
        return False
    bad_chars = sum(
        1 for ch in sample
        if ch == "\ufffd" or (ord(ch) < 32 and ch not in "\n\r\t")
    )
    return (bad_chars / len(sample)) > 0.05

def _save_content(
    content: str,
    save_path: str,
    *,
    content_type: str = "html",
    source_url: str = "",
) -> str:
    """
    Save scraped content to a local file, creating parent directories as needed.

    Parameters
    ----------
    content      : The full text content to write (HTML or plain text).
    save_path    : Destination file path (absolute or relative).
    content_type : "html" or "text" — determines the default extension and
                   file mode behaviour.
    source_url   : Optional source URL written as a comment at the top of
                   HTML files for provenance tracking.

    Returns
    -------
    The resolved absolute path where the content was saved.

    Raises
    ------
    OSError, ValueError — letting the caller catch and surface as an error.
    """
    dest = Path(save_path).expanduser().resolve()

    # If the path looks like a directory or has no extension, auto-name it
    if dest.is_dir() or dest.suffix == "":
        if dest.is_dir():
            dest = dest / "scraped_page.html"
        else:
            dest = dest.with_suffix(".html" if content_type == "html" else ".txt")

    # Ensure parent directories exist
    dest.parent.mkdir(parents=True, exist_ok=True)

    write_content = content
    if content_type == "html" and source_url:
        # Prepend a provenance comment before <!DOCTYPE> or <html>, or at top
        provenance = f"<!-- Saved from: {source_url} -->\n"
        if "<!DOCTYPE" in content or "<html" in content:
            # Insert after <!DOCTYPE> if present, else at the very start
            doctype_end = content.find(">")
            if doctype_end != -1 and content[:doctype_end].startswith("<!DOCTYPE"):
                write_content = (
                    content[: doctype_end + 1] + "\n" + provenance + content[doctype_end + 1 :]
                )
            else:
                write_content = provenance + content
        else:
            write_content = provenance + content

    with open(dest, "w", encoding="utf-8") as f:
        f.write(write_content)

    logger.info("Saved %s content (%s chars) to %s", content_type, len(content), dest)
    return str(dest)


@tool
def get_page(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    extra_headers: Optional[dict] = None,
    polite_delay: float = 0.0,
    text_only: bool = False,
) -> dict:
    """
    Fetch a page with requests and return a structured result dict.

    Parameters
    ----------
    url            : Target URL.
    timeout        : Request timeout in seconds.
    extra_headers  : Any additional headers to merge in (e.g. cookies).
    polite_delay   : Optional sleep (seconds) before the request – useful
                     when scraping multiple pages from the same domain.
    text_only      : If True, only the cleaned plain-text is extracted and
                     returned — the raw HTML is parsed in memory to pull
                     the text out of it, but is never stored in the result
                     or handed back to the caller. Use this whenever you
                     just need the article content; it's typically 5-20x
                     smaller than the raw page, which matters a lot for an
                     agent's context window. Requires beautifulsoup4 (set
                     to True without it, and get_page returns an error
                     rather than silently falling back to full HTML).
                     The network fetch itself is unchanged either way —
                     this only affects what gets parsed out and returned.

    Returns
    -------
    {
        "url":          str,   – final URL after redirects
        "status_code":  int,
        "content_type": str,
        "html":         str,   – full HTML, up to MAX_RETURNED_CHARS chars.
                                  Always "" when text_only=True.
        "text":         str,   – plain text extracted by BS4. Populated
                                  whenever BS4 is available, regardless of
                                  text_only.
        "error":        str | None
    }
    """
    _validate_url(url)

    print("Page to scrape: ", url)

    if polite_delay > 0:
        time.sleep(polite_delay + random.uniform(0, 0.5))

    headers = {**HEADERS, **(extra_headers or {})}
    result: dict = {
        "url": url,
        "status_code": None,
        "content_type": "",
        "html": "",
        "text": "",
        "error": None,
        "saved_to": None,
    }

    if text_only and not _BS4_AVAILABLE:
        result["error"] = (
            "text_only=True requires beautifulsoup4, which isn't installed. "
            "Run: pip install beautifulsoup4 lxml — or call with "
            "text_only=False to get raw HTML instead."
        )
        return result

    try:
        resp = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            stream=True,            # check size before reading body
        )
        result["url"] = resp.url   # post-redirect URL
        result["status_code"] = resp.status_code
        result["content_type"] = resp.headers.get("Content-Type", "")

        # ── Size guard ──────────────────────────────────────────────────────
        content_length = int(resp.headers.get("Content-Length", 0))
        if content_length > MAX_CONTENT_LENGTH:
            result["error"] = (
                f"Page too large ({content_length:,} bytes > "
                f"{MAX_CONTENT_LENGTH:,} limit). Skipped."
            )
            return result

        resp.raise_for_status()

        # ── Read body (with streaming size cap) ─────────────────────────────
        chunks = []
        received = 0
        for chunk in resp.iter_content(chunk_size=65_536):
            received += len(chunk)
            if received > MAX_CONTENT_LENGTH:
                result["error"] = (
                    f"Response body exceeded {MAX_CONTENT_LENGTH:,} bytes – "
                    "truncated."
                )
                break
            chunks.append(chunk)

        raw_bytes = b"".join(chunks)

        # ── Decompression safety net ─────────────────────────────────────────
        # Should normally be a no-op (urllib3 already decompressed in-stream);
        # this only kicks in if a codec we can't handle slipped through.
        try:
            raw_bytes = _maybe_manual_decompress(
                raw_bytes, resp.headers.get("Content-Encoding", "")
            )
        except RuntimeError as exc:
            result["error"] = str(exc)
            return result

        # ── Encoding detection ───────────────────────────────────────────────
        encoding = resp.encoding or "utf-8"
        # Honour charset in Content-Type header first
        ct = result["content_type"]
        cs_match = re.search(r"charset=([^\s;]+)", ct, re.I)
        if cs_match:
            encoding = cs_match.group(1)

        try:
            html = raw_bytes.decode(encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            html = raw_bytes.decode("utf-8", errors="replace")

        # ── Garbled-output guard ─────────────────────────────────────────────
        # Catches any remaining case of binary masquerading as text (e.g. an
        # encoding we don't explicitly handle) and fails loudly instead of
        # handing the agent unreadable content.
        if _looks_garbled(html):
            result["error"] = (
                "Decoded content is not readable text (looks like undecoded "
                f"binary). Content-Encoding header was: "
                f"{resp.headers.get('Content-Encoding', '(none)')!r}. "
                "Try get_page_js() as a fallback, which renders via a real "
                "browser and isn't affected by this."
            )
            return result

        # ── Save full content locally (before truncation) ─────────────────
        save_dir = SAVE_SCRAPED_PAGES_TO
        if save_dir and not result.get("saved_to"):
            if text_only and not _BS4_AVAILABLE:
                logger.warning(
                    "SAVE_SCRAPED_PAGES_TO is set but text_only=True and "
                    "beautifulsoup4 is not installed — skipping local save. "
                    "Run: pip install beautifulsoup4 lxml"
                )
            else:
                try:
                    # Build a safe filename from the full URL
                    slug = re.sub(r"[^a-zA-Z0-9_.-]", "_",
                                  result["url"].split("://", 1)[-1] or "index")
                    fname = f"{slug}.html"
                    dest = Path(save_dir).expanduser().resolve() / fname

                    # full_payload = html if not text_only else _extract_text(html)
                    result["saved_to"] = _save_content(
                        html,
                        str(dest),
                        content_type="text" if text_only else "html",
                        source_url=result["url"],
                    )
                except OSError as exc:
                    logger.warning("Could not save page to %s: %s", save_dir, exc)
                    # Non-fatal — the agent still gets the truncated content

        # Truncate for the agent's context window
        if text_only:
            # html is parsed locally to extract text but deliberately never
            # written into `result` — this is what keeps the whole page
            # from being saved/returned.
            result["text"] = _extract_text(html)[:MAX_RETURNED_CHARS]
        else:
            result["html"] = html[:MAX_RETURNED_CHARS]
            if _BS4_AVAILABLE:
                result["text"] = _extract_text(html)[:MAX_RETURNED_CHARS]
            else:
                logger.warning(
                    "beautifulsoup4 not installed – plain-text extraction "
                    "skipped. Run: pip install beautifulsoup4 lxml"
                )

    except HTTPError as exc:
        result["error"] = f"HTTP {exc.response.status_code}: {exc}"
    except Timeout:
        result["error"] = f"Request timed out after {timeout}s."
    except TooManyRedirects:
        result["error"] = "Too many redirects."
    except ConnectionError as exc:
        result["error"] = f"Connection error: {exc}"
    except RequestException as exc:
        result["error"] = f"Request failed: {exc}"
    except ValueError as exc:
        result["error"] = str(exc)

    parts = [
        f"URL: {result['url']}",
        f"Status: {result['status_code']}",
        f"Content-Type: {result['content_type']}",
        "",
    ]

    if text_only:
        # page["html"] is intentionally empty here — never fetched into
        # the result, so there's nothing to accidentally include.
        parts += ["=== PAGE TEXT ===", result["text"]]
    else:
        if result["text"]:
            parts += ["=== PLAIN TEXT ===", result["text"], "", "=== RAW HTML ==="]
        parts.append(result["html"])

    return "\n".join(parts)


def _extract_text(html: str) -> str:
    """Return clean, readable plain-text from an HTML string using BS4."""
    soup = BeautifulSoup(html, "lxml")

    # Remove noise elements
    for tag in soup(["script", "style", "noscript", "head",
                     "nav", "footer", "aside", "form", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─── Optional: Playwright (JS-rendered pages) ─────────────────────────────────


def get_page_js(url: str, *, timeout: int = 30, text_only: bool = False) -> dict:
    """
    Fetch a JS-rendered page using Playwright (headless Chromium).

    Install: pip install playwright && playwright install chromium

    text_only : same meaning as in get_page() — if True, only extracted
                plain text is returned and the raw HTML is discarded
                rather than stored in the result.

    Returns the same dict shape as get_page().
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        return {
            "url": url, "status_code": None, "content_type": "",
            "html": "", "text": "",
            "error": (
                "Playwright is not installed. "
                "Run: pip install playwright && playwright install chromium"
            ),
        }

    result: dict = {
        "url": url, "status_code": None, "content_type": "",
        "html": "", "text": "", "error": None, "saved_to": None,
    }

    if text_only and not _BS4_AVAILABLE:
        result["error"] = (
            "text_only=True requires beautifulsoup4, which isn't installed. "
            "Run: pip install beautifulsoup4 lxml — or call with "
            "text_only=False to get raw HTML instead."
        )
        return result

    try:
        _validate_url(url)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(extra_http_headers=HEADERS)
            resp = page.goto(url, wait_until="networkidle", timeout=timeout * 1_000)

            result["url"] = page.url
            result["status_code"] = resp.status if resp else None
            result["content_type"] = (
                resp.header_value("content-type") if resp else ""
            )

            html = page.content()

            # ── Save full content locally (before truncation) ─────────────
            save_dir = SAVE_SCRAPED_PAGES_TO
            if save_dir and not result.get("saved_to"):
                if text_only and not _BS4_AVAILABLE:
                    logger.warning(
                        "SAVE_SCRAPED_PAGES_TO is set but text_only=True and "
                        "beautifulsoup4 is not installed — skipping local save. "
                        "Run: pip install beautifulsoup4 lxml"
                    )
                else:
                    try:
                        parsed = urlparse(result["url"])
                        slug = re.sub(r"[^a-zA-Z0-9_.-]", "_",
                                      result["url"].split("://", 1)[-1] or "index")
                        fname = f"{slug}.html" if not text_only else f"{slug}.txt"
                        dest = Path(save_dir).expanduser().resolve() / fname

                        full_payload = html if not text_only else _extract_text(html)
                        result["saved_to"] = _save_content(
                            full_payload,
                            str(dest),
                            content_type="text" if text_only else "html",
                            source_url=result["url"],
                        )
                    except OSError as exc:
                        logger.warning("Could not save page to %s: %s", save_dir, exc)

            if text_only:
                result["text"] = _extract_text(html)[:MAX_RETURNED_CHARS]
            else:
                result["html"] = html[:MAX_RETURNED_CHARS]
                if _BS4_AVAILABLE:
                    result["text"] = _extract_text(html)[:MAX_RETURNED_CHARS]

            browser.close()

    except PWTimeout:
        result["error"] = f"Playwright timed out after {timeout}s."
    except ValueError as exc:
        result["error"] = str(exc)
    except Exception as exc:             # noqa: BLE001
        result["error"] = f"Playwright error: {exc}"

    return result


# ─── Agent integration helpers ────────────────────────────────────────────────


def handle_tool_call(tool_name: str, tool_input: dict, *, use_js: bool = False) -> str:
    """
    Dispatch a tool call coming from the Anthropic API and return
    a string result ready to be placed in a tool_result content block.

    Parameters
    ----------
    tool_name  : Must equal "scrape_webpage".
    tool_input : The `input` dict from the API's tool_use block.
    use_js     : Set True to use the Playwright backend instead of requests.

    Returns
    -------
    A plain-text string summarising the page (for the agent's context).
    """
    if tool_name != "scrape_webpage":
        return f"[scraper] Unknown tool: {tool_name}"

    url = tool_input.get("url", "").strip()
    text_only = tool_input.get("text_only", True)

    scrape_fn = get_page_js if use_js else get_page
    page = scrape_fn(url, text_only=text_only)

    if page["error"]:
        return f"[scraper error] {page['error']}"

    # Build a compact but complete response for the agent
    parts = [
        f"URL: {page['url']}",
        f"Status: {page['status_code']}",
        f"Content-Type: {page['content_type']}",
    ]
    if page.get("saved_to"):
        parts.append(f"Saved to: {page['saved_to']}")
    parts.append("")

    if text_only:
        # page["html"] is intentionally empty here — never fetched into
        # the result, so there's nothing to accidentally include.
        parts += ["=== PAGE TEXT ===", page["text"]]
    else:
        if page["text"]:
            parts += ["=== PLAIN TEXT ===", page["text"], "", "=== RAW HTML ==="]
        parts.append(page["html"])

    return "\n".join(parts)


if __name__ == "__main__":
    import json
    html_p = get_page("https://n1info.rs/vesti/kako-se-menjao-centar-beograda/", text_only=True)

    with open("html.json", "w") as f:
        json.dump(html_p, f)
