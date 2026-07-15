"""Shared task-string construction for the pipeline's HTTP entry points.

``api_server.py`` (sync/async ``/run``) and ``streaming.py`` (``/run-stream``)
both need to turn a ``task`` + optional ``files`` request into the string
that seeds graph state. This module is the single place that shape is
defined, so the two paths can't drift apart again.

It's also the single place binary (base64) file content is turned into
text — see ``_extract_binary_text`` — so both entry points get the same
PDF/docx handling for free.
"""

import base64
import io
import os
from typing import Any

# Server-side safety cap — client should truncate first, but this is a backstop.
MAX_FILE_CHARS = 100_000

# Extensions we know how to extract text from. Dispatch is by extension for
# now; switching to magic-byte sniffing is a deferred follow-up.
SUPPORTED_BINARY_EXTENSIONS = {".pdf", ".docx"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a base64-encoded file's extension has no text extractor."""


class EmptyExtractedTextError(ValueError):
    """Raised when extraction succeeds but yields no usable text (e.g. a
    scanned PDF with no text layer — OCR is not supported)."""


def _extract_pdf_text(raw: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx_text(raw: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(raw))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_binary_text(filename: str, raw: bytes) -> str:
    """Decode a binary file's bytes into text, dispatching on extension.

    Raises ``UnsupportedFileTypeError`` for unknown extensions and
    ``EmptyExtractedTextError`` when extraction yields only whitespace.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        text = _extract_pdf_text(raw)
    elif ext == ".docx":
        text = _extract_docx_text(raw)
    else:
        supported = ", ".join(sorted(SUPPORTED_BINARY_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported binary file type '{ext or filename}' for '{filename}'. "
            f"Supported types: {supported}."
        )

    if not text.strip():
        raise EmptyExtractedTextError(
            f"'{filename}' has no extractable text — it may be a scanned "
            "document without a text layer (OCR is not supported)."
        )
    return text


def _truncate_file_content(content: str, max_chars: int = MAX_FILE_CHARS) -> str:
    """Truncate *content* to *max_chars*, keeping head + tail with a notice."""
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return (
        content[:half]
        + f"\n\n... [TRUNCATED — {len(content) - max_chars:,} chars omitted] ...\n\n"
        + content[-half:]
    )


def _file_attrs(f: Any) -> tuple[str, str, str | None]:
    """Read (filename, content, encoding) off either a FileInput model or a plain dict."""
    if hasattr(f, "filename"):
        return f.filename, f.content, f.encoding
    return f.get("filename", "unknown"), f.get("content", ""), f.get("encoding")


# Per-file cap when full document text is injected into a worker's prompt.
MAX_WORKER_FILE_CHARS = 20_000


def build_task_string(task: str, files: list[Any] | None) -> str:
    """Build the one canonical task string passed into graph state as ``task``.

    Layout: ``Query: <task>`` followed by a one-line list of attached
    filenames (not their content — full text lives in ``state["files"]`` and
    is routed to individual steps by the planner instead of being embedded
    in the task string), so the query itself always reads first.
    """
    task_string = f"Query: {task}"
    if files:
        names = ", ".join(_file_attrs(f)[0] for f in files)
        task_string += f"\n\nAttached files: {names}"
    return task_string


def build_files_state(files: list[Any] | None) -> dict[str, str]:
    """Build the filename -> text content mapping stored in ``AgentState.files``.

    Plain-text files are used as-is. Base64-encoded files are decoded and run
    through ``_extract_binary_text`` (pypdf / python-docx) before truncation —
    the 100k-char cap applies to the *extracted* text, not the raw bytes.
    """
    if not files:
        return {}
    state: dict[str, str] = {}
    for f in files:
        filename, content, encoding = _file_attrs(f)
        if encoding == "base64":
            raw = base64.b64decode(content)
            text = _extract_binary_text(filename, raw)
        else:
            text = content
        state[filename] = _truncate_file_content(text)
    return state


def render_files_block(files: dict[str, str], max_chars_per_file: int = MAX_WORKER_FILE_CHARS) -> str:
    """Render a filename -> content mapping as an '## Attached documents' Markdown block.

    Used both for worker prompt injection (agents/sub_agents_nodes.py) and the
    writer's empty-plan fallback (yotta_graph.py) — the two places raw
    document text is allowed to reach an agent directly.
    """
    if not files:
        return ""
    blocks = [
        f"### File: {filename}\n```\n{_truncate_file_content(content, max_chars_per_file)}\n```"
        for filename, content in files.items()
    ]
    return "## Attached documents\n\n" + "\n\n".join(blocks)
