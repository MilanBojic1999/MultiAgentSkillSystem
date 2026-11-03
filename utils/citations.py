"""Source registry and citation resolution for the answer-writer contract.

``skills/answer-writer/SKILL.md`` asks the writer to cite with source IDs in
``{{ID}}`` form — never URLs, titles or file paths — and states that "the
system resolves IDs to the actual sources after you respond". This module
provides both halves of that contract:

- :func:`build_source_map` assigns stable 1-based IDs to every citable source
  the writer is given (yotta search URLs, rendered attached files, worker
  findings), in a deterministic order;
- :func:`render_source_registry` renders the registry block the writer sees;
- :func:`parse_source_registry` is its inverse — the streaming path recovers
  the map from the writer's prompt, the only place the registry travels;
- :func:`resolve_and_append` rewrites emitted markers into Markdown links /
  bare numbers and appends the cited-only ``### Sources`` list;
- :class:`StreamingCitationResolver` does the same translation incrementally
  on the ``/run-stream`` token stream, holding a partial marker across chunk
  boundaries.

Pure functions and data — no graph or LLM imports — so the graph node and
the streaming translator can share it without an import cycle. Code fences
and inline code spans are protected from resolution in the non-streaming
path; the streaming resolver cannot see future text, so markers inside code
there resolve too (documented divergence, kept deliberately simple).
"""

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from utils.json_utils import extract_json

_HTTP_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_MARKER_RE = re.compile(r"\{\{\s*(\d+)\s*\}\}")
_SEARCH_SOURCE_RE = re.compile(r"\[source:\s*([^\]]+?)\s*\]")
_REGISTRY_LINE_RE = re.compile(
    r"^[ \t]*[-*][ \t]*\{\{\s*(\d+)\s*\}\}[ \t]*[—–-][ \t]*(.+?)[ \t]*$",
    re.MULTILINE,
)
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"(`+)([^\n]*?)\1")

# Mirrors ``graphs/yotta_graph.py``'s unusable-output markers. Duplicated on
# purpose: importing the graph here would create a cycle.
_UNUSABLE_OUTPUT_PREFIXES = ("[STEP FAILED", "[SKIPPED")

_REGISTRY_HEADER = "## Sources for citation"
_REGISTRY_INSTRUCTION = (
    "Cite only the IDs listed below, in `{{ID}}` form (for example `{{1}}`). "
    "Never write a URL, title, or file path in a citation; the system resolves "
    "the IDs after you respond. Do not cite anything that is not listed here."
)


@dataclass(frozen=True)
class Source:
    """One citable source, identified by the ``{{id}}`` marker the writer emits."""

    id: int                     # 1-based, contiguous in registration order
    label: str                  # URL, filename, or short worker text
    url: str | None = None      # only http(s) sources are linkable
    origin: str = ""            # "search" | "file" | "worker"
    step: int | None = None     # worker step that reported it, if any


@dataclass(frozen=True)
class SourceMap:
    """Every source the writer may cite, plus diagnostics about what was left out.

    ``steps_without_sources`` names worker steps whose output could not be
    parsed into findings (non-JSON, or JSON without a ``findings`` list);
    ``skipped_findings`` counts individual findings whose source was empty or
    ``"none"``.
    """

    sources: tuple[Source, ...] = ()
    steps_without_sources: tuple[int, ...] = ()
    skipped_findings: int = 0


@dataclass(frozen=True)
class CitationResolution:
    """The result of resolving one writer output.

    ``text`` has every known marker rewritten (and unknown ones stripped);
    ``cited`` is the cited sources, ascending by id; ``unknown_ids`` are the
    distinct IDs the writer invented, in first-seen order; ``marker_count``
    counts every marker seen outside protected code, known or not.
    """

    text: str
    cited: tuple[Source, ...] = ()
    unknown_ids: tuple[int, ...] = ()
    marker_count: int = 0


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def _is_http_url(label: str) -> bool:
    return bool(_HTTP_URL_RE.match(label))


def _normalized_key(label: str) -> str:
    """Dedupe key: URLs ignore a trailing slash; plain labels are
    whitespace-folded and case-insensitive."""
    label = label.strip()
    if _is_http_url(label):
        return label.rstrip("/")
    return " ".join(label.split()).casefold()


def _match_attached_filename(source: str, filenames: tuple[str, ...]) -> str | None:
    """Return the attached filename a worker ``source`` string starts with.

    Document-reader workers report ``"<filename>, section/location, quote…"``
    (see skills/document-reader-worker), so the filename is a *prefix*; a
    source merely starting with the same characters must not match
    (``report.pdfish`` is not ``report.pdf``).
    """
    for name in sorted(filenames, key=len, reverse=True):
        if not name:
            continue
        if source == name:
            return name
        if source.startswith(name):
            rest = source[len(name):]
            if rest and not rest[0].isalnum():
                return name
    return None


def build_source_map(*, search_results: str = "",
                     rendered_files: Mapping[str, str] | None = None,
                     results: dict | None = None,
                     attached_filenames: Iterable[str] = ()) -> SourceMap:
    """Assign contiguous IDs to every citable source the writer was given.

    Deterministic order — first appearance wins, duplicates (by normalized
    URL or label) keep the first ID:

    1. ``[source: <url>]`` occurrences in the yotta search results, top to
       bottom;
    2. the files actually rendered into the writer's prompt (direct/instant
       route only), insertion order;
    3. worker findings — numeric step keys ascending, then ``findings``
       order. Each finding's ``source`` (or, failing that, a top-level
       ``source``) is matched to an attached filename first, then to an
       ``http(s)://`` URL, else kept as plain text. Failed/skipped step
       outputs contribute nothing; non-JSON outputs are reported in
       ``steps_without_sources`` rather than raising.
    """
    filenames = tuple(dict.fromkeys(
        name for name in attached_filenames if isinstance(name, str) and name
    ))
    sources: list[Source] = []
    seen: set[str] = set()
    skipped_findings = 0

    def add(label: str, *, origin: str, step: int | None = None) -> None:
        label = label.strip()
        if not label:
            return
        key = _normalized_key(label)
        if key in seen:
            return
        seen.add(key)
        sources.append(Source(
            id=len(sources) + 1,
            label=label,
            url=label if _is_http_url(label) else None,
            origin=origin,
            step=step,
        ))

    if isinstance(search_results, str) and search_results:
        for match in _SEARCH_SOURCE_RE.finditer(search_results):
            add(match.group(1), origin="search")

    if rendered_files:
        for filename in rendered_files:
            if isinstance(filename, str):
                add(filename, origin="file")

    steps_without_sources: list[int] = []
    results = results or {}
    for step_num in sorted(k for k in results if isinstance(k, int) and k >= 1):
        output = results[step_num]
        text = output if isinstance(output, str) else str(output)
        if text.startswith(_UNUSABLE_OUTPUT_PREFIXES):
            continue
        try:
            parsed = extract_json(text)
        except (ValueError, TypeError):
            steps_without_sources.append(step_num)
            continue
        findings = parsed.get("findings") if isinstance(parsed, dict) else None
        if not isinstance(findings, list):
            steps_without_sources.append(step_num)
            continue

        top_source = parsed.get("source")
        fallback = top_source if isinstance(top_source, str) else None
        for finding in findings:
            raw = finding.get("source") if isinstance(finding, dict) else None
            if not isinstance(raw, str):
                raw = fallback
            if not isinstance(raw, str):
                skipped_findings += 1
                continue
            source_text = raw.strip()
            if not source_text or source_text.casefold() == "none":
                skipped_findings += 1
                continue
            filename = _match_attached_filename(source_text, filenames)
            add(filename or source_text, origin="worker", step=step_num)

    return SourceMap(
        sources=tuple(sources),
        steps_without_sources=tuple(steps_without_sources),
        skipped_findings=skipped_findings,
    )


# ---------------------------------------------------------------------------
# Registry rendering / parsing (the writer prompt is the transport)
# ---------------------------------------------------------------------------

def render_source_registry(source_map: SourceMap) -> str:
    """Render the registry block injected into the writer's prompt."""
    if source_map is None or not source_map.sources:
        return ""
    lines = [_REGISTRY_HEADER, "", _REGISTRY_INSTRUCTION, ""]
    # Concatenation, not an f-string: the ``{{``/``}}`` are literal marker
    # delimiters, and four braces in an f-string would escape to two literal
    # braces instead of delimiting the ID.
    lines.extend(
        "- {{" + str(source.id) + "}} — " + source.label
        for source in source_map.sources
    )
    return "\n".join(lines)


def parse_source_registry(text: str) -> tuple[Source, ...]:
    """Recover the source map from any text containing a rendered registry.

    Used by the streaming translator on the writer's model input (the
    registry reaches the model only as part of its prompt). Dataclass fields
    the registry does not carry (``origin``/``step``) keep their defaults.
    """
    if not isinstance(text, str) or not text:
        return ()
    sources: list[Source] = []
    seen: set[int] = set()
    for match in _REGISTRY_LINE_RE.finditer(text):
        source_id = int(match.group(1))
        if source_id in seen:
            continue
        seen.add(source_id)
        label = match.group(2).strip()
        if not label:
            continue
        sources.append(Source(
            id=source_id,
            label=label,
            url=label if _is_http_url(label) else None,
        ))
    return tuple(sources)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class _CitationAccumulator:
    """Mutable bookkeeping shared by the full and streaming resolvers."""

    __slots__ = ("cited", "unknown", "markers")

    def __init__(self) -> None:
        self.cited: dict[int, Source] = {}
        self.unknown: list[int] = []
        self.markers = 0


def _format_marker(source: Source) -> str:
    """One marker as Markdown: a link for HTTP(S) sources, a bare number
    otherwise. URLs with whitespace or parentheses use the angle-bracket
    destination form so the link stays valid."""
    if not source.url:
        return f"[{source.id}]"
    if any(char.isspace() or char in "()" for char in source.url):
        return f"[{source.id}](<{source.url}>)"
    return f"[{source.id}]({source.url})"


def _strip_trailing_space(pieces: list[str]) -> None:
    """Drop the space an unknown marker leaves behind: ``Claim {{99}}.``
    becomes ``Claim.`` rather than ``Claim .``."""
    for index in range(len(pieces) - 1, -1, -1):
        piece = pieces[index]
        if not piece:
            continue
        pieces[index] = piece.rstrip(" \t")
        return


def _resolve_segment(text: str, by_id: dict[int, Source],
                     acc: _CitationAccumulator) -> str:
    """Resolve every complete marker in one unprotected text segment."""
    if "{{" not in text:
        return text
    pieces: list[str] = []
    last = 0
    for match in _MARKER_RE.finditer(text):
        pieces.append(text[last:match.start()])
        source_id = int(match.group(1))
        acc.markers += 1
        source = by_id.get(source_id)
        if source is None:
            _strip_trailing_space(pieces)
            if source_id not in acc.unknown:
                acc.unknown.append(source_id)
        else:
            pieces.append(_format_marker(source))
            acc.cited[source_id] = source
        last = match.end()
    pieces.append(text[last:])
    return "".join(pieces)


def _code_segments(text: str) -> list[tuple[bool, str]]:
    """Split *text* into ``(protected, segment)`` pieces, where protected
    pieces are fenced code blocks (``` / ~~~), which must stay verbatim."""
    segments: list[tuple[bool, str]] = []
    buffer: list[str] = []
    fence: tuple[str, int] | None = None

    def flush(protected: bool) -> None:
        if buffer:
            segments.append((protected, "".join(buffer)))
            buffer.clear()

    for line in text.splitlines(keepends=True):
        match = _FENCE_RE.match(line)
        if fence is None:
            if match:
                flush(False)
                fence = (match.group(1)[0], len(match.group(1)))
                buffer.append(line)
            else:
                buffer.append(line)
        else:
            buffer.append(line)
            if (match and match.group(1)[0] == fence[0]
                    and len(match.group(1)) >= fence[1]):
                flush(True)
                fence = None
    flush(fence is not None)
    return segments


def _resolve_text(text: str, by_id: dict[int, Source],
                  acc: _CitationAccumulator, *, protect_code: bool) -> str:
    if not protect_code:
        return _resolve_segment(text, by_id, acc)
    pieces: list[str] = []
    for protected, segment in _code_segments(text):
        if protected:
            pieces.append(segment)
            continue
        last = 0
        for match in _INLINE_CODE_RE.finditer(segment):
            if match.start() > last:
                pieces.append(
                    _resolve_segment(segment[last:match.start()], by_id, acc)
                )
            pieces.append(match.group(0))
            last = match.end()
        if last < len(segment):
            pieces.append(_resolve_segment(segment[last:], by_id, acc))
    return "".join(pieces)


def resolve_citations(text: str, sources: Iterable[Source] = ()) -> CitationResolution:
    """Rewrite ``{{ID}}`` markers in *text*; code fences/spans stay untouched."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    by_id = {source.id: source for source in sources}
    acc = _CitationAccumulator()
    resolved = _resolve_text(text, by_id, acc, protect_code=True)
    return CitationResolution(
        text=resolved,
        cited=tuple(sorted(acc.cited.values(), key=lambda source: source.id)),
        unknown_ids=tuple(acc.unknown),
        marker_count=acc.markers,
    )


def _format_sources_section(cited: Iterable[Source]) -> str:
    ordered = sorted(cited, key=lambda source: source.id)
    if not ordered:
        return ""
    items = "\n".join(
        f"{index}. {source.label}" for index, source in enumerate(ordered, start=1)
    )
    return f"### Sources\n\n{items}"


def append_sources_section(text: str, cited: Iterable[Source]) -> str:
    """Append a ``### Sources`` list of the cited sources (ascending, once
    each). No citations → *text* is returned byte-for-byte unchanged."""
    section = _format_sources_section(cited)
    if not section:
        return text
    body = text.rstrip()
    return f"{body}\n\n{section}" if body else section


def resolve_and_append(
    text: str, sources: Iterable[Source] = ()
) -> tuple[str, CitationResolution]:
    """Resolve markers, then append the sources section; returns both."""
    resolution = resolve_citations(text, sources)
    return append_sources_section(resolution.text, resolution.cited), resolution


# ---------------------------------------------------------------------------
# Streaming translation
# ---------------------------------------------------------------------------

def _partial_marker_start(text: str) -> int:
    """Index where a trailing, still-incomplete marker begins — or ``-1``.

    A marker split across chunks (``{{`` / ``1`` / ``}}``) must be held, not
    emitted early; anything that cannot grow into ``{{<digits>}}`` is not a
    partial marker.
    """
    double = text.rfind("{{")
    if double != -1 and re.fullmatch(r"\{\{\s*\d*\s*\}?", text[double:]):
        return double
    if text.endswith("{"):
        return len(text) - 1
    return -1


class StreamingCitationResolver:
    """Incremental ``{{ID}}`` translator for the writer's token stream.

    ``feed`` returns the text that is safe to emit, holding a trailing partial
    marker; ``flush`` releases whatever is still buffered. Cited sources
    accumulate until ``reset`` (a writer attempt restart) or the end of the
    answer, when ``sources_section`` supplies the appendix.
    """

    def __init__(self, sources: Iterable[Source] = ()):
        self._sources: tuple[Source, ...] = ()
        self._by_id: dict[int, Source] = {}
        self._buffer = ""
        self._acc = _CitationAccumulator()
        self.set_sources(sources)

    def set_sources(self, sources: Iterable[Source]) -> None:
        """Update the map; the buffer and the cited set are preserved."""
        self._sources = tuple(sources)
        self._by_id = {source.id: source for source in self._sources}

    @property
    def cited(self) -> tuple[Source, ...]:
        return tuple(sorted(self._acc.cited.values(), key=lambda source: source.id))

    @property
    def marker_count(self) -> int:
        return self._acc.markers

    def feed(self, text: str) -> str:
        """Consume a chunk; returns the resolved text safe to emit now."""
        if text is None:
            return ""
        self._buffer += text if isinstance(text, str) else str(text)
        cut = _partial_marker_start(self._buffer)
        if cut < 0:
            ready, self._buffer = self._buffer, ""
        else:
            ready, self._buffer = self._buffer[:cut], self._buffer[cut:]
        if not ready:
            return ""
        return _resolve_segment(ready, self._by_id, self._acc)

    def flush(self) -> str:
        """Emit everything still buffered (a dangling partial marker included)."""
        ready, self._buffer = self._buffer, ""
        if not ready:
            return ""
        return _resolve_segment(ready, self._by_id, self._acc)

    def sources_section(self) -> str:
        """The ``### Sources`` appendix for the answer stream, or ``""``."""
        section = _format_sources_section(self.cited)
        return f"\n\n{section}" if section else ""

    def reset(self) -> None:
        """Writer attempt restart: everything sent so far is a dead draft."""
        self._buffer = ""
        self._acc = _CitationAccumulator()
