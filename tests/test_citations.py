"""utils.citations — source registry, {{ID}} resolution, streaming translation.

Hermetic: pure data in, pure data out. No graph, no LLM, no network.
"""

import json

from utils.citations import (
    Source,
    StreamingCitationResolver,
    append_sources_section,
    build_source_map,
    parse_source_registry,
    render_source_registry,
    resolve_and_append,
    resolve_citations,
)

SEARCH = (
    "- Alpha is true [source: https://example.com/a]\n"
    "- Beta is false [source: https://example.com/b]"
)


def worker_json(*findings, result="sufficient"):
    """A worker output in the contract of skills/yotta-researcher."""
    return json.dumps({
        "subquery_id": "q",
        "result": result,
        "findings": [{"content": text, "source": source} for text, source in findings],
        "note": "",
    })


def sources(*pairs):
    """``(label, url)`` pairs -> ``Source`` objects with contiguous IDs."""
    return tuple(
        Source(id=index, label=label, url=url)
        for index, (label, url) in enumerate(pairs, start=1)
    )


URL_SOURCES = sources(
    ("https://example.com/a", "https://example.com/a"),
    ("notes.md", None),
)


# ---------------------------------------------------------------------------
# build_source_map — discovery and deterministic IDs
# ---------------------------------------------------------------------------

def test_search_sources_get_contiguous_ids_in_reading_order():
    source_map = build_source_map(search_results=SEARCH)
    assert [source.id for source in source_map.sources] == [1, 2]
    assert [source.label for source in source_map.sources] == [
        "https://example.com/a", "https://example.com/b"]
    assert all(source.origin == "search" for source in source_map.sources)
    assert all(source.url == source.label for source in source_map.sources)
    assert source_map.steps_without_sources == ()
    assert source_map.skipped_findings == 0


def test_search_urls_dedupe_ignoring_a_trailing_slash():
    text = "- a [source: https://example.com/a]\n- b [source: https://example.com/a/]"
    source_map = build_source_map(search_results=text)
    assert len(source_map.sources) == 1


def test_rendered_files_become_file_sources_in_insertion_order():
    source_map = build_source_map(
        rendered_files={"notes.md": "text", "report.pdf": "text"})
    assert [(s.id, s.label, s.url, s.origin) for s in source_map.sources] == [
        (1, "notes.md", None, "file"),
        (2, "report.pdf", None, "file"),
    ]


def test_unrendered_files_are_not_citable():
    """``attached_filenames`` only lets worker findings *match* a file; a file
    that was not rendered into the writer's prompt has no ID."""
    source_map = build_source_map(attached_filenames=("notes.md",))
    assert source_map.sources == ()


def test_worker_output_in_a_json_fence_is_parsed():
    output = "```json\n" + worker_json(("fact", "https://example.com/w")) + "\n```"
    source_map = build_source_map(results={1: output})
    assert [(s.id, s.label, s.origin, s.step) for s in source_map.sources] == [
        (1, "https://example.com/w", "worker", 1)]


def test_non_json_worker_output_is_reported_without_sources():
    source_map = build_source_map(results={2: "just prose, not JSON"})
    assert source_map.sources == ()
    assert source_map.steps_without_sources == (2,)


def test_json_without_findings_is_reported_without_sources():
    source_map = build_source_map(results={3: json.dumps({"result": "no_result"})})
    assert source_map.sources == ()
    assert source_map.steps_without_sources == (3,)


def test_failed_and_skipped_outputs_contribute_no_sources():
    source_map = build_source_map(results={
        1: "[STEP FAILED] boom",
        2: "[SKIPPED — dependency failed]",
    })
    assert source_map.sources == ()
    # They are already accounted for elsewhere — no extra diagnostic noise.
    assert source_map.steps_without_sources == ()


def test_none_and_empty_finding_sources_are_skipped():
    output = worker_json(
        ("a", "none"), ("b", ""), ("c", "https://example.com/c"))
    source_map = build_source_map(results={1: output})
    assert [s.label for s in source_map.sources] == ["https://example.com/c"]
    assert source_map.skipped_findings == 2


def test_duplicate_url_across_search_and_worker_reuses_the_first_id():
    output = worker_json(("fact", "https://example.com/a"))
    source_map = build_source_map(
        search_results="- hi [source: https://example.com/a]",
        results={1: output},
    )
    assert len(source_map.sources) == 1
    assert source_map.sources[0].id == 1
    assert source_map.sources[0].origin == "search"


def test_filename_prefix_worker_source_becomes_a_file_source():
    output = worker_json(
        ("quote", "report.pdf, section 2, quote: something exact"))
    source_map = build_source_map(
        results={1: output}, attached_filenames=("report.pdf",))
    assert [(s.id, s.label, s.url, s.origin, s.step)
            for s in source_map.sources] == [
        (1, "report.pdf", None, "worker", 1)]


def test_filename_match_requires_a_boundary():
    output = worker_json(("x", "report.pdfish, quote: nope"))
    source_map = build_source_map(
        results={1: output}, attached_filenames=("report.pdf",))
    assert source_map.sources[0].label == "report.pdfish, quote: nope"


def test_worker_steps_are_ordered_numerically_not_by_insertion():
    source_map = build_source_map(results={
        3: worker_json(("three", "https://example.com/3")),
        1: worker_json(("one", "https://example.com/1")),
    })
    assert [s.label for s in source_map.sources] == [
        "https://example.com/1", "https://example.com/3"]


def test_non_http_worker_sources_stay_plain_text():
    source_map = build_source_map(
        results={1: worker_json(("fact", "ftp://example.com/x"))})
    assert source_map.sources[0].url is None
    assert resolve_citations("{{1}}", source_map.sources).text == "[1]"


# ---------------------------------------------------------------------------
# Registry render <-> parse
# ---------------------------------------------------------------------------

def test_registry_render_parse_round_trip():
    source_map = build_source_map(
        search_results=SEARCH, rendered_files={"notes.md": "text"})
    registry = render_source_registry(source_map)
    assert registry.startswith("## Sources for citation")
    assert "- {{1}} — https://example.com/a" in registry
    assert "- {{3}} — notes.md" in registry

    parsed = parse_source_registry("prefix noise\n" + registry + "\nother noise")
    assert [(s.id, s.label, s.url) for s in parsed] == [
        (s.id, s.label, s.url) for s in source_map.sources]


def test_empty_registry_renders_nothing():
    assert render_source_registry(build_source_map()) == ""
    assert parse_source_registry("") == ()


def test_parse_ignores_instruction_mentions_of_ids():
    assert parse_source_registry("use `{{1}}` form, for example `{{2}}`") == ()


# ---------------------------------------------------------------------------
# resolve_citations — markup and bookkeeping
# ---------------------------------------------------------------------------

def test_url_marker_resolves_to_a_markdown_link():
    resolution = resolve_citations("Fact {{1}}.", URL_SOURCES)
    assert resolution.text == "Fact [1](https://example.com/a)."
    assert resolution.cited == (URL_SOURCES[0],)
    assert resolution.unknown_ids == ()
    assert resolution.marker_count == 1


def test_file_marker_resolves_to_a_bare_number():
    assert resolve_citations("See {{2}}.", URL_SOURCES).text == "See [2]."


def test_adjacent_markers_resolve_together():
    assert resolve_citations("{{1}}{{2}}", URL_SOURCES).text == (
        "[1](https://example.com/a)[2]")


def test_duplicate_marker_is_cited_once():
    resolution = resolve_citations("{{1}} and {{1}}", URL_SOURCES)
    assert resolution.text == (
        "[1](https://example.com/a) and [1](https://example.com/a)")
    assert resolution.cited == (URL_SOURCES[0],)


def test_unknown_and_out_of_range_ids_are_stripped():
    resolution = resolve_citations(
        "Claim {{99}}. Also {{0}} and {{2}}.", URL_SOURCES)
    assert resolution.text == "Claim. Also and [2]."
    assert resolution.unknown_ids == (99, 0)
    assert resolution.cited == (URL_SOURCES[1],)
    assert resolution.marker_count == 3


def test_code_fences_are_protected():
    text = "Before {{1}}\n```\n{{1}}\n```\nAfter {{1}}"
    resolution = resolve_citations(text, URL_SOURCES)
    assert resolution.text == (
        "Before [1](https://example.com/a)\n"
        "```\n{{1}}\n```\n"
        "After [1](https://example.com/a)")
    assert resolution.marker_count == 2


def test_inline_code_spans_are_protected():
    resolution = resolve_citations(
        "Use `{{1}}` literally, but cite {{1}}.", URL_SOURCES)
    assert resolution.text == (
        "Use `{{1}}` literally, but cite [1](https://example.com/a).")
    assert resolution.marker_count == 1


def test_urls_with_parentheses_use_angle_brackets():
    only = sources(("https://example.com/a_(b)", "https://example.com/a_(b)"))
    assert resolve_citations("x {{1}}", only).text == (
        "x [1](<https://example.com/a_(b)>)")


def test_text_without_citations_is_unchanged():
    resolution = resolve_citations("Plain answer.", URL_SOURCES)
    assert resolution.text == "Plain answer."
    assert resolution.cited == ()
    assert resolution.marker_count == 0
    assert append_sources_section(resolution.text, ()) == "Plain answer."


# ---------------------------------------------------------------------------
# append_sources_section / resolve_and_append
# ---------------------------------------------------------------------------

def test_appendix_lists_only_cited_sources_ascending():
    text, resolution = resolve_and_append("B {{2}} then A {{1}}.", URL_SOURCES)
    assert text == (
        "B [2] then A [1](https://example.com/a).\n\n"
        "### Sources\n\n"
        "1. https://example.com/a\n"
        "2. notes.md")
    assert [source.id for source in resolution.cited] == [1, 2]


def test_appendix_is_omitted_when_only_unknown_ids_were_emitted():
    text, resolution = resolve_and_append("Invented {{7}}.", URL_SOURCES)
    assert text == "Invented."
    assert resolution.cited == ()
    assert "### Sources" not in text


def test_append_sources_section_leaves_text_byte_for_byte_unchanged():
    assert append_sources_section("FINAL", ()) == "FINAL"
    assert append_sources_section("FINAL\n\n", ()) == "FINAL\n\n"


# ---------------------------------------------------------------------------
# StreamingCitationResolver — chunk boundaries, reset, appendix
# ---------------------------------------------------------------------------

def test_streaming_resolves_a_marker_split_across_feeds():
    resolver = StreamingCitationResolver(URL_SOURCES)
    parts = [resolver.feed("Claim {{"), resolver.feed("1"), resolver.feed("}}.")]
    assert "{{" not in "".join(parts)
    assert "".join(parts) == "Claim [1](https://example.com/a)."
    assert [source.id for source in resolver.cited] == [1]


def test_streaming_holds_a_partial_marker_until_flush():
    resolver = StreamingCitationResolver(URL_SOURCES)
    assert resolver.feed("tail {{") == "tail "
    assert resolver.flush() == "{{"
    assert resolver.flush() == ""


def test_streaming_appendix_tracks_cited_sources_only():
    resolver = StreamingCitationResolver(URL_SOURCES)
    resolver.feed("B {{2}}")
    assert resolver.sources_section() == "\n\n### Sources\n\n1. notes.md"
    resolver.feed(" and A {{1}}")
    assert resolver.sources_section() == (
        "\n\n### Sources\n\n1. https://example.com/a\n2. notes.md")
    assert [source.id for source in resolver.cited] == [1, 2]


def test_streaming_strips_unknown_ids():
    resolver = StreamingCitationResolver(URL_SOURCES)
    assert resolver.feed("Claim {{9}}.") == "Claim."
    assert resolver.sources_section() == ""


def test_streaming_reset_clears_buffer_and_cited():
    resolver = StreamingCitationResolver(URL_SOURCES)
    resolver.feed("draft {{1}}")
    assert resolver.cited
    resolver.reset()
    assert resolver.cited == ()
    assert resolver.sources_section() == ""
    resolver.feed("held {{")
    resolver.reset()
    assert resolver.flush() == ""


def test_streaming_set_sources_preserves_buffer_and_cited():
    resolver = StreamingCitationResolver()
    resolver.set_sources(URL_SOURCES)
    assert resolver.feed("Claim {{1") == "Claim "
    resolver.set_sources(URL_SOURCES)
    assert resolver.feed("}}.") == "[1](https://example.com/a)."
    assert [source.id for source in resolver.cited] == [1]
