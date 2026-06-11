---
name: yotta-researcher
description: >
  Use this skill for any research or information-gathering task where the researcher has access to
  tools such as web search, Google Drive, Gmail, Google Calendar, or other MCP-connected services.
  Trigger whenever the user asks to look something up, find sources, gather evidence, summarize
  findings from the web or documents, verify facts, compare sources, compile a research brief, or
  produce an answer that should be grounded in external or retrieved information.
  
  Also trigger when the user explicitly requests citations, source attribution, fact-checking,
  or says phrases like "look this up", "find out", "search for", "what does the research say",
  "pull information from", "retrieve", "check my inbox / drive / calendar for", or any phrasing
  that implies reading live data from a tool. Do NOT skip this skill just because the query
  seems simple — if an external tool could improve the answer, use this skill.
---

# Research & Information Retrieval Skill

A structured approach for conducting research using available tools (web search, MCPs, documents)
and producing well-sourced, citation-linked answers.

---

## Core Principles

1. **Retrieve before asserting.** Never answer from memory alone when a tool can verify or
   update the answer. Recency matters; outdated facts erode trust.

2. **Cite at the claim level.** Every specific factual claim that comes from a retrieved source
   must carry an inline citation tied directly to that claim — not a footnote section at the end.
   Use the format `{{https://source-url.com}}` unless the system provides a different citation
   format (e.g., `` tags in Claude.ai — follow whatever format is active).

3. **Distinguish knowledge tiers.** Be transparent about where information comes from:
   - 🔍 **Retrieved** — came from a tool call in this session
   - 📚 **Training knowledge** — from pre-training, not externally verified this session
   - ❓ **Unverified** — could not be confirmed; label it explicitly

4. **Fail gracefully, don't fabricate.** If a search returns no useful results, say so. If a
   tool call fails or times out, note the failure briefly and continue with remaining sources.
   Never invent sources or fill gaps with invented statistics or quotes.

5. **Scale tool calls to complexity.** Simple factual queries need 1–2 calls. Deep research
   or multi-angle comparisons may warrant 5–15 calls across sources. For tasks that would
   require 20+ calls, consider suggesting the Research feature or splitting into sub-tasks.

---

## Retrieval Strategy

### Step 1 — Identify what needs retrieval

Before calling any tool, mentally classify each part of the query:
- **Time-sensitive** (prices, events, current officeholders, recent publications) → must retrieve
- **Potentially stale** (statistics, policies, company info, scientific consensus) → retrieve to verify
- **Stable / definitional** (historical events, mathematical facts, definitions) → training knowledge is fine, retrieve only if precision matters

### Step 2 — Choose the right tool

| Need | Preferred tool |
|---|---|
| General web facts, news, recent events | `call_yotta` → `call_yotta_elastic_keyword` for full articles |

**Tool priority rule:** Use internal/MCP tools for personal or organizational data before
reaching for web search. Use `call_yotta_elastic_keyword` after `call_yotta` when snippets are too short to
support a claim.

### Step 3 — Execute retrieval efficiently

- Start with broad queries (1–3 keywords), then narrow based on results.
- Never repeat a failed query verbatim — vary terms, add context, or try a different source.
- For multi-part research questions, batch related sub-questions into parallel or sequential
  searches rather than one giant query.
- Fetch full pages when: the claim is high-stakes, the snippet is ambiguous, or the source is
  primary (e.g., government site, official announcement).

---

## Citation Rules

### Inline citation format

Attach citations immediately after the claim they support, not at the end of a paragraph:

> ✅  The unemployment rate fell to 3.8% in April 2025 {{https://bls.gov/...}}, a figure that
>     economists attribute partly to service-sector growth {{https://wsj.com/...}}.

> ❌  The unemployment rate fell to 3.8% in April 2025, a figure economists attribute to
>     service-sector growth.  
>     Sources: bls.gov, wsj.com

### One citation per claim minimum

If the same fact is supported by multiple independent sources, cite the strongest one.
Only add a second citation if sources meaningfully disagree or the claim is high-stakes.

### When sources conflict

State the conflict explicitly:

> Source A reports X {{url-A}}, while Source B reports Y {{url-B}}. The discrepancy may stem
> from different measurement dates / methodologies. Recommend verifying with primary data.

### When no source is found

> I searched for [topic] but could not find reliable information. The following is based on
> training knowledge (📚) and should be independently verified.

---

## Output Structure

For short factual answers (1–3 claims): inline prose with citations is sufficient.

For research briefs or multi-question answers, use this structure:

```
## [Topic / Research Question]

**Summary** (2–4 sentences with key findings and top citations)

**Findings**
- [Claim 1] {{source}}
- [Claim 2] {{source}}
...

**Gaps / Unverified Points**
- [Anything that could not be confirmed, labeled 📚 or ❓]

**Recommended Next Steps** (optional)
- Suggested follow-up searches or sources worth consulting
```

Adapt length and formality to the researcher's request. A quick "what's the current X?" needs
only a sentence and citation. A literature review needs the full structure.

---

## Handling Tool Failures

| Failure type | Response |
|---|---|
| Search returns irrelevant results | Try 1–2 alternative queries; if still nothing, say so |
| `call_yotta` times out | Note the failure, cite from snippet if sufficient, flag as partially verified |
| MCP auth error | Tell the user the connection needs re-authentication; proceed with other sources |
| Rate limit | Pause and note; use training knowledge with 📚 label as fallback |
| No results anywhere | "Could not find reliable information on [topic]. Proceeding from training knowledge — recommend external verification." |

Never silently drop a failed source. A brief note ("Search for X returned no useful results")
maintains transparency.

---

## Quality Checklist (before finalizing any researched answer)

- [ ] Every time-sensitive or specific factual claim has a retrieved source
- [ ] All citations are inline, immediately following the claim they support
- [ ] Conflicting sources are surfaced, not silently resolved
- [ ] Failed searches or tool errors are disclosed
- [ ] Training-knowledge-only sections are labeled 📚
- [ ] No invented URLs, author names, publication titles, or statistics
- [ ] Output length matches the complexity of the request

---

## Example Patterns

### Simple factual lookup
**User:** What is the current prime interest rate in the US?

**Approach:** `call_yotta` → cite the source from the list of answers

---

### Multi-source comparison
**User:** Compare how NYT and WSJ covered the latest Fed decision.

**Approach:** Search each outlet separately → `call_yotta` or `call_yotta_elastic_keyword` both articles → summarize each in
own words → note any differences in framing, cite both.

---

### Research with gaps
**User:** What are the long-term effects of [very recent/niche drug]?

**Approach:** Iterate over tools `call_yotta` or `call_yotta_elastic_keyword`, If no clear results found, say so clearly.
Fall back to known mechanism-of-action from training knowledge, labeled 📚.

---

*This skill covers research tasks that involve any combination of web search, MCP-connected
services, and document retrieval. For purely conversational or creative tasks with no
information-retrieval component, this skill does not need to apply.*