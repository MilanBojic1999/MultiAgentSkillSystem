---
name: answer-writer
description: >
  Use this skill whenever Claude needs to compose a final answer that synthesizes research, 
  subquery results, tool outputs, or multi-step reasoning into a single cohesive response. 
  Trigger this skill when the task involves: integrating findings from multiple sources or 
  searches into one unified reply, writing a response that cites external evidence, producing 
  a polished answer after a research or investigation phase, or any time the user asks for a 
  comprehensive, professional summary of gathered information. Also trigger when the user 
  asks Claude to "write up" findings, "summarize the results", "put it all together", or 
  "give me a final answer". Do NOT skip this skill just because the answer feels obvious — 
  if you ran subqueries, used tools, or gathered sources, apply this skill to format the output.
---

# Answer Writer

A skill for composing clear, professional, well-cited final answers that synthesize research
and multi-step reasoning into a single cohesive response.

---

## Core Principles

### 1. One Unified Voice
The final answer must read as a single, continuous thought — not as a list of tool outputs
stitched together. Never use phrases like "From the search results...", "As the tool returned...",
"According to source 1...", or "Based on the data above...". Write as if you already knew
everything and are simply explaining it to the reader.

### 2. Match Length to Complexity
- **Simple factual lookup** → one sentence or short paragraph, no padding
- **Multi-part research question** → several paragraphs, each covering one aspect
- **Deep analytical question** → structured prose with a clear logical arc
- Long answers open with a short answer (see Structure Guide).

Never pad. Never repeat yourself to hit a length target.

### 3. Clarity and Professionalism
- Write in an engaging, professional style
- Use plain language; avoid jargon unless the audience clearly expects it
- Lead with the most important information; support it with detail below

---

## Citation Rules

Cite every specific claim that depends on an external source. Citations must:

- Appear **immediately after** the statement they support (inline, not at the end)
- Point to the **single most direct source** for that claim — do not over-cite
- Be **unobtrusive** — the prose must read naturally around them
- Never be piled up; one tight citation per discrete claim is the target

### Citation format

Cite using **source IDs** in double curly braces: `{{ID}}`. Each source you were given
carries an ID assigned by the system (for example `1`, `2`, `3`). Place the ID directly
after the claim, before the sentence's closing punctuation. Example:

> The company reported a 40% increase in revenue last quarter {{3}}.

**Rules for IDs:**

- **Never write URLs, titles, or file paths in citations.** Use only the ID. The system
  resolves IDs to the actual sources after you respond.
- **Use only IDs that were provided to you.** Never invent an ID, guess one, or reuse an ID
  for a source it doesn't belong to. Copy IDs exactly as given.
- **If a claim needs two sources**, place the markers adjacent, with no space or punctuation
  between them: `{{3}}{{7}}`. Do this rarely; prefer the single best source.
- **If a source has no ID**, do not cite it. Either leave the claim uncited and state it as
  background knowledge only if it is safe to do so, or omit the claim.
- **Never cite a source for something it doesn't say.** If you are unsure which source
  supports a claim, check before citing.

**What requires a citation:**
- Statistics, figures, dates, named findings
- Statements of fact that could be contested or that the reader couldn't derive independently
- Direct attributions ("the company announced...", "researchers found...")

**What does NOT require a citation:**
- General background knowledge
- Logical inferences clearly drawn from cited facts
- Your own analytical conclusions

---

## Structure Guide

### Opening
Start with the direct answer. Do not warm up with "Great question!" or "Let me explain...".
Get to the point immediately.

**For long answers, open with a short answer.** If the full answer will run past roughly
250 words or more than three paragraphs, begin with a short summary that lets the reader
get the gist without reading further. Format it as a bold label followed by one to three
sentences (about 40 to 60 words), then a blank line, then the full answer:

> Remote workers earn about 8% less on average than office-based peers in the
> same roles, though the gap nearly disappears in senior positions {{4}}{{6}}.

Rules for the short answer:

- **Write it like a news lede or a paper's abstract.** State the conclusion itself, not a
  description of what follows. Never write "Below I explain..." or "This answer covers...".
- **It must stand alone.** A reader who stops after it should come away with the correct
  main takeaway.
- **Carry the uncertainty.** If the evidence is thin, conflicting, or the question could not
  be fully answered, the short answer says so. It must never read as more confident than
  the body.
- **Cite it.** Specific figures, dates, and findings in the short answer follow the same
  citation rules as the body.
- **Skip it for short answers.** If the whole response is a sentence or a short paragraph,
  the answer itself is the summary. Do not add a label or repeat yourself.

### Body
Develop supporting reasoning in natural prose. Group related ideas into paragraphs.
Each paragraph should have one clear focus. Transition between paragraphs with logical
connectors, not bullet points or headers (unless the content is a genuine list or
step-by-step guide). When a short answer is present, the body should add evidence,
detail, and nuance rather than restate it.

### Closing
Only include a closing paragraph if there is something important to add — a caveat, a
nuance, a recommended next step. Do not summarize what you just said; the short answer
already serves that purpose.

---

## Voice

Write in a **professional** register: formal but not stiff, with complete sentences and
precise word choices. No slang. Avoid contractions where they feel too loose.

> Example: "The API rate limit was increased to 1,000 requests per minute in the latest release {{2}}."

Three topic-driven adjustments apply:

| Topic type | Adjustment |
|---|---|
| Sensitive or personal topic | Be warmer and more considerate in phrasing |
| Safety-critical information | Be plain and exact; remove anything decorative |
| Contested or ambiguous topic | Present the competing positions with balance |

---

## Common Failure Modes to Avoid

- **Source-narration**: "Source 1 says X, Source 2 says Y" → rewrite as unified prose
- **Padding**: restating the question, saying "in conclusion" and repeating yourself → cut it
- **Citation dumping**: stacking 4 citations on one sentence → pick the most relevant one
- **Bullet-ification**: converting a nuanced answer into a fragmented list → use prose
- **Hedging every sentence**: "it seems", "possibly", "one might argue" on basic facts → be direct
- **Structural leakage**: writing "Step 1 of my research was..." → the reader doesn't need your process
- **AI vocabulary**: using words and phrases that signal machine-generated text → see banned list below

---

## Banned AI Vocabulary

The following words and phrases are **strictly forbidden** in any final answer. They are
hallmarks of AI-generated writing and immediately undermine credibility and naturalness.
Never use them, even in passing:

**Filler intensifiers:** truly, deeply, profoundly, certainly, absolutely, undoubtedly,
essentially, fundamentally, basically, literally (used for emphasis, not fact)

**Vague grandeur nouns:** tapestry, landscape, testament, journey (metaphorical), realm,
beacon, cornerstone, paradigm, ecosystem (overused metaphor), facet, mosaic, fabric (metaphorical)

**AI-flavored verbs:** delve, delve into, dive deep, unpack, leverage (overused), utilize
(when "use" suffices), foster, underscore, underscore the importance of, showcase, highlight
(as a substitute for just saying the thing)

**Hollow openers:** "It's worth noting that...", "It's important to understand that...",
"In today's world...", "In the ever-changing...", "At its core...", "When it comes to..."

**Pseudo-thoughtful closers:** "The possibilities are endless.", "Only time will tell.",
"This is a complex issue with no easy answers.", "As we move forward..."

**What to do instead:** Say the thing directly. Replace "delve into the complexities of X"
with a sentence that actually explains X. Replace "a testament to human ingenuity" with
a specific description of what makes it impressive.

---

## Quick Self-Check Before Responding

Before finalizing the answer, ask:

1. Does this read as one unified voice, or does it feel like a transcript of tool calls?
2. Is every specific claim that needs a citation actually cited in `{{ID}}` format, using only IDs I was given, with no URLs written out?
3. Is there any sentence that could be cut without losing meaning?
4. Does the length match the actual complexity of the question?
5. Does it sound like a knowledgeable person explaining something, or like a summary report?
6. Does it contain any word from the Banned AI Vocabulary list? If so, rewrite that sentence.
7. Is the register professional, with the topic adjustments applied where relevant?
8. If the answer is long, does it open with a short answer that stands alone, matches the
   body's confidence level, and is not repeated at the end? If the answer is short, did I
   leave the label off?
  
If any answer is "no", revise before delivering.