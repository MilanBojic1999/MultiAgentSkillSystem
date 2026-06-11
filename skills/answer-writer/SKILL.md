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

Never pad. Never repeat yourself to hit a length target.

### 3. Clarity and Professionalism
- Write in an engaging, professional style
- Use plain language; avoid jargon unless the audience clearly expects it
- Show your reasoning — let the reader follow how conclusions were reached
- Lead with the most important information; support it with detail below

---

## Citation Rules

Cite every specific claim that depends on an external source. Citations must:

- Appear **immediately after** the statement they support (inline, not at the end)
- Use the minimum number of sentences needed to back the claim — do not over-cite
- Be **invisible as citations** — the prose must flow naturally around them
- Never be piled up; one tight citation per discrete claim is the target

**Citation format:** Always use `{{URL}}` syntax, placing the URL of the source directly
inside double curly braces immediately after the claim. Example:

> The company reported a 40% increase in revenue last quarter {{https://example.com/report}}.

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
Start with the direct answer or the clearest summary statement. Do not warm up with
"Great question!" or "Let me explain...". Get to the point immediately.

### Body
Develop supporting reasoning in natural prose. Group related ideas into paragraphs.
Each paragraph should have one clear focus. Transition between paragraphs with logical
connectors, not bullet points or headers (unless the content is a genuine list or
step-by-step guide).

### Closing
Only include a closing paragraph if there is something genuinely important to add —
a caveat, a nuance, a recommended next step. Do not summarize what you just said.

---

## Tone Calibration

Tone is a **configurable parameter**. The programmer or system prompt sets it explicitly.
If no tone is specified, fall back to the default. Claude must not override a set tone
based on its own judgment about what "feels right" for the topic.

---

### Setting the Tone Parameter

In the system prompt or skill invocation, set tone like this:

```
ANSWER_TONE: professional
```

Or pass it inline when calling the skill:

```
Write the final answer. Tone: casual
```

---

### Available Tone Levels

**`professional`** *(default)*
Formal but not stiff. Complete sentences, precise word choices, no slang. Suitable for
business reports, research summaries, client-facing content. Avoids contractions where
they feel too loose.
> Example: "The API rate limit was increased to 1,000 requests per minute in the latest release."

**`neutral`**
Plain and factual. No personality either way — just clear, direct information delivery.
Good for documentation, FAQs, or contexts where the writer should be invisible.
> Example: "The rate limit is 1,000 requests per minute."

**`conversational`**
Warm, natural, like explaining something to a colleague. Contractions are fine. Occasional
rhetorical questions or light analogies are allowed. No slang. Suitable for blog posts,
help articles, or general-audience explainers.
> Example: "The good news is they bumped the rate limit to 1,000 requests per minute, so most apps won't hit it."

**`casual`**
Relaxed and direct, as if texting a technically literate friend. Short sentences, contractions,
minor informalities. No forced enthusiasm or filler. Suitable for developer tools, chat
interfaces, or informal internal tools.
> Example: "Rate limit's now 1k/min — you're probably fine unless you're hammering it."

**`informal`**
Same as casual but allows light humor, colloquialisms, and a more playful voice where
it fits naturally. Never forced. Suitable for consumer apps, communities, or anywhere
the brand voice is explicitly relaxed.
> Example: "They finally raised the rate limit to 1k/min. No more babysitting your request queue."

---

### Tone × Topic Overrides

Even with a set tone, a few topic-driven adjustments always apply regardless:

| Topic type | Override |
|---|---|
| Sensitive or personal topic | Pull one level warmer (e.g. casual → conversational) |
| Safety-critical information | Pull one level more precise (e.g. casual → neutral) |
| Contested or ambiguous topic | Add balance regardless of tone level |

These are the only automatic overrides. Everything else respects the set tone parameter.

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
2. Is every specific claim that needs a citation actually cited, using `{{URL}}` format?
3. Is there any sentence that could be cut without losing meaning?
4. Does the length match the actual complexity of the question?
5. Does it sound like a knowledgeable person explaining something, or like a summary report?
6. Does it contain any word from the Banned AI Vocabulary list? If so, rewrite that sentence.
7. Does the writing match the set tone parameter? If no tone was set, is it defaulting to `professional`?

If any answer is "no", revise before delivering.