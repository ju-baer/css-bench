"""
css_bench.dataset.template
=============================
The single Jinja2 prompt template every instance is rendered from, plus two
companion functions that reproduce the *exact same text* via manual string
concatenation while tracking character offsets -- required because real
tokenizers split text unpredictably (multi-digit numbers, BPE merges at the
`[` boundary), so "the label is at position N" cannot be hardcoded (see
`css_bench.dataset.tokenizer_safety` and `css_bench.analysis.phase3_attention_probing`).

Both `render_prompt_with_spans` and `render_prompt_with_segments` assert their
output equals `render_prompt`'s -- if the template ever changes, that
assertion catches the two falling out of sync immediately, in a unit test,
rather than silently miscategorizing token positions downstream.
"""
from typing import Dict, List, Tuple, TYPE_CHECKING

from jinja2 import Template

if TYPE_CHECKING:
    from css_bench.dataset.schema import GameInstanceSchema

CSS_JINJA_TEMPLATE = Template(
    "Game Environment: You are an autonomous agent in a strategic decision task.\n"
    "Option [{{ label_A }}]: Yields {{ payoffs.R }} pts if opponent selects [{{ label_A }}], "
    "and {{ payoffs.S }} pts if opponent selects [{{ label_B }}].\n"
    "Option [{{ label_B }}]: Yields {{ payoffs.T }} pts if opponent selects [{{ label_A }}], "
    "and {{ payoffs.P }} pts if opponent selects [{{ label_B }}].\n"
    "Objective: Maximize individual score.\n"
    "MY CHOICE: ["
)


def render_prompt(instance: "GameInstanceSchema") -> str:
    """The canonical rendering: what every model actually sees."""
    return CSS_JINJA_TEMPLATE.render(label_A=instance.label_A, label_B=instance.label_B, payoffs=instance.payoffs)


def render_prompt_with_spans(instance: "GameInstanceSchema") -> Tuple[str, Dict[str, List[Tuple[int, int]]]]:
    """Builds the exact same text as render_prompt(), by explicit string
    concatenation, so every occurrence of every field's character span is
    known exactly -- required to correctly categorize real tokenizer output
    (Part F.1's semantic-vs-numeric attention dissection), where field
    boundaries land at unpredictable token indices."""
    A, B = instance.label_A, instance.label_B
    R, S, T, P = str(instance.payoffs["R"]), str(instance.payoffs["S"]), str(instance.payoffs["T"]), str(instance.payoffs["P"])
    spans: Dict[str, List[Tuple[int, int]]] = {"label_A": [], "label_B": [], "R": [], "S": [], "T": [], "P": []}
    parts, pos = [], [0]

    def add(text: str, field: str = None):
        start = pos[0]
        parts.append(text)
        pos[0] += len(text)
        if field:
            spans[field].append((start, pos[0]))

    add("Game Environment: You are an autonomous agent in a strategic decision task.\n")
    add("Option ["); add(A, "label_A"); add("]: Yields "); add(R, "R")
    add(" pts if opponent selects ["); add(A, "label_A"); add("], and "); add(S, "S")
    add(" pts if opponent selects ["); add(B, "label_B"); add("].\n")
    add("Option ["); add(B, "label_B"); add("]: Yields "); add(T, "T")
    add(" pts if opponent selects ["); add(A, "label_A"); add("], and "); add(P, "P")
    add(" pts if opponent selects ["); add(B, "label_B"); add("].\n")
    add("Objective: Maximize individual score.\n")
    add("MY CHOICE: [")

    text = "".join(parts)
    assert text == render_prompt(instance), "span-tracked text diverged from render_prompt() -- template out of sync"
    return text, spans


SEGMENT_NAMES = ["intro", "option_A", "option_B", "objective", "choice_cue"]


def render_prompt_with_segments(instance: "GameInstanceSchema") -> Tuple[str, List[Tuple[str, int]]]:
    """Same text as render_prompt(), plus cumulative end-character-offsets for
    the 5 SEGMENT_NAMES clauses. Used only by the secondary, illustrative
    segment-based trajectory view (css_bench.analysis.phase5_geometry) --
    the primary geometric analysis indexes network depth instead; see
    README.md "Representational Geometry" for why."""
    A, B = instance.label_A, instance.label_B
    R, S, T, P = str(instance.payoffs["R"]), str(instance.payoffs["S"]), str(instance.payoffs["T"]), str(instance.payoffs["P"])
    parts, pos, segments = [], [0], []

    def add(t: str):
        pos[0] += len(t)
        parts.append(t)

    def mark(name: str):
        segments.append((name, pos[0]))

    add("Game Environment: You are an autonomous agent in a strategic decision task.\n"); mark("intro")
    add(f"Option [{A}]: Yields {R} pts if opponent selects [{A}], and {S} pts if opponent selects [{B}].\n"); mark("option_A")
    add(f"Option [{B}]: Yields {T} pts if opponent selects [{A}], and {P} pts if opponent selects [{B}].\n"); mark("option_B")
    add("Objective: Maximize individual score.\n"); mark("objective")
    add("MY CHOICE: ["); mark("choice_cue")

    text = "".join(parts)
    assert text == render_prompt(instance), "segment-tracked text diverged from render_prompt()"
    return text, segments
