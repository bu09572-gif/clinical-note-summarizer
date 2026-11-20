"""Summarize a clinical note into fixed sections with a free local model, and check the result.

Checks on every summary (no reference summaries exist for these notes):
  leaked identifiers   identifiers found in the original note that appear in the summary
  unsupported numbers  numbers in the summary that are not in the note (doses, vitals, lab values)
  keyword coverage     share of the note's MTSamples keywords that the summary mentions
  judge                a second model reads note + summary and lists statements the note does not support
"""
from __future__ import annotations

import json
import re

from docqa import llm

SYSTEM = """You summarize clinical notes for another clinician. Use only facts stated in the note.
Write these sections, each 1-3 short bullet points, and write "not stated" if the note has nothing:
Reason for visit:
Key findings:
Assessment:
Plan:
Medications:
Do not include names, dates, places or record numbers."""

JUDGE = """You check a summary of a clinical note. List every statement in the summary that the note does
not support (wrong, made up, or contradicting the note). Reply with JSON only:
{"unsupported": ["...", ...]}  (an empty list if every statement is supported)"""

NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def summarize(note: str, model: str | None = None) -> llm.Completion:
    return llm.chat(SYSTEM, f"Note:\n{note[:9000]}", model=model, max_tokens=350)


def unsupported_numbers(summary: str, note: str) -> list[str]:
    have = set(NUM.findall(note))
    return [n for n in NUM.findall(summary) if n not in have and n.rstrip("0").rstrip(".") not in have]


def leaked(summary: str, identifiers: list[str]) -> list[str]:
    s = summary.lower()
    return sorted({i for i in identifiers if len(i) > 3 and i.lower() in s})


def keyword_coverage(summary: str, keywords: str) -> float | None:
    kws = [k.strip().lower() for k in str(keywords or "").split(",")[1:] if len(k.strip()) > 3]   # first is the specialty
    if not kws:
        return None
    s = summary.lower()
    return sum(k in s for k in kws) / len(kws)


def judge(note: str, summary: str, model: str) -> list[str] | None:
    c = llm.chat(JUDGE, f"Note:\n{note[:9000]}\n\nSummary:\n{summary}", model=model, max_tokens=300, json_mode=True)
    try:
        v = json.loads(re.search(r"\{.*\}", c.text, re.S).group(0)).get("unsupported", [])
        return [str(x) for x in v] if isinstance(v, list) else None
    except (AttributeError, json.JSONDecodeError):
        return None
