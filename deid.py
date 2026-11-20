"""De-identification for clinical notes, aimed at HIPAA Safe Harbor identifiers that show up in
transcribed notes: names (after titles, and spaCy PERSON entities), full dates, phone and fax
numbers, e-mail addresses, record / account numbers, places below state level, and ages over 89.

Durations and relative times ("3 days ago", "in 2 weeks") are kept: they are clinical facts.
Each identifier is replaced by a tag such as [NAME] or [DATE].
"""
from __future__ import annotations

import re
from functools import lru_cache

MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
RULES = [
    ("NAME", r"\b(?:Mr|Mrs|Ms|Miss|Dr|Doctor)\.?\s+[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)?"),
    ("NAME", r"\b[A-Z][a-zA-Z'-]+,?\s+(?:M\.?D|D\.?O|R\.?N|P\.?A|N\.?P)\b\.?"),
    ("DATE", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    ("DATE", rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"),
    ("DATE", rf"\b\d{{1,2}}\s+{MONTHS}\s+\d{{4}}\b"),
    ("PHONE", r"(?:\(\d{3}\)\s*|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}\b"),
    ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    ("ID", r"\b(?:MRN|medical record(?: number)?|account(?: number)?|acct|SSN|social security(?: number)?)\s*(?:#|no\.?|number|:)?\s*(?=[A-Z0-9-]*\d)[A-Z0-9-]{4,}\b"),
    ("AGE", r"\b(?:9\d|1[0-1]\d)[-\s](?:year|yr)s?[-\s]old\b"),
]
RULES_RE = [(t, re.compile(p, re.I if t in ("ID", "AGE") else 0)) for t, p in RULES]
NER_KEEP = {"PERSON": "NAME", "GPE": "PLACE", "FAC": "PLACE"}
# spaCy's general model tags many drug and eponym words as PERSON; never mask these
CLINICAL_WORDS = re.compile(r"(?i)^(?:tylenol|motrin|advil|coumadin|lasix|foley|doppler|holter|babinski|romberg|"
                            r"glasgow|apgar|parkinson|alzheimer|crohn|hashimoto|bell|down|cushing|addison|"
                            r"graves|wilson|hodgkin|kaposi|homans|murphy|mcburney|valsalva|trendelenburg|betadine|"
                            r"xylocaine|marcaine|vicryl|prolene|dermabond|steri-strips?|kenalog|depo-medrol)\b")


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_md", disable=["lemmatizer", "parser"])


def find(text: str, use_ner: bool = True) -> list[tuple[int, int, str]]:
    spans = [(m.start(), m.end(), t) for t, r in RULES_RE for m in r.finditer(text)]
    if use_ner:
        # spaCy's general model mistakes many clinical words for names ("Soft", "Oropharynx", drug names),
        # so only two-word capitalised PERSON names, and places right after "in / from / at / to"
        for e in _nlp()(text).ents:
            words = e.text.split()
            if CLINICAL_WORDS.match(e.text) or not all(w[:1].isupper() and not w.isupper() for w in words):
                continue
            before = text[max(0, e.start_char - 6):e.start_char].lower()
            if (e.label_ == "PERSON" and len(words) >= 2) or \
               (e.label_ in ("GPE", "FAC") and re.search(r"\b(in|from|at|to)\s+$", before)):
                spans.append((e.start_char, e.end_char, NER_KEEP[e.label_]))
    out: list[tuple[int, int, str]] = []
    for s, e, t in sorted(spans):
        if out and s < out[-1][1]:
            ps, pe, pt = out[-1]
            out[-1] = (ps, max(pe, e), pt)
        else:
            out.append((s, e, t))
    return out


def redact(text: str, use_ner: bool = True) -> tuple[str, list[str]]:
    """Return the redacted text and the list of identifier strings that were removed."""
    spans = find(text, use_ner)
    out, last, removed = [], 0, []
    for s, e, t in spans:
        out += [text[last:s], f"[{t}]"]
        removed.append(text[s:e])
        last = e
    return "".join(out + [text[last:]]), removed
