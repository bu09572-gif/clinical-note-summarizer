"""De-identify the 300-note chart sample once (needs spaCy) and save it, so the LLM run in CI
doesn't need spaCy. -> results/deid_sample.jsonl and results/deid_counts.json"""
import json
from collections import Counter
from pathlib import Path

import data
import deid

OUT = Path(__file__).resolve().parent / "results"

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    s = data.chart_sample()
    counts, notes_with = Counter(), 0
    with (OUT / "deid_sample.jsonl").open("w") as fh:
        for r in s.itertuples():
            spans = deid.find(r.text)
            red, removed = deid.redact(r.text)
            counts.update(t for _, _, t in spans)
            notes_with += bool(spans)
            fh.write(json.dumps({"note_id": int(r.note_id), "specialty": r.medical_specialty, "keywords": r.keywords,
                                 "text": r.text, "redacted": red, "removed": removed}) + "\n")
    (OUT / "deid_counts.json").write_text(json.dumps({"notes": len(s), "notes_with_identifiers": notes_with,
                                                     "identifiers_by_type": dict(counts)}, indent=2))
    print(notes_with, dict(counts))
