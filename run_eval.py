"""Summarize the 300 chart notes twice, from the original note and from the de-identified note,
and score both summaries. Runs in CI with free local models.

    python run_eval.py --shard 0/20
    python run_eval.py --merge
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import summarize as sm

OUT = Path(__file__).resolve().parent / "results"
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
JUDGE = os.environ.get("EVAL_JUDGE_MODEL", "llama3.2:3b")


def score(note: dict, source: str) -> dict:
    c = sm.summarize(note[source], model=MODEL)
    s = c.text
    unsup = sm.judge(note["text"], s, JUDGE)
    return {"summary": s, "leaked": sm.leaked(s, note["removed"]),
            "unsupported_numbers": sm.unsupported_numbers(s, note["text"]),
            "keyword_coverage": sm.keyword_coverage(s, note["keywords"]),
            "judge_unsupported": unsup, "words": len(s.split()), "latency_s": round(c.latency_s, 1)}


def run(shard: str | None, limit: int | None) -> None:
    notes = [json.loads(line) for line in (OUT / "deid_sample.jsonl").open()]
    i, n = map(int, shard.split("/")) if shard else (0, 1)
    notes = notes[i::n][:limit] if limit else notes[i::n]
    with (OUT / f"summaries_shard{i}.jsonl").open("w") as fh:
        for k, note in enumerate(notes):
            row = {"note_id": note["note_id"], "specialty": note["specialty"], "note_words": len(note["text"].split()),
                   "identifiers_in_note": len(note["removed"])}
            for cfg, src in [("original", "text"), ("deidentified", "redacted")]:
                try:
                    row[cfg] = score(note, src)
                except Exception as e:  # noqa: BLE001 - one failed call skips one summary
                    row[cfg] = {"error": type(e).__name__}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(k, len(notes), flush=True)


def summarize_rows(df: pd.DataFrame) -> dict:
    out = {"notes": len(df), "notes_with_identifiers": int((df.identifiers_in_note > 0).sum()), "configs": {}}
    has_ids = df.identifiers_in_note > 0
    for cfg in ["original", "deidentified"]:
        r = pd.json_normalize(df[cfg].tolist()).set_index(df.index)
        ok = r.get("error", pd.Series(index=r.index, dtype=object)).isna()
        r = r[ok]
        judged = r.judge_unsupported.dropna()
        out["configs"][cfg] = {
            "summaries": int(ok.sum()),
            "summaries_leaking_an_identifier": int((r.leaked.map(len) > 0).sum()),
            "leak_rate_on_notes_with_identifiers": round(float((r.leaked[has_ids[r.index]].map(len) > 0).mean()), 4),
            "summaries_with_unsupported_numbers": round(float((r.unsupported_numbers.map(len) > 0).mean()), 4),
            "keyword_coverage_mean": round(float(r.keyword_coverage.dropna().mean()), 4),
            "judge_parsed": int(len(judged)),
            "judge_found_no_unsupported_statement": round(float((judged.map(len) == 0).mean()), 4),
            "median_words": float(r.words.median()), "median_latency_s": float(r.latency_s.median())}
    out["median_note_words"] = float(df.note_words.median())
    return json.loads(json.dumps(out, default=float))


def merge() -> dict:
    files = sorted(OUT.glob("summaries_shard*.jsonl"))
    df = pd.concat([pd.read_json(f, lines=True) for f in files if f.stat().st_size], ignore_index=True)
    df.to_json(OUT / "summaries.jsonl", orient="records", lines=True)
    for f in files:
        f.unlink()
    s = summarize_rows(df)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    if a.merge:
        print(json.dumps(merge(), indent=2))
    else:
        run(a.shard, a.limit)


if __name__ == "__main__":
    main()
