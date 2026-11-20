"""Tests: de-identification rules, summary checks, and the run loop with a scripted model."""
import json

import deid
import run_eval
import summarize as sm

from docqa import llm


def test_rules_catch_identifiers_and_keep_clinical_facts():
    t = ("Ms. Wilson was seen on 03/16/2008 by Dr. Chang. Call (555) 123-4567. MRN: 448812. "
         "She is a 93-year-old woman with pain for 3 days. Soft, nontender abdomen. BP 130/70.")
    red, removed = deid.redact(t, use_ner=False)
    for x in ["Wilson", "03/16/2008", "Chang", "555", "448812", "93-year-old"]:
        assert x not in red, x
    for x in ["3 days", "Soft, nontender", "130/70"]:
        assert x in red, x
    assert deid.find("take into account sexual history", use_ner=False) == []


def test_summary_checks():
    note = "Patient on lisinopril 10 mg. Seen by Dr. Chang. Keywords text."
    assert sm.unsupported_numbers("Lisinopril 10 mg daily, potassium 4.5", note) == ["4.5"]
    assert sm.leaked("Follow up with Dr. Chang.", ["Dr. Chang"]) == ["Dr. Chang"]
    assert sm.keyword_coverage("hypertension and lisinopril", "cardio, hypertension, lisinopril, asthma") == 2 / 3


def test_run_loop_with_scripted_model(monkeypatch, tmp_path):
    note = {"note_id": 1, "specialty": "x", "keywords": "x, chest pain", "text": "Dr. Chang saw him for chest pain, BP 140/90.",
            "redacted": "[NAME] saw him for chest pain, BP 140/90.", "removed": ["Dr. Chang"]}
    (tmp_path / "deid_sample.jsonl").write_text(json.dumps(note) + "\n")
    monkeypatch.setattr(run_eval, "OUT", tmp_path)

    def chat(system, user, **kw):
        if "check a summary" in system:
            return llm.Completion('{"unsupported": []}', 1, 1, 0, "j", "test")
        text = "Reason for visit: chest pain seen by Dr. Chang" if "Dr. Chang saw" in user else "Reason for visit: chest pain"
        return llm.Completion(text, 1, 1, 0, "m", "test")
    monkeypatch.setattr(llm, "chat", chat)
    run_eval.run(None, None)
    s = run_eval.merge()
    assert s["configs"]["original"]["summaries_leaking_an_identifier"] == 1
    assert s["configs"]["deidentified"]["summaries_leaking_an_identifier"] == 0
    assert s["configs"]["deidentified"]["keyword_coverage_mean"] == 1.0
