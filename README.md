# Clinical Note Summarizer with De-identification

This project summarizes real clinical notes into the five sections a clinician wants (reason for visit, findings, assessment, plan, medications) with a free local model. It removes patient identifiers *before* the note reaches the model. Every summary is checked for leaked identifiers, invented numbers and coverage of the note's key terms. I also test whether an "LLM as judge" can be trusted to catch made-up statements. It can't, at this size.

## Data

**MTSamples:** about 5,000 real transcribed medical reports, shared publicly for education, with a specialty and keywords for each. I use the Hugging Face copy `harishnair04/mtsamples`, mirrored into this repo's `data-v2` release with a SHA-256 check.

I take a fixed sample of 300 notes across specialties: consults, discharge summaries, progress notes and so on. The median length is 445 words. The notes were already partly scrubbed by their publisher, but many still have dates, doctor names, hospital names and places.

## How it works

1. **De-identification** (`deid.py`): rules for the HIPAA Safe Harbor identifiers that appear in transcribed notes.
   - The rules cover full dates, names after titles ("Dr.", "Mrs.") or before credentials ("MD"), phone numbers, e-mail addresses, record and account numbers, and ages over 89.
   - spaCy's English NER finds other names and places, restricted to cut false alarms. A PERSON must be two or more capitalised words, and a place must follow "in/from/at/to".
   - Durations like "3 days ago" are kept, because they are clinical facts.
2. **Summary** (`summarize.py`): Qwen2.5 3B on CPU (Ollama, GitHub Actions) writes the five sections from the note, at temperature 0. Each note is summarized twice: from the original text and from the de-identified text.
3. **Checks** on every summary (no reference summaries exist for these notes):
   - **Leaked identifiers:** any identifier found in the original note that appears in the summary.
   - **Unsupported numbers:** numbers in the summary that are not in the note, such as doses, vitals or lab values.
   - **Keyword coverage:** the share of the note's MTSamples keywords that the summary mentions.
   - **LLM judge:** Llama 3.2 3B reads the note and summary and lists statements the note does not support.

## De-identification results

`results/deid_counts.json` and `results/deid_sample.jsonl`:
- 166 of the 300 notes contain at least one identifier.
- The detections are 281 dates, 116 names, 80 places, 1 phone number and 2 ages over 89.

**My manual review of 60 random detections:**
- One clear false positive, a phrase ("account sexual") caught by the record-number rule. I fixed the rule to require a digit.
- Some places are states or countries. Safe Harbor only requires places smaller than a state, so these are over-removal, which is safe but loses a little context.
- Many detected "names" are the publisher's placeholders (Dr. XYZ, Ms. ABC), so real names in this corpus are rarer than the counts suggest.

## Summary results (300 notes)

From `results/summary.json`. Every summary is in `results/summaries.jsonl`.

| | From original note | From de-identified note |
|---|---|---|
| summaries repeating an identifier | 36 | 6 |
| ... share of notes that contain identifiers | 21.7% | 3.6% |
| summaries with a number not in the note | 8.0% | 6.7% |
| note keywords covered (mean) | 61.6% | 64.2% |
| median summary length | 119 words | 124 words |
| median time per summary (CPU) | 31.6 s | 21.7 s |

- **De-identifying first works, and costs nothing in quality.**
  - Asked not to include names, dates or places, the model still repeats one in 22% of notes that have them. Instructions alone are not a privacy control.
  - From de-identified text that falls to 6 summaries, and keyword coverage is slightly higher.
  - The 6 remaining "leaks" are mostly my de-identifier's own false detections, which then match normal words in the summary:
    - two drug names (Arava, Tekturna) and a disease (Huntington's), flagged as a name or place;
    - generic "Medical Center" and "Memorial Hospital";
    - one real miss: a country named in a sentence my place rule doesn't cover.
- **About 7% of summaries contain a number not in the note.** This is the error that matters most in clinical text, a wrong dose or value. Checking numbers is cheap and exact, so it should run on every summary.
- **The 3B LLM judge is not usable.** It flagged at least one "unsupported" statement in 98% of summaries, 4.7 per summary on average. I checked the 861 statements it flagged against the note: **81% have at least 90% of their words in the note**, so most flags are wrong. A judge this small adds noise, not safety. The deterministic checks above are more useful, and a judge needs its own accuracy test before anyone relies on it.

## Limits

- The notes are public teaching transcriptions, not a hospital's EHR. Real notes have more identifiers and messier text.
- With no reference summaries, I can measure leaks, numbers and coverage, but not overall clinical correctness. That needs clinician review.
- The de-identifier is rule-based plus a general NER model. A production system would use a clinical de-identification model and measure recall on labelled data, as in the [PII redaction project](../../core/pii-redaction).

## Running it

```bash
pip install spacy && python -m spacy download en_core_web_md
python deid_sample.py          # de-identify the 300-note sample
pytest -q test_summarizer.py
# summaries: edit run-request.txt and push; GitHub Actions runs the models and commits results/
```
