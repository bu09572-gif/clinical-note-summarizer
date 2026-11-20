"""MTSamples: 4,999 de-identified sample medical transcriptions (mtsamples.com) with specialty,
a one-line description and keywords, from the harishnair04/mtsamples dataset on Hugging Face,
mirrored to this repo's data-v2 release with its SHA-256 pinned here.

The notes were published as samples, so most names are gone, but some still carry doctor and
patient names ("Dr. Chang", "Ms. Wilson") and full dates. That is what the de-identification step removes.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PUBLIC_V2_DIR", ROOT / "data" / "raw" / "public_v2"))
URL = "https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v2/mtsamples__train__0.parquet"
SHA256 = "89d8e5eabe6769ea530b0c07da50f27c7a1605b836570e878cef5b956ca9ed80"
CHART_TYPES = ["Consult - History and Phy.", "SOAP / Chart / Progress Notes", "Discharge Summary",
               "Emergency Room Reports", "General Medicine"]


def notes() -> pd.DataFrame:
    p = RAW / "mtsamples__train__0.parquet"
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        p.write_bytes(requests.get(URL, timeout=300).content)
    if hashlib.sha256(p.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("mtsamples file does not match the mirrored copy")
    d = pd.read_parquet(p).rename(columns={"Unnamed: 0": "note_id"})
    for c in ["description", "medical_specialty", "sample_name", "transcription", "keywords"]:
        d[c] = d[c].fillna("").str.strip()
    # the source joins lines with ",": turn ",  " separators back into line breaks
    d["text"] = d.transcription.str.replace(r",\s*(?=[A-Z][A-Z /&-]{2,}:)", "\n", regex=True)
    return d


def chart_sample(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Chart-style notes (H&P, progress notes, discharge summaries, ER reports) of normal length."""
    d = notes()
    d = d[d.medical_specialty.isin(CHART_TYPES) & d.text.str.len().between(800, 8000)]
    d = d.drop_duplicates("text")
    return d.sample(min(n, len(d)), random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    s = chart_sample()
    print(len(notes()), "notes;", len(s), "in the chart sample;", s.medical_specialty.value_counts().to_dict())
