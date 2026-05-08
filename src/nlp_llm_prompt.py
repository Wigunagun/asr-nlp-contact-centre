"""
nlp_llm_prompt.py
-----------------
NLP and LLM processing pipeline for INSW Contact Centre transcripts.
Uses GPT-4.1 Mini for topic classification, named entity recognition,
and conversation summarisation.

Reference:
    Cahyana, W. et al. "Automatic Speech Recognition for Voice Log Analysis:
    A Case Study in the Indonesia National Single Window Contact Center."
    Binus Online, 2025.

Usage:
    python nlp_llm_prompt.py --db ./database_log.db

Requirements:
    - OPENAI_API_KEY must be set as an environment variable
    - Run asr_pipeline.py first to populate the database
"""

import os
import json
import sqlite3
import argparse
from openai import OpenAI


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME = "gpt-4.1-mini"

# Service classification labels used at INSW
VALID_CLASSIFICATIONS = ["eCOO", "SSM Izin", "SSM QC", "PIB", "PEB", "N/A"]


# ── Prompt Template ────────────────────────────────────────────────────────────

def build_prompt(transcript: str) -> str:
    """
    Build the GPT-4.1 Mini extraction prompt.
    Instructs the model to return structured JSON with:
    - summary, date, caller name, caller gender, reference number, classification
    """
    return f"""Extract elemen informasi dari percakapan berikut ini:
nama agent, nama penelepon, gender penelepon, summary percakapan,
nomor car / nomor aju (26 digit),
klasifikasi (eCOO, SSM Izin, SSM QC, PIB, PEB, dan N/A jika tidak ada)

Return dalam format JSON seperti:
{{
  "data": {{
    "summary": "...",
    "tanggal": "...",
    "nama_penelepon": "...",
    "gender_penelepon": "...",
    "no_aju": "...",
    "klasifikasi": "..."
  }}
}}

Percakapan:
{transcript}
"""


# ── OpenAI Client ──────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """Initialise and return OpenAI client using environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Export it as an environment variable before running this script."
        )
    return OpenAI(api_key=api_key)


# ── LLM Extraction ─────────────────────────────────────────────────────────────

def extract_insights(client: OpenAI, transcript: str) -> dict:
    """
    Send transcript to GPT-4.1 Mini and return structured JSON output.

    Returns:
        Dictionary with keys: summary, tanggal, nama_penelepon,
        gender_penelepon, no_aju, klasifikasi
    """
    prompt = build_prompt(transcript)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert analyst for the Indonesia National Single Window "
                    "contact centre. Extract structured information from Indonesian-language "
                    "call transcripts and return only valid JSON. No explanation, no markdown."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,   # Deterministic output for consistent classification
        max_tokens=512
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        return parsed.get("data", {})
    except json.JSONDecodeError:
        print(f"  ⚠ Failed to parse JSON response. Raw output:\n{raw}")
        return {}


def validate_classification(classification: str) -> str:
    """Ensure classification matches a valid INSW service category."""
    if classification in VALID_CLASSIFICATIONS:
        return classification
    return "N/A"


# ── Database ───────────────────────────────────────────────────────────────────

def get_pending_records(db_path: str) -> list:
    """Fetch all records where NLP processing is not yet complete (finished = 0)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, uraian FROM hasil_log WHERE finished = 0
    """)
    records = cursor.fetchall()
    conn.close()
    return records


def update_record(db_path: str, record_id: int, insights: dict):
    """Update a record with NLP/LLM results and mark as finished = 1."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE hasil_log SET
            summary            = ?,
            nama_penelepon     = ?,
            gender_penelepon   = ?,
            no_aju             = ?,
            klasifikasi        = ?,
            finished           = 1
        WHERE id = ?
    """, (
        insights.get("summary", ""),
        insights.get("nama_penelepon", ""),
        insights.get("gender_penelepon", ""),
        insights.get("no_aju", ""),
        validate_classification(insights.get("klasifikasi", "N/A")),
        record_id
    ))
    conn.commit()
    conn.close()


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(db_path: str):
    """
    Main NLP/LLM pipeline:
    1. Fetch all unprocessed transcripts (finished = 0)
    2. Send each to GPT-4.1 Mini for extraction
    3. Validate and save structured results
    4. Mark record as finished = 1
    """
    client = get_client()
    pending = get_pending_records(db_path)

    if not pending:
        print("No pending records to process. All transcripts are up to date.")
        return

    print(f"Found {len(pending)} record(s) to process with NLP/LLM.\n")

    for record_id, transcript in pending:
        print(f"Processing record ID: {record_id}")

        if not transcript or len(transcript.strip()) < 20:
            print("  ⚠ Transcript too short or empty — skipping.")
            continue

        try:
            insights = extract_insights(client, transcript)
            if insights:
                update_record(db_path, record_id, insights)
                print(f"  ✓ Classification: {insights.get('klasifikasi', 'N/A')}")
                print(f"  ✓ Summary: {insights.get('summary', '')[:80]}...")
            else:
                print("  ✗ No insights extracted.")
        except Exception as e:
            print(f"  ✗ Error on record {record_id}: {e}")

    print(f"\n✓ NLP/LLM processing complete. Status updated to finished = 1.")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLP/LLM pipeline for INSW Contact Centre transcripts"
    )
    parser.add_argument(
        "--db", type=str, default="./database_log.db",
        help="Path to SQLite database file (output from asr_pipeline.py)"
    )
    args = parser.parse_args()

    run_pipeline(db_path=args.db)
