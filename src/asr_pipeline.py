"""
asr_pipeline.py
---------------
Automatic Speech Recognition (ASR) pipeline for INSW Contact Centre voice logs.
Uses Faster-Whisper for Indonesian-language transcription.

Reference:
    Cahyana, W. et al. "Automatic Speech Recognition for Voice Log Analysis:
    A Case Study in the Indonesia National Single Window Contact Center."
    Binus Online, 2025.

Usage:
    python asr_pipeline.py --audio_dir ./audio --db ./database_log.db
"""

import os
import shutil
import sqlite3
import argparse
from datetime import datetime
from faster_whisper import WhisperModel


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_SIZE = "small"       # Options: tiny, base, small, medium, large-v2
DEVICE = "cpu"             # Use "cuda" if GPU is available
COMPUTE_TYPE = "int8"      # int8 for CPU efficiency; float16 for GPU
LANGUAGE = "id"            # Indonesian
PROCESSED_DIR = "./processed"
AUDIO_EXTENSIONS = (".wav", ".mp3")


# ── Database ───────────────────────────────────────────────────────────────────

def init_db(db_path: str):
    """Initialise SQLite database and create table if not exists."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hasil_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            nama_agent  TEXT,
            uraian      TEXT,
            id_audio    TEXT UNIQUE,
            tanggal     TEXT,
            finished    INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(db_path: str, records: list):
    """Save transcription results to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR IGNORE INTO hasil_log (nama_agent, uraian, id_audio, tanggal, finished)
        VALUES (?, ?, ?, ?, 0)
    """, records)
    conn.commit()
    conn.close()


# ── Audio Utilities ────────────────────────────────────────────────────────────

def get_audio_files(audio_dir: str) -> list:
    """Return list of audio file paths from the given directory."""
    files = [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.lower().endswith(AUDIO_EXTENSIONS)
    ]
    return sorted(files)


def move_to_processed(file_path: str):
    """Move a processed audio file to the processed directory."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filename = os.path.basename(file_path)
    dest = os.path.join(PROCESSED_DIR, filename)
    shutil.move(file_path, dest)
    print(f"  → Moved to processed: {filename}")


def extract_agent_name(filename: str) -> str:
    """
    Extract agent name from filename convention.
    Expected format: AGENTNAME_YYYYMMDD_HHMMSS.wav
    Falls back to 'unknown' if pattern doesn't match.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    return parts[0] if parts else "unknown"


# ── ASR Core ───────────────────────────────────────────────────────────────────

def load_model() -> WhisperModel:
    """Load and return the Faster-Whisper model."""
    print(f"Loading Faster-Whisper model: {MODEL_SIZE} ({DEVICE}, {COMPUTE_TYPE})")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Model loaded.\n")
    return model


def transcribe(model: WhisperModel, audio_path: str) -> str:
    """
    Transcribe a single audio file to Indonesian text.

    Returns:
        Full transcript as a single string.
    """
    segments, info = model.transcribe(audio_path, language=LANGUAGE)
    transcript = ""
    for segment in segments:
        transcript += f"{segment.text.strip()}\n"
    return transcript.strip()


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(audio_dir: str, db_path: str):
    """
    Main ASR pipeline:
    1. Load model
    2. Scan audio directory
    3. Transcribe each file
    4. Save results to database
    5. Move processed files
    """
    init_db(db_path)
    model = load_model()

    audio_files = get_audio_files(audio_dir)
    if not audio_files:
        print(f"No audio files found in: {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio file(s) to process.\n")
    records = []

    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        id_audio = os.path.splitext(filename)[0]
        nama_agent = extract_agent_name(filename)
        tanggal = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"Processing: {filename}")
        try:
            transcript = transcribe(model, audio_path)
            records.append((nama_agent, transcript, id_audio, tanggal))
            print(f"  ✓ Transcription complete ({len(transcript.split())} words)")
            move_to_processed(audio_path)
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")

    if records:
        save_to_db(db_path, records)
        print(f"\n✓ Saved {len(records)} record(s) to database: {db_path}")
        print("  Status: finished = 0 (awaiting NLP/LLM processing)")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASR pipeline for INSW Contact Centre voice logs"
    )
    parser.add_argument(
        "--audio_dir", type=str, default="./audio",
        help="Directory containing input audio files (.wav/.mp3)"
    )
    parser.add_argument(
        "--db", type=str, default="./database_log.db",
        help="Path to SQLite database file"
    )
    args = parser.parse_args()

    run_pipeline(audio_dir=args.audio_dir, db_path=args.db)
