# asr-nlp-contact-centre
End-to-end ASR + NLP + LLM pipeline to analyse Indonesian-language contact centre voice logs. Integrates Faster-Whisper, GPT-4.1 Mini, Apache Airflow, and Streamlit. Published research — INSW Agency, Ministry of Finance


# 🎙️ ASR/NLP Contact Centre Intelligence
### Automatic Speech Recognition for Voice Log Analysis: A Case Study in the Indonesia National Single Window Contact Center

> **Published research paper** · Binus University, Information Systems Study Program · 2025

---

## 📄 Abstract

The INSW Contact Center recorded **13,765 service requests in March 2025**, including 3,652 incoming calls — yet all voice log data was stored only as raw audio, making it impossible to analyse at scale. This project builds an end-to-end **ASR + NLP + LLM pipeline** to automatically transcribe, classify, summarise, and visualise Indonesian-language contact centre calls, turning unstructured audio into actionable service insights.

---

## 📊 Key Results

| Metric | Score | Benchmark |
|---|---|---|
| Word Error Rate (WER) | **12.6%** | Outperforms Whisper-Base (CPU) by 9% |
| Character Error Rate (CER) | **6.4%** | — |
| Topic Classification (macro-F1) | **0.87** | Across 5 service categories |
| Summarisation ROUGE-1 | **0.79** | Strong alignment with human references |
| BERTScore | **0.83** | — |
| Inter-Annotator Agreement (IAA) | **0.91** | Topic labelling consistency |

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **ASR (Speech-to-Text)** | [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — `small` model, CPU inference, `int8` precision |
| **NLP / LLM** | GPT-4.1 Mini — topic classification, NER, summarisation |
| **Topic Modelling** | Latent Dirichlet Allocation (LDA) + few-shot GPT-4.1 Mini refinement |
| **Orchestration** | Apache Airflow (DAG-1: ingestion · DAG-2: LLM processing) |
| **Database** | PostgreSQL — staging + `hasil_log` main table |
| **Dashboard** | Streamlit — real-time filtering, trend visualisation |
| **Language** | Python |

---

## 🔄 Pipeline Architecture

```
Raw Audio (.wav/.mp3) from Vendor DB
           │
           ▼
  ┌─────────────────┐
  │  DAG-1 Airflow  │  Scheduled daily at 00:00 WIB (UTC+7)
  │  Ingestion      │  → Pulls audio → stores in PostgreSQL staging
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────┐
  │  Faster-Whisper ASR         │  model_size="small", device="cpu"
  │  Indonesian transcription   │  → saves transcript + agent name
  │  finished = 0               │    (status: ASR done, NLP pending)
  └────────┬────────────────────┘
           │
           ▼
  ┌─────────────────────────────┐
  │  DAG-2 Airflow              │
  │  GPT-4.1 Mini (NLP/LLM)    │
  │  ├─ Topic Classification    │  → eCOO / SSM Izin / SSM QC / PIB / PEB / N/A
  │  ├─ Named Entity Recog.     │  → caller name, gender, no_aju (26-digit ref)
  │  └─ Summarisation           │  → concise call intent summary
  │  finished = 1               │    (status: ready for dashboard)
  └────────┬────────────────────┘
           │
           ▼
  ┌─────────────────────────────┐
  │  Streamlit Dashboard        │
  │  ├─ Call distribution chart │
  │  ├─ Trend over time         │
  │  ├─ Call driver summary     │
  │  └─ Filter by agent/gender/ │
  │       category/date         │
  └─────────────────────────────┘
```

---

## 💻 Code Sample — ASR Stage

```python
from faster_whisper import WhisperModel

# Initialise model (CPU-optimised, int8 precision)
model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Transcribe Indonesian audio
segments, info = model.transcribe(full_path, language="id")

# Combine segments into full transcript
uraian = ""
for segment in segments:
    uraian += f"{segment.text.strip()}\n"

savetosqlite("database_log.db", [(nama_agent, uraian, id_audio)])
```

---

## 💻 Code Sample — GPT-4.1 Mini NLP Prompt

```python
prompt_summary = """Extract elemen informasi dari percakapan berikut ini:
nama agent, nama penelepon, gender penelepon, summary percakapan,
nomor car / nomor aju (26 digit),
klasifikasi (eCOO, SSM Izin, SSM QC, PIB, PEB, dan N/A jika tidak ada)

Return dalam format JSON seperti:
{
  "data": {
    "summary": "...",
    "tanggal": "...",
    "nama_penelepon": "...",
    "gender_penelepon": "...",
    "no_aju": "...",
    "klasifikasi": "..."
  }
}

Percakapan:
"""
```

---

## 📁 Dataset

- **41 anonymised voice logs** from INSW Contact Center (recorded 2022)
- Average call duration: **5–7 minutes**
- Language: **Bahasa Indonesia** (conversational, domain-specific vocabulary)
- Stored in PostgreSQL with metadata: agent ID, timestamp, call duration

> ⚠️ *Raw audio and full transcripts are not publicly available due to government data classification policy. This repository documents the methodology, architecture, and results for academic and portfolio purposes.*

---

## 🏛️ Context & Affiliation

| | |
|---|---|
| **Institution** | Indonesia National Single Window (INSW) Agency, Ministry of Finance of the Republic of Indonesia |
| **Academic Program** | Information Systems, Binus Online |
| **Research Supervisor** | Dr. Agus Putranto S.Kom., M.T., M.Sc. |
| **Technical Mentor** | Muhammad Arif Rahman Isma, Head of Data Analysis and Information Presentation Section, INSW Agency |

---

## 🔍 Service Topics Classified

| Code | Service Area |
|---|---|
| `eCOO` | Electronic Certificate of Origin |
| `SSM Izin` | Permit/Licensing via Single Submission |
| `SSM QC` | Quality Control via Single Submission |
| `PIB` | Import Customs Declaration |
| `PEB` | Export Customs Declaration |
| `N/A` | Out-of-scope / unclassified |

---

## 💡 Reflections

- **Domain vocabulary is a real challenge** — generic Whisper models struggled with customs and trade terms; language-specific tuning (`language="id"`) and the `small` model size struck the right balance between accuracy and CPU speed.
- **LLMs as structured extractors** — using GPT-4.1 Mini for JSON-format entity extraction (rather than rule-based NER) proved highly adaptable to varied conversational styles.
- **Airflow for reproducibility** — the DAG-based approach with `finished=0/1` status flags prevented duplicate processing and made the pipeline auditable, which matters in a government setting.
- **The dashboard is the real product** — the technical pipeline serves non-technical decision-makers; the Streamlit interface was as important as model accuracy.

---

## 👤 Author

**Wiguna Cahyana**
Database Administrator · INSW Agency, Ministry of Finance of Indonesia
[linkedin.com/in/wigunacahyana](https://linkedin.com/in/wigunacahyana) · wiguna.cahyana@kemenkeu.go.id
