# SummitAI – AI that helps you “summit” your meetings with clarity.

**AI-Powered Meeting Transcription & Summarization Platform**

SummitAI is an end-to-end meeting intelligence system that automatically **transcribes**, **analyzes**, and **summarizes** meetings.
It leverages **Whisper** for speech recognition, **Gemini** for abstractive summarization, and a **FastAPI** backend integrated with **MongoDB** and **Tebi** for efficient data storage and retrieval.

---

## 🚀 Features

* 🎧 **Automatic Speech Recognition (ASR)** using [OpenAI Whisper](https://github.com/openai/whisper)
* 🧠 **Summarization** powered by **Google Gemini** for concise, context-aware meeting overviews
* ⚡ **Audio Preprocessing** with [FFmpeg](https://ffmpeg.org/) for robust media handling
* 🧩 **FastAPI Backend** for RESTful endpoints and async data processing
* 🗃️ **MongoDB + Tebi Integration** for scalable storage of transcripts, summaries, and metadata
* 📊 **Performance Benchmarking** utilities for ASR and summarization workloads

---

## 🏗️ System Architecture

```
                    ┌──────────────────┐
                    │   Audio Input    │
                    │ (.wav/.mp3 file) │
                    └───────┬──────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   FFmpeg     │
                     │ Audio Preproc│
                     └───────┬──────┘
                             │
                             ▼
                     ┌──────────────┐
                     │   Whisper    │
                     │  Transcribe  │
                     └───────┬──────┘
                             │
                             ▼
                     ┌──────────────┐
                     │   Gemini     │
                     │ Summarization│
                     └───────┬──────┘
                             │
                             ▼
                     ┌──────────────────┐
                     │ FastAPI + Mongo  │
                     │  + Tebi Backend  │
                     └──────────────────┘
```

---

## ⚙️ Tech Stack

| Component            | Technology            |
| -------------------- | --------------------- |
| **ASR**              | OpenAI Whisper        |
| **Summarization**    | Google Gemini API     |
| **Audio Processing** | FFmpeg                |
| **Backend**          | FastAPI               |
| **Database**         | MongoDB, Tebi         |
| **Benchmarking**     | Custom Python Scripts |

---

## 🧪 Benchmark Results

Tested on a **1-minute WAV file** processed end-to-end.

| Metric                  | Value / Range                |
| ----------------------- | ---------------------------- |
| **Avg CPU Usage**       | ~49.8%                       |
| **Peak RAM Usage**      | ~64.1%                       |
| **Disk Read (MB)**      | Negligible (~0.08 MB total)  |
| **Disk Write (MB)**     | ~15 MB (peaks around 5 MB/s) |
| **Avg Processing Time** | ~25 seconds for 1 min audio  |
| **Throughput**          | ~2.4× real-time              |

**Interpretation:**
SummitAI maintains a stable CPU footprint (~50%) and consistent memory usage (~64%), with minimal disk I/O. This indicates efficient streaming and in-memory processing suitable for scalable multi-session workloads.

---

## 🧰 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/prabhsuratsingh/SummitAI.git
cd SummitAI
```

### 2. Build Docker container

```bash
docker build -t server .
```

### 3. Run the Container

```bash
docker run -p 8000:8000 server
```

---

## 📈 Future Roadmap

* 🔊 Real-time streaming ASR pipeline
* 🗣️ Speaker diarization and emotion tagging
* 📅 Meeting analytics dashboard (insights, action items)
* ☁️ Multi-cloud deployment (Tebi, GCP, AWS)

---

## 🧑‍💻 Author

**Prabhsurat Singh**
[Linkedin: @prabhsuratsingh](www.linkedin.com/in/prabhsurat-singh-1868052ab)
