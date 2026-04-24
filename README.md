# 🛡️ TruthLens — AI-Powered Fake News Detection System

> Analyse news articles, images, and videos for credibility using state-of-the-art AI models.

![TruthLens](https://img.shields.io/badge/AI-Fake%20News%20Detector-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge)
![React](https://img.shields.io/badge/Frontend-React%20+%20Vite-61DAFB?style=for-the-badge)

---

## 📋 Features

| Feature | Description |
|---|---|
| 📝 **Text Analysis** | Classifies news text as REAL / FAKE / MISLEADING using HuggingFace transformers |
| 🖼️ **Image Analysis** | Detects manipulation via Error Level Analysis (ELA) + EXIF metadata inspection |
| 🎬 **Video Analysis** | Evaluates video URL credibility via domain reputation + URL pattern matching |
| 🌐 **Multilingual** | Supports 20+ languages including Hindi, Bengali, Tamil, Telugu, and more |
| 📊 **Credibility Score** | 0–100 score with animated ring + progress bar |
| 🔍 **Explainability** | Key phrases, flagged patterns, and confidence levels |
| 🔗 **Fact-check Sources** | Links to Snopes, AltNews, BoomLive, FactCheck.org, and more |

---

## 🗂️ Project Structure

```
Fake_news_detection/
├── backend/
│   ├── app/
│   │   ├── main.py              ← FastAPI app entry point
│   │   ├── models/
│   │   │   └── schemas.py       ← Pydantic request/response schemas
│   │   ├── routers/
│   │   │   └── analysis.py      ← API route handlers
│   │   └── services/
│   │       ├── text_analyzer.py    ← NLP / transformer model
│   │       ├── image_analyzer.py   ← ELA + EXIF image analysis
│   │       ├── video_analyzer.py   ← URL + domain credibility
│   │       └── language_detector.py ← langdetect wrapper
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              ← Main shell + tab navigation
│   │   ├── api.js               ← Axios API client
│   │   ├── index.css            ← Global design system
│   │   └── components/
│   │       ├── TextAnalyzer.jsx
│   │       ├── ImageAnalyzer.jsx
│   │       ├── VideoAnalyzer.jsx
│   │       ├── ResultCard.jsx
│   │       └── CredibilityRing.jsx
│   ├── index.html
│   └── vite.config.js
│
├── docs/
│   └── architecture.md
├── AGENT.md
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** — [python.org](https://python.org)
- **Node.js 18+** — [nodejs.org](https://nodejs.org)

---

### 1️⃣ Clone / open the project

```bash
cd d:\Fake_news_detection
```

---

### 2️⃣ Start the Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs auto-generated at: **http://127.0.0.1:8000/docs**

> **Note:** On first run, the HuggingFace model (`facebook/bart-large-mnli`) will be downloaded (~1.6 GB). Subsequent starts are instant.  
> The backend gracefully falls back to rule-based analysis if the model cannot be loaded.

---

### 3️⃣ Start the Frontend (React + Vite)

```bash
cd frontend
npm install        # only needed once
npm run dev
```

Open: **http://localhost:5173**

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI (interactive) |
| `POST` | `/api/analyze/text` | Analyse news text |
| `POST` | `/api/analyze/image` | Analyse uploaded image |
| `POST` | `/api/analyze/video` | Analyse video URL |

### Text Analysis — Request body
```json
{ "text": "Paste your news content here..." }
```

### Image Analysis — multipart/form-data
```
file: <binary image>
```

### Video Analysis — Request body
```json
{ "url": "https://...", "description": "optional caption" }
```

### Response (all endpoints)
```json
{
  "label": "FAKE",
  "confidence": 0.87,
  "credibility_score": 12,
  "explanation": "AI model classified this as 'FAKE' with 87.0% confidence...",
  "key_phrases": ["…SHOCKING secret…", "…they don't want you to know…"],
  "sources": ["https://www.snopes.com", "https://www.factcheck.org"],
  "detected_language": "English"
}
```

---

## 🧠 How the AI Works

### Text Analysis
1. Uses **`facebook/bart-large-mnli`** (zero-shot classification) via HuggingFace Transformers
2. Classifies into: `real news`, `fake news`, `misleading content`
3. Augmented with regex-based misinformation pattern detection
4. Falls back to pure rule-based heuristics if model is unavailable

### Image Analysis
1. **Error Level Analysis (ELA)** — detects JPEG compression inconsistencies from editing
2. **EXIF Metadata** — checks for editing software (Photoshop, GIMP, etc.)
3. **Edge Density** — high edge density can indicate compositing/pasting
4. Combines signals into a manipulation score → REAL / MISLEADING / FAKE

### Video Analysis
1. **Domain Reputation** — cross-checks against curated trusted/suspicious domain lists
2. **URL Pattern Matching** — detects clickbait and sensationalist URL structures
3. **Description NLP** — runs text analysis on the video caption if provided
4. Trust score determines verdict

---

## 🌐 Multilingual Support

Supported languages include:

| Language | Code | Language | Code |
|---|---|---|---|
| English | `en` | Hindi | `hi` |
| Bengali | `bn` | Telugu | `te` |
| Marathi | `mr` | Tamil | `ta` |
| Urdu | `ur` | Gujarati | `gu` |
| Kannada | `kn` | Malayalam | `ml` |
| Punjabi | `pa` | Arabic | `ar` |
| French | `fr` | Spanish | `es` |

---

## 🔐 Environment Variables

Copy `backend/.env.example` to `backend/.env`:

```bash
GOOGLE_FACTCHECK_API_KEY=your_key   # Optional
NEWS_API_KEY=your_key               # Optional
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + Tailwind CSS v4 |
| Backend | FastAPI (Python) + Uvicorn |
| NLP Model | HuggingFace Transformers (BART-MNLI) |
| Image Processing | Pillow (PIL) |
| Language Detection | langdetect |
| HTTP Client | Axios |

---

## ⚠️ Disclaimer

TruthLens provides AI-generated assessments. Always cross-verify important information with official and trusted news sources. This tool is designed to assist — not replace — human judgement.

---

## 📄 License

MIT License — free to use, modify, and distribute.
