# Architecture — TruthLens Fake News Detection System

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER BROWSER                            │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │             React Frontend (Vite)                    │  │
│  │                                                      │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │  │
│  │  │   Text   │  │  Image   │  │      Video       │   │  │
│  │  │ Analyzer │  │ Analyzer │  │     Analyzer     │   │  │
│  │  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │  │
│  │       │             │                  │             │  │
│  │       └─────────────┴──────────────────┘             │  │
│  │                        │                             │  │
│  │               api.js (Axios)                         │  │
│  └────────────────────────┼─────────────────────────────┘  │
└───────────────────────────┼─────────────────────────────────┘
                            │ HTTP/REST (JSON + multipart)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI Backend (Python)                      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               routers/analysis.py                   │  │
│  │                                                      │  │
│  │  POST /api/analyze/text                              │  │
│  │  POST /api/analyze/image                             │  │
│  │  POST /api/analyze/video                             │  │
│  └──┬─────────────────────┬──────────────────────┬─────┘  │
│     │                     │                      │         │
│     ▼                     ▼                      ▼         │
│  ┌──────────┐      ┌──────────┐          ┌───────────┐    │
│  │  text_   │      │  image_  │          │  video_   │    │
│  │ analyzer │      │ analyzer │          │  analyzer │    │
│  └──┬───────┘      └──┬───────┘          └─────┬─────┘    │
│     │                 │                        │           │
│     │                 │                        │           │
│     ▼                 ▼                        ▼           │
│  HuggingFace       PIL/ELA               Domain DB +       │
│  BART-MNLI         EXIF                  URL Patterns       │
│  + Heuristics      Analysis                                 │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         language_detector.py (langdetect)            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Request / Response Flow

1. User submits content (text / image / video URL) via React UI
2. Axios sends request to `/api/analyze/{type}`
3. FastAPI router validates input (Pydantic schemas)
4. Service layer processes content and returns a structured result
5. Result includes: `label`, `confidence`, `credibility_score`, `explanation`, `key_phrases`, `sources`, `detected_language`
6. React renders `ResultCard` with animated credibility ring and progress bars

## Credibility Score Formula

| Label | Base Score | Modifiers |
|---|---|---|
| REAL | 50 + confidence×50 | +2 per credible signal, -3 per fake pattern |
| MISLEADING | 50 - confidence×30 | Capped 20–70 |
| FAKE | 50 - confidence×45 | Capped 0–45 |

## Data Flow Diagram

```
Input Text
  │
  ├─► Language Detection (langdetect)
  │         └─► ISO code → human name
  │
  ├─► Zero-Shot Classification (BART-MNLI)
  │         └─► label + confidence scores
  │
  ├─► Regex Pattern Matching
  │         ├─► Fake patterns (16 patterns)
  │         ├─► Credible signals (7 patterns)
  │         └─► Misleading patterns (5 patterns)
  │
  └─► Score Aggregation
            └─► AnalysisResult JSON
```

## Scalability Notes

- The FastAPI backend is async and can handle concurrent requests
- The transformer model is loaded once at startup (singleton pattern)
- For production: wrap in Docker, add Redis cache for repeated URLs, use GPU inference
- MongoDB can be added to log all analysis requests for trend monitoring
