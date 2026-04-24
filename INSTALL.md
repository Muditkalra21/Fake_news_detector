# 🛠️ TruthLens — Installation Guide

## Prerequisites

Make sure the following are installed on your machine before continuing:

| Tool | Version | Download |
|---|---|---|
| **Python** | 3.10 or 3.11 | [python.org](https://www.python.org/downloads/) |
| **Node.js** | 18+ | [nodejs.org](https://nodejs.org/) |
| **Git** | any | [git-scm.com](https://git-scm.com/) |

---

## 1. Clone the Repository

```bash
git clone https://github.com/yagya2324/fake_news_detection.git
cd fake_news_detection
```

---

## 2. Backend Setup (FastAPI)

### 2a. Create a virtual environment

```bash
cd backend

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2b. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ `torch` and `transformers` are large (~500 MB). This may take a few minutes on first install.

### 2c. Set up environment variables

```bash
# Copy the example file
cp .env.example .env
```

Open `.env` and fill in your keys *(all are optional for local testing)*:

```env
# Optional: enables live fact-check cross-referencing
GOOGLE_FACTCHECK_API_KEY=your_key_here

# Optional: enables additional news source checks
NEWS_API_KEY=your_key_here
```

### 2d. Start the backend

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
- **API root:** http://127.0.0.1:8000
- **Swagger docs:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

> 📝 The first startup downloads the AI model (~17 MB). Subsequent starts are instant.

---

## 3. Frontend Setup (React + Vite)

Open a **new terminal** (keep the backend running):

### 3a. Install Node dependencies

```bash
cd frontend
npm install
```

### 3b. Set up environment variables

Create a `.env.local` file inside the `frontend/` folder:

```bash
# frontend/.env.local

# Leave blank for local dev — Vite proxy handles /api → localhost:8000
# VITE_API_URL=
```

> For production (Vercel), set `VITE_API_URL=https://your-render-backend.onrender.com`

### 3c. Start the frontend

```bash
npm run dev
```

The app will open at → **http://localhost:5173**

---

## 4. Running Both at Once (optional)

You can use two terminals side by side:

| Terminal 1 — Backend | Terminal 2 — Frontend |
|---|---|
| `cd backend` | `cd frontend` |
| `venv\Scripts\activate` (Windows) | `npm install` *(first time only)* |
| `uvicorn app.main:app --reload` | `npm run dev` |

---

## 5. Project Structure

```
fake_news_detection/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── routers/
│   │   │   └── analysis.py      # API endpoints
│   │   ├── services/
│   │   │   ├── text_analyzer.py   # AI text analysis
│   │   │   ├── image_analyzer.py  # Image manipulation detection
│   │   │   ├── video_analyzer.py  # Video URL credibility
│   │   │   └── language_detector.py
│   │   └── models/
│   │       └── schemas.py       # Pydantic models
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app shell
│   │   ├── api.js               # Axios API client
│   │   └── components/
│   │       ├── TextAnalyzer.jsx
│   │       ├── ImageAnalyzer.jsx
│   │       ├── VideoAnalyzer.jsx
│   │       ├── ResultCard.jsx
│   │       └── CredibilityRing.jsx
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── render.yaml                  # Render deployment config
└── INSTALL.md                   # This file
```

---

## 6. Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'app'`
Make sure you're running `uvicorn` from **inside the `backend/` folder**, not the root.

### ❌ CORS error in browser console
The backend must be running on port `8000`. Check that `uvicorn` is active and visit http://127.0.0.1:8000/health — it should return `{"status":"ok"}`.

### ❌ Frontend shows "Analysis failed. Is the backend running?"
Backend is not running or crashed. Check the backend terminal for errors.

### ❌ Model download fails / no internet
The app automatically falls back to a **rule-based heuristic mode** — it still works, just with lower accuracy. Ensure you have internet on first run so the model can cache.

### ❌ `torch` install fails on Windows
Try installing the CPU-only version explicitly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### ❌ `npm install` fails
Make sure Node.js 18+ is installed: `node --version`. Then delete `node_modules/` and run `npm install` again.

---

## 7. Building for Production

```bash
# Frontend — generates dist/ folder
cd frontend
npm run build

# Backend — no build step needed; deploy directly to Render
```

See [README.md](./README.md) for full deployment instructions on **Vercel** (frontend) + **Render** (backend).
