/**
 * api.js — Axios instance pre-configured for the FastAPI backend.
 * All components import from here so the base URL is set in one place.
 */
import axios from 'axios';

// ── Startup debug info ─────────────────────────────────────────────────────
const VITE_API_URL = import.meta.env.VITE_API_URL;
const BASE_URL = VITE_API_URL ? `${VITE_API_URL}/api` : '/api';

console.group('%c[TruthLens] API Config', 'color: #3b82f6; font-weight: bold;');
console.log('VITE_API_URL env var:', VITE_API_URL || '(not set — using Vite proxy /api)');
console.log('Resolved BASE_URL:   ', BASE_URL);
console.groupEnd();

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60_000,  // 60s — model inference can take a moment
});

// ── Request interceptor ────────────────────────────────────────────────────
api.interceptors.request.use((config) => {
  config.metadata = { startTime: Date.now() };
  console.group(`%c[API →] ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`, 'color: #f59e0b; font-weight: bold;');
  console.log('Payload:', config.data);
  console.groupEnd();
  return config;
});

// ── Response interceptor ───────────────────────────────────────────────────
api.interceptors.response.use(
  (response) => {
    const ms = Date.now() - (response.config.metadata?.startTime ?? Date.now());
    console.group(`%c[API ✓] ${response.config.method?.toUpperCase()} ${response.config.url} → ${response.status} (${ms}ms)`, 'color: #22c55e; font-weight: bold;');
    console.log('Response data:  ', response.data);
    console.log('Label:          ', response.data?.label);
    console.log('Confidence:     ', response.data?.confidence);
    console.log('Credibility:    ', response.data?.credibility_score);
    console.log('Language:       ', response.data?.detected_language);
    console.groupEnd();
    return response;
  },
  (error) => {
    const status = error.response?.status;
    const msg =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred.';
    console.group('%c[API ✗] Request failed', 'color: #ef4444; font-weight: bold;');
    console.log('Status:', status);
    console.log('Error message:', msg);
    console.log('Full error:', error.response?.data ?? error.message);
    console.groupEnd();
    return Promise.reject(new Error(msg));
  }
);

/* ── Typed API helpers ─────────────────────────────────────────────── */

/**
 * Analyse a news text string.
 * @param {string} text
 * @returns {Promise<AnalysisResult>}
 */
export const analyzeText = (text) => {
  console.log(`%c[TruthLens] Sending text for analysis (${text.length} chars)`, 'color: #8b5cf6;');
  return api.post('/analyze/text', { text }).then((r) => r.data);
};

/**
 * Analyse an uploaded image File object.
 * @param {File} file
 * @returns {Promise<AnalysisResult>}
 */
export const analyzeImage = (file) => {
  console.log(`%c[TruthLens] Sending image for analysis: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`, 'color: #8b5cf6;');
  const form = new FormData();
  form.append('file', file);
  return api.post('/analyze/image', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then((r) => r.data);
};

/**
 * Analyse a video URL.
 * @param {string} url
 * @param {string} [description]
 * @returns {Promise<AnalysisResult>}
 */
export const analyzeVideo = (url, description = '') => {
  console.log(`%c[TruthLens] Sending video URL for analysis: ${url}`, 'color: #8b5cf6;');
  return api.post('/analyze/video', { url, description }).then((r) => r.data);
};

export default api;
