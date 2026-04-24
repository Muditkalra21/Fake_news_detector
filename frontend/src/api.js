/**
 * api.js — Axios instance pre-configured for the FastAPI backend.
 * All components import from here so the base URL is set in one place.
 */
import axios from 'axios';

// In production (Vercel), set VITE_API_URL to your Render backend URL.
// In development, Vite proxy rewrites /api → http://localhost:8000/api.
const BASE_URL = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60_000,          // 60s — model inference can take a moment
});

// Attach request timing for debug logging
api.interceptors.request.use((config) => {
  config.metadata = { startTime: Date.now() };
  return config;
});

api.interceptors.response.use(
  (response) => {
    const ms = Date.now() - (response.config.metadata?.startTime ?? Date.now());
    console.debug(`[API] ${response.config.method?.toUpperCase()} ${response.config.url} → ${response.status} (${ms}ms)`);
    return response;
  },
  (error) => {
    const msg =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred.';
    return Promise.reject(new Error(msg));
  }
);

/* ── Typed API helpers ─────────────────────────────────────────────── */

/**
 * Analyse a news text string.
 * @param {string} text
 * @returns {Promise<AnalysisResult>}
 */
export const analyzeText = (text) =>
  api.post('/analyze/text', { text }).then((r) => r.data);

/**
 * Analyse an uploaded image File object.
 * @param {File} file
 * @returns {Promise<AnalysisResult>}
 */
export const analyzeImage = (file) => {
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
export const analyzeVideo = (url, description = '') =>
  api.post('/analyze/video', { url, description }).then((r) => r.data);

export default api;
