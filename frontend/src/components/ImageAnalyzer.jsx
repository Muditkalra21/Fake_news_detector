/**
 * ImageAnalyzer.jsx
 * Drag-and-drop / click-to-upload image panel.
 */
import { useState, useRef } from 'react';
import { analyzeImage } from '../api';
import ResultCard from './ResultCard';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
const MAX_MB = 10;

export default function ImageAnalyzer() {
  const [file, setFile]         = useState(null);
  const [preview, setPreview]   = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);
  const inputRef = useRef();

  const handleFile = (f) => {
    setError(null);
    setResult(null);

    if (!ACCEPTED_TYPES.includes(f.type)) {
      setError('Unsupported file type. Please upload a JPEG, PNG, WebP, or BMP image.');
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`File too large. Maximum size is ${MAX_MB} MB.`);
      return;
    }

    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const handleAnalyze = async () => {
    if (!file) { setError('Please select an image first.'); return; }
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const data = await analyzeImage(file);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Analysis failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div>
      {/* Drop zone */}
      {!preview ? (
        <div
          className={`dropzone ${dragging ? 'active' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <div style={{ fontSize: '3rem', marginBottom: '0.75rem' }}>🖼️</div>
          <p style={{ color: '#f1f5f9', fontWeight: 600, marginBottom: '0.35rem' }}>
            Drop an image here or click to browse
          </p>
          <p style={{ color: '#64748b', fontSize: '0.85rem' }}>
            JPEG, PNG, WebP, BMP · Max {MAX_MB} MB
          </p>
        </div>
      ) : (
        /* Preview */
        <div style={{ position: 'relative', borderRadius: '0.75rem', overflow: 'hidden', marginBottom: '1rem' }}>
          <img
            src={preview}
            alt="Upload preview"
            style={{ width: '100%', maxHeight: 340, objectFit: 'contain', background: 'rgba(0,0,0,0.4)', display: 'block' }}
          />
          <button
            onClick={handleClear}
            style={{
              position: 'absolute', top: 10, right: 10,
              background: 'rgba(0,0,0,0.7)', color: '#fff',
              border: 'none', borderRadius: '0.5rem',
              padding: '0.3rem 0.7rem', cursor: 'pointer', fontSize: '0.82rem',
            }}
          >
            ✕ Remove
          </button>
          <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '0.5rem 0.75rem', background: 'rgba(0,0,0,0.6)', fontSize: '0.8rem', color: '#94a3b8' }}>
            {file?.name} — {(file?.size / 1024).toFixed(1)} KB
          </div>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(',')}
        style={{ display: 'none' }}
        onChange={(e) => { if (e.target.files[0]) handleFile(e.target.files[0]); }}
      />

      {/* Error */}
      {error && (
        <div style={{ padding: '0.75rem 1rem', borderRadius: '0.6rem', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)', color: '#fca5a5', fontSize: '0.9rem', marginBottom: '1rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Analyse button */}
      {file && (
        <button className="btn-primary" style={{ marginTop: '1rem' }} onClick={handleAnalyze} disabled={loading}>
          {loading ? (
            <><span className="spinner" /> Analysing image…</>
          ) : (
            <><span>🔍</span> Analyse Image</>
          )}
        </button>
      )}

      <ResultCard result={result} />
    </div>
  );
}
