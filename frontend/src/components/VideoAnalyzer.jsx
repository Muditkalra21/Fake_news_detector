/**
 * VideoAnalyzer.jsx
 * Video URL input panel for credibility analysis.
 */
import { useState } from 'react';
import { analyzeVideo } from '../api';
import ResultCard from './ResultCard';

const DEMO_URLS = [
  { label: '🟢 BBC News', url: 'https://www.bbc.com/news/video' },
  { label: '🔴 Suspicious', url: 'https://beforeitsnews.com/watch?v=shocking-exclusive-you-wont-believe' },
  { label: '🟡 YouTube', url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' },
];

export default function VideoAnalyzer() {
  const [url, setUrl]                 = useState('');
  const [description, setDescription] = useState('');
  const [loading, setLoading]         = useState(false);
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState(null);

  const handleAnalyze = async () => {
    const trimmed = url.trim();
    if (!trimmed) { setError('Please enter a video URL.'); return; }
    if (!trimmed.startsWith('http://') && !trimmed.startsWith('https://')) {
      setError('URL must start with http:// or https://'); return;
    }
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const data = await analyzeVideo(trimmed, description);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Analysis failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Demo quick-fill */}
      <div style={{ marginBottom: '1rem' }}>
        <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.5rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Try a demo:
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {DEMO_URLS.map((d, i) => (
            <button
              key={i}
              onClick={() => { setUrl(d.url); setError(null); setResult(null); }}
              style={{
                padding: '0.35rem 0.9rem', borderRadius: '0.5rem',
                border: '1px solid rgba(255,255,255,0.1)',
                background: 'rgba(255,255,255,0.04)',
                color: '#cbd5e1', fontSize: '0.82rem', cursor: 'pointer', transition: 'all 0.2s',
              }}
              onMouseEnter={e => e.target.style.borderColor = 'rgba(59,130,246,0.4)'}
              onMouseLeave={e => e.target.style.borderColor = 'rgba(255,255,255,0.1)'}
            >
              {d.label}
            </button>
          ))}
        </div>
      </div>

      {/* URL input */}
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.4rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Video URL
        </label>
        <div style={{ position: 'relative' }}>
          <span style={{ position: 'absolute', left: '0.9rem', top: '50%', transform: 'translateY(-50%)', fontSize: '1.1rem' }}>🎬</span>
          <input
            type="url"
            className="input-field"
            placeholder="https://www.youtube.com/watch?v=..."
            value={url}
            onChange={(e) => { setUrl(e.target.value); setError(null); }}
            style={{ paddingLeft: '2.5rem' }}
          />
        </div>
      </div>

      {/* Optional description */}
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.4rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Description / Caption <span style={{ color: '#475569', fontWeight: 400 }}>(optional — improves accuracy)</span>
        </label>
        <textarea
          className="input-field"
          rows={3}
          placeholder="Paste the video's title, caption, or description here for better analysis…"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
      </div>

      {/* Error */}
      {error && (
        <div style={{ padding: '0.75rem 1rem', borderRadius: '0.6rem', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)', color: '#fca5a5', fontSize: '0.9rem', marginBottom: '1rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Analyse button */}
      <button className="btn-primary" onClick={handleAnalyze} disabled={loading}>
        {loading ? (
          <><span className="spinner" /> Analysing…</>
        ) : (
          <><span>🔍</span> Analyse Video</>
        )}
      </button>

      <ResultCard result={result} />
    </div>
  );
}
