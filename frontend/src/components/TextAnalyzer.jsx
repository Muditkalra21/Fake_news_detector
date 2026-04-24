/**
 * TextAnalyzer.jsx
 * Panel for pasting / typing news text and analysing it.
 */
import { useState } from 'react';
import { analyzeText } from '../api';
import ResultCard from './ResultCard';

const PLACEHOLDER = `Paste a news article, social media post, or any text content here...

Example: "Scientists confirm that drinking lemon water cures cancer — the secret they don't want you to know!"`;

// Sample texts for demo purposes
const DEMO_TEXTS = [
  {
    label: "🔴 Likely Fake",
    text: "BREAKING: Scientists discover SHOCKING secret cure for all diseases that governments are hiding! You won't believe this exclusive report. Deep state operatives are suppressing this 100% proven natural remedy. Share before they delete this!"
  },
  {
    label: "🟡 Potentially Misleading",
    text: "New study shows coffee can cause heart problems in some patients. Out of context: the study only involved people who drank 15+ cups per day. Mainstream media won't tell you the whole story."
  },
  {
    label: "🟢 Likely Real",
    text: "According to a peer-reviewed study published in the New England Journal of Medicine, researchers found that regular exercise reduces cardiovascular disease risk by 30%. The official statement from WHO confirms these findings align with existing guidelines."
  }
];

export default function TextAnalyzer() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim() || text.trim().length < 20) {
      setError('Please enter at least 20 characters of text.');
      return;
    }
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const data = await analyzeText(text);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Analysis failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const handleDemo = (demoText) => {
    setText(demoText);
    setResult(null);
    setError(null);
  };

  return (
    <div>
      {/* Demo buttons */}
      <div style={{ marginBottom: '1rem' }}>
        <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.5rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Try a demo:
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {DEMO_TEXTS.map((d, i) => (
            <button
              key={i}
              onClick={() => handleDemo(d.text)}
              style={{
                padding: '0.35rem 0.9rem',
                borderRadius: '0.5rem',
                border: '1px solid rgba(255,255,255,0.1)',
                background: 'rgba(255,255,255,0.04)',
                color: '#cbd5e1',
                fontSize: '0.82rem',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
              onMouseEnter={e => e.target.style.borderColor = 'rgba(59,130,246,0.4)'}
              onMouseLeave={e => e.target.style.borderColor = 'rgba(255,255,255,0.1)'}
            >
              {d.label}
            </button>
          ))}
        </div>
      </div>

      {/* Text input */}
      <textarea
        className="input-field"
        rows={8}
        placeholder={PLACEHOLDER}
        value={text}
        onChange={(e) => { setText(e.target.value); setError(null); }}
      />

      {/* Character count */}
      <p style={{ textAlign: 'right', fontSize: '0.78rem', color: '#475569', marginTop: '0.4rem', marginBottom: '1rem' }}>
        {text.length} characters
      </p>

      {/* Error */}
      {error && (
        <div style={{ padding: '0.75rem 1rem', borderRadius: '0.6rem', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)', color: '#fca5a5', fontSize: '0.9rem', marginBottom: '1rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Analyse button */}
      <button className="btn-primary pulse-glow" onClick={handleAnalyze} disabled={loading}>
        {loading ? (
          <><span className="spinner" /> Analysing…</>
        ) : (
          <><span>🔍</span> Analyse Text</>
        )}
      </button>

      {/* Result */}
      <ResultCard result={result} />
    </div>
  );
}
