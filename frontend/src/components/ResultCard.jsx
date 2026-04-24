/**
 * ResultCard.jsx
 * Displays the full analysis result:
 *  - Verdict badge + credibility ring
 *  - Confidence progress bar
 *  - Explanation text
 *  - Key phrases (explainability)
 *  - Detected language
 *  - Source links
 */
import CredibilityRing from './CredibilityRing';

const BADGE_CLASS = {
  REAL:       'badge badge-real',
  FAKE:       'badge badge-fake',
  MISLEADING: 'badge badge-mislead',
  UNKNOWN:    'badge badge-unknown',
};

const VERDICT_ICON = {
  REAL:       '✅',
  FAKE:       '❌',
  MISLEADING: '⚠️',
  UNKNOWN:    '❓',
};

const BAR_COLOR = {
  REAL:       'linear-gradient(90deg,#16a34a,#22c55e)',
  FAKE:       'linear-gradient(90deg,#b91c1c,#ef4444)',
  MISLEADING: 'linear-gradient(90deg,#b45309,#f59e0b)',
  UNKNOWN:    'linear-gradient(90deg,#4b5563,#6b7280)',
};

export default function ResultCard({ result }) {
  if (!result) return null;

  const {
    label = 'UNKNOWN',
    confidence = 0,
    credibility_score = 0,
    explanation = '',
    key_phrases = [],
    sources = [],
    detected_language,
  } = result;

  const confPct = Math.round(confidence * 100);

  return (
    <div className="glass-card slide-up" style={{ padding: '2rem', marginTop: '1.5rem' }}>

      {/* ── Top row: ring + verdict ── */}
      <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <CredibilityRing score={credibility_score} label={label} />

        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
            <span className={BADGE_CLASS[label] ?? 'badge badge-unknown'}>
              {VERDICT_ICON[label]} {label}
            </span>
            {detected_language && (
              <span style={{ fontSize: '0.78rem', color: '#94a3b8', padding: '0.2rem 0.6rem', background: 'rgba(255,255,255,0.05)', borderRadius: '0.4rem' }}>
                🌐 {detected_language}
              </span>
            )}
          </div>

          <p style={{ fontSize: '0.82rem', color: '#94a3b8', marginBottom: '0.5rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Credibility Score
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
            <div className="progress-bar-wrap" style={{ flex: 1 }}>
              <div
                className="progress-bar-fill"
                style={{ width: `${credibility_score}%`, background: BAR_COLOR[label] }}
              />
            </div>
            <span style={{ fontWeight: 700, fontSize: '0.9rem', color: '#f1f5f9', minWidth: 36 }}>
              {credibility_score}%
            </span>
          </div>

          <p style={{ fontSize: '0.82rem', color: '#94a3b8', marginBottom: '0.35rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Confidence
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div className="progress-bar-wrap" style={{ flex: 1 }}>
              <div
                className="progress-bar-fill"
                style={{ width: `${confPct}%`, background: 'linear-gradient(90deg,#3b82f6,#8b5cf6)' }}
              />
            </div>
            <span style={{ fontWeight: 700, fontSize: '0.9rem', color: '#f1f5f9', minWidth: 36 }}>
              {confPct}%
            </span>
          </div>
        </div>
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid rgba(255,255,255,0.07)', margin: '1.5rem 0' }} />

      {/* ── Explanation ── */}
      <div style={{ marginBottom: '1.25rem' }}>
        <h3 style={{ fontSize: '0.8rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '0.5rem' }}>
          🔍 Why this verdict?
        </h3>
        <p style={{ color: '#cbd5e1', fontSize: '0.95rem', lineHeight: 1.7 }}>{explanation}</p>
      </div>

      {/* ── Key phrases ── */}
      {key_phrases.length > 0 && (
        <div style={{ marginBottom: '1.25rem' }}>
          <h3 style={{ fontSize: '0.8rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '0.6rem' }}>
            🔖 Key Observations
          </h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {key_phrases.map((phrase, i) => (
              <span key={i} className="phrase-chip">{phrase}</span>
            ))}
          </div>
        </div>
      )}

      {/* ── Sources ── */}
      {sources.length > 0 && (
        <div>
          <h3 style={{ fontSize: '0.8rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '0.6rem' }}>
            🔗 Verify with trusted sources
          </h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {sources.map((src, i) => {
              let host;
              try { host = new URL(src).hostname.replace('www.', ''); } catch { host = src; }
              return (
                <a key={i} href={src} target="_blank" rel="noopener noreferrer" className="source-link">
                  ↗ {host}
                </a>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
