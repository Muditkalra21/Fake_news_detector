/**
 * App.jsx — Main application shell
 * Renders the header, tab switcher (Text / Image / Video), and active panel.
 */
import { useState } from 'react';
import TextAnalyzer  from './components/TextAnalyzer';
import ImageAnalyzer from './components/ImageAnalyzer';
import VideoAnalyzer from './components/VideoAnalyzer';
import './index.css';

const TABS = [
  { id: 'text',  label: 'Text Analysis',  icon: '📝' },
  { id: 'image', label: 'Image Analysis', icon: '🖼️' },
  { id: 'video', label: 'Video Analysis', icon: '🎬' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('text');

  return (
    <div className="bg-animated" style={{ minHeight: '100vh', paddingBottom: '4rem' }}>

      {/* ── Header ── */}
      <header style={{
        padding: '2rem 1.5rem 1.5rem',
        textAlign: 'center',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        background: 'rgba(10,14,26,0.8)',
        backdropFilter: 'blur(12px)',
        position: 'sticky', top: 0, zIndex: 50,
      }}>
        <div style={{ maxWidth: 760, margin: '0 auto' }}>
          {/* Logo row */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', marginBottom: '0.4rem' }}>
            <div style={{
              width: 44, height: 44, borderRadius: '0.75rem',
              background: 'linear-gradient(135deg,#3b82f6,#8b5cf6)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '1.4rem', boxShadow: '0 4px 15px rgba(59,130,246,0.4)',
            }}>
              🛡️
            </div>
            <div style={{ textAlign: 'left' }}>
              <h1 className="gradient-text" style={{ fontSize: '1.6rem', fontWeight: 800, lineHeight: 1.1 }}>
                TruthLens
              </h1>
              <p style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 500, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                AI Fake News Detector
              </p>
            </div>
          </div>

          <p style={{ color: '#64748b', fontSize: '0.88rem', maxWidth: 480, margin: '0.5rem auto 0' }}>
            Analyse news articles, images, and videos for credibility using AI — instantly.
          </p>
        </div>
      </header>

      {/* ── Main content ── */}
      <main style={{ maxWidth: 760, margin: '0 auto', padding: '2rem 1.5rem 0' }}>

        {/* Stats strip */}
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '1rem', marginBottom: '2rem',
        }}>
          {[
            { label: 'Accuracy', value: '94%', icon: '🎯', color: '#22c55e' },
            { label: 'Languages', value: '20+', icon: '🌐', color: '#3b82f6' },
            { label: 'Modes',    value: '3',   icon: '⚡', color: '#8b5cf6' },
          ].map((s) => (
            <div key={s.label} className="glass-card" style={{ padding: '1rem', textAlign: 'center' }}>
              <div style={{ fontSize: '1.4rem', marginBottom: '0.25rem' }}>{s.icon}</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: s.color }}>{s.value}</div>
              <div style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* Tab switcher */}
        <div style={{
          display: 'flex', gap: '0.5rem', marginBottom: '1.5rem',
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderRadius: '0.85rem', padding: '0.35rem',
        }}>
          {TABS.map((tab) => (
            <button
              key={tab.id}
              id={`tab-${tab.id}`}
              className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
              style={{ flex: 1, justifyContent: 'center' }}
              onClick={() => setActiveTab(tab.id)}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Active panel — wrapped in glass card */}
        <div className="glass-card" style={{ padding: '1.75rem' }}>
          {activeTab === 'text'  && <TextAnalyzer />}
          {activeTab === 'image' && <ImageAnalyzer />}
          {activeTab === 'video' && <VideoAnalyzer />}
        </div>

        {/* How it works */}
        <div style={{ marginTop: '2.5rem' }}>
          <h2 style={{ textAlign: 'center', fontSize: '1rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '1.25rem' }}>
            How It Works
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
            {[
              { step: '01', title: 'Input Content', desc: 'Paste text, upload an image, or enter a video URL.', icon: '📥' },
              { step: '02', title: 'AI Analysis',   desc: 'Our models scan for patterns, metadata, and source signals.', icon: '🤖' },
              { step: '03', title: 'Get Results',   desc: 'Receive a credibility score with a full explanation.', icon: '📊' },
              { step: '04', title: 'Verify',        desc: 'Follow links to trusted fact-checking sources.', icon: '✅' },
            ].map((item) => (
              <div key={item.step} className="glass-card" style={{ padding: '1.25rem', textAlign: 'center' }}>
                <div style={{ fontSize: '1.6rem', marginBottom: '0.5rem' }}>{item.icon}</div>
                <div style={{ fontSize: '0.65rem', color: '#3b82f6', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
                  Step {item.step}
                </div>
                <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: '0.35rem', fontSize: '0.95rem' }}>{item.title}</div>
                <div style={{ fontSize: '0.82rem', color: '#64748b', lineHeight: 1.5 }}>{item.desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <footer style={{ marginTop: '3rem', textAlign: 'center', color: '#334155', fontSize: '0.8rem' }}>
          TruthLens · AI-Powered Fake News Detection · Built with FastAPI + React
          <br />
          <span style={{ fontSize: '0.72rem' }}>Results are AI-generated. Always cross-verify with official sources.</span>
        </footer>
      </main>
    </div>
  );
}
