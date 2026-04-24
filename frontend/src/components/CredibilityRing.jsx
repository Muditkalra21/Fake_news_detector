/**
 * CredibilityRing.jsx
 * Animated SVG ring showing the credibility score (0-100).
 */
import { useEffect, useState } from 'react';

const RADIUS = 54;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

// Map label → stroke color
const LABEL_COLORS = {
  REAL:       '#22c55e',
  FAKE:       '#ef4444',
  MISLEADING: '#f59e0b',
  UNKNOWN:    '#6b7280',
};

export default function CredibilityRing({ score = 0, label = 'UNKNOWN' }) {
  const [animatedScore, setAnimatedScore] = useState(0);

  // Animate the number on mount / score change
  useEffect(() => {
    let start = 0;
    const step = score / 40;  // ~40 frames
    const timer = setInterval(() => {
      start += step;
      if (start >= score) {
        setAnimatedScore(score);
        clearInterval(timer);
      } else {
        setAnimatedScore(Math.round(start));
      }
    }, 25);
    return () => clearInterval(timer);
  }, [score]);

  const color = LABEL_COLORS[label] ?? LABEL_COLORS.UNKNOWN;
  const offset = CIRCUMFERENCE - (animatedScore / 100) * CIRCUMFERENCE;

  return (
    <div className="ring-container">
      <svg className="ring-svg" viewBox="0 0 120 120">
        {/* Track */}
        <circle className="ring-track" cx="60" cy="60" r={RADIUS} />
        {/* Fill */}
        <circle
          className="ring-fill"
          cx="60" cy="60" r={RADIUS}
          stroke={color}
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
        />
      </svg>

      {/* Centre label */}
      <div className="ring-center">
        <span style={{ fontSize: '1.7rem', fontWeight: 800, color, lineHeight: 1 }}>
          {animatedScore}
        </span>
        <span style={{ fontSize: '0.65rem', color: '#94a3b8', fontWeight: 600, marginTop: 2 }}>
          / 100
        </span>
      </div>
    </div>
  );
}
