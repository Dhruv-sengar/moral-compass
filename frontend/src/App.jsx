import { useState } from 'react';
import axios from 'axios';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const EXAMPLES = [
  "I found a wallet with $500. I returned it.",
  "I lied to grandma about her cooking.",
  "I redirected a runaway trolley."
];

const COLOR_MAP = {
  "Utilitarian": "var(--color-util)",
  "Ethical": "var(--color-ethic)",
  "Selfish": "var(--color-self)"
};

const ICON_MAP = {
  "Ethical": "✧",
  "Utilitarian": "⎈",
  "Selfish": "✗"
};

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    if (!text.trim()) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await axios.post(`${API_URL}/predict`, { text });
      setTimeout(() => setResult(res.data), 200);
    } catch (err) {
      setError(err.response?.data?.detail || "Connection failed.");
    } finally {
      setLoading(false);
    }
  };

  const getProbabilities = () => {
    if (!result?.probabilities) return [];
    return ["Ethical", "Utilitarian", "Selfish"].map(label => ({
      label, value: result.probabilities[label] || 0
    }));
  };

  let stateClass = "idle";
  if (loading) stateClass = "analyzing";
  else if (result) stateClass = result.prediction.toLowerCase();

  return (
    <div className="ethereal-app">
      {/* Dynamic Aurora Background */}
      <div className="aurora-bg">
        <div className={`aurora-orb orb-1 ${stateClass}`}></div>
        <div className={`aurora-orb orb-2 ${stateClass}`}></div>
        <div className={`aurora-orb orb-3 ${stateClass}`}></div>
      </div>

      {/* Grid overlay for tech feel */}
      <div className="tech-grid"></div>

      <div className="dashboard-container">
        <header className="brand-header">
          <div className="logo-mark">A.I.</div>
          <h1>Moral Intelligence Core</h1>
          <div className="status-badge">
            <span className={`status-dot ${stateClass}`}></span>
            {loading ? 'HYPER-SCAN ACTIVE' : result ? 'VERDICT RENDERED' : 'SYSTEM ONLINE'}
          </div>
        </header>

        <div className="objective-panel glass-card">
          <p><span className="obj-label">SYS.OBJECTIVE:</span> The Moral Intelligence Core evaluates human scenarios against ethical frameworks. Input an action or dilemma below, and the quantum core will calculate its underlying moral alignment (Ethical, Utilitarian, or Selfish) using advanced semantic analysis.</p>
        </div>

        <div className="main-grid">

          {/* LEFT: Input Terminal */}
          <div className="glass-card panel-left">
            <div className="card-header">
              <span className="label">SYS.INPUT</span>
              <span className="line"></span>
            </div>

            <div className="input-wrapper">
              <textarea
                placeholder="Enter scenario data for moral evaluation..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && (e.metaKey || e.ctrlKey) && handlePredict()}
              />
              <div className="input-glow"></div>
            </div>

            <button
              className={`btn-execute ${loading ? 'loading' : ''}`}
              onClick={handlePredict}
              disabled={loading || !text.trim()}
            >
              {loading ? (
                <span className="btn-text">PROCESSING <span className="dots">...</span></span>
              ) : (
                <span className="btn-text">EXECUTE ANALYSIS [⌘+↵]</span>
              )}
            </button>

            <div className="quick-presets">
              <div className="presets-label">CACHED SCENARIOS</div>
              <div className="presets-list">
                {EXAMPLES.map((ex, i) => (
                  <button key={i} className="preset-btn" onClick={() => setText(ex)}>
                    <span className="p-id">0{i + 1}</span>
                    <span className="p-text">{ex}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* CENTER: The AI Core */}
          <div className="core-display">
            <div className={`quantum-core ${stateClass}`}>
              <div className="core-rings">
                <div className="ring r1"></div>
                <div className="ring r2"></div>
                <div className="ring r3"></div>
              </div>
              <div className="core-nucleus"></div>
            </div>
          </div>

          {/* RIGHT: Results Matrix */}
          <div className="glass-card panel-right">
            <div className="card-header">
              <span className="label">SYS.OUTPUT</span>
              <span className="line"></span>
            </div>

            {!result && !loading && !error && (
              <div className="idle-state">
                <div className="idle-ring"></div>
                <p>AWAITING DATA INPUT</p>
              </div>
            )}

            {loading && (
              <div className="analyzing-state">
                <div className="scan-line"></div>
                <p>DECODING MORAL MATRIX...</p>
              </div>
            )}

            {error && <div className="error-state">ERR: {error}</div>}

            {result && !loading && (
              <div className="results-matrix">
                <div className="primary-verdict">
                  <div className="v-label">DOMINANT ALIGNMENT</div>
                  <div className="v-value" style={{ color: COLOR_MAP[result.prediction] }}>
                    {ICON_MAP[result.prediction]} {result.prediction}
                  </div>
                </div>

                <div className="probability-bars">
                  <div className="v-label">PROBABILITY DISTRIBUTION</div>
                  {getProbabilities().map((prob, i) => (
                    <div key={i} className="prob-row">
                      <div className="prob-info">
                        <span className="p-name">{prob.label}</span>
                        <span className="p-val">{(prob.value * 100).toFixed(1)}%</span>
                      </div>
                      <div className="prob-track">
                        <div
                          className="prob-fill"
                          style={{
                            width: `${prob.value * 100}%`,
                            backgroundColor: COLOR_MAP[prob.label],
                            boxShadow: `0 0 10px ${COLOR_MAP[prob.label]}`
                          }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>

                {result.explanation?.length > 0 && (
                  <div className="semantic-weights">
                    <div className="v-label">SEMANTIC WEIGHTS</div>
                    <div className="weights-grid">
                      {result.explanation.map((item, i) => (
                        <div key={i} className="weight-tag" style={{ borderLeftColor: COLOR_MAP[item.class] }}>
                          {item.feature}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
