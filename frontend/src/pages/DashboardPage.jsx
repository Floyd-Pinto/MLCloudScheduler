// src/pages/DashboardPage.jsx — Research overview with Phase 2 multi-resource info
import { useState, useEffect, useCallback } from 'react';
import { mlAPI, metricsAPI } from '../services/api';

export default function DashboardPage() {
  const [mlStatus, setMlStatus] = useState(null);
  const [metrics, setMetrics]   = useState(null);
  const [loading, setLoading]   = useState(true);

  const fetchAll = useCallback(async () => {
    try {
      const [mlRes, metRes] = await Promise.all([
        mlAPI.status(),
        metricsAPI.summary(),
      ]);
      setMlStatus(mlRes.data);
      setMetrics(metRes.data);
    } catch (e) { console.error(e); }
    setLoading(false);
  }, []);

  useEffect(() => { fetchAll(); const id = setInterval(fetchAll, 30000); return () => clearInterval(id); }, [fetchAll]);

  const s = mlStatus?.statuses || {};
  const r = metrics?.reactive;
  const p = metrics?.predictive;

  const models = [
    { key: 'lstm',     name: 'LSTM',     desc: 'Multi-Resource LSTM — 3-input (CPU, Memory, Network) deep learning forecaster' },
    { key: 'arima',    name: 'ARIMA',    desc: 'Per-Channel ARIMA — independent order selection via AIC grid search per resource' },
    { key: 'combined', name: 'Combined', desc: 'Adaptive Hybrid Ensemble — LSTM + ARIMA with per-resource inverse-RMSE weights' },
    { key: 'gbr',      name: 'GBR',      desc: 'MultiOutput GradientBoosting — baseline tree-based regressor across all resources' },
  ];

  const fmtR2 = (v) => v != null ? v.toFixed(4) : '—';
  const fmtVal = (v) => v != null ? v.toFixed(2) : '—';

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Research Overview</div>
        <div className="page-subtitle">
          ML-Based Adaptive Cloud Resource Scheduling — Multi-Resource Predictive vs Reactive Approach
        </div>
      </div>

      {/* Hypothesis */}
      <div className="card">
        <div className="section-title">Research Hypothesis</div>
        <p style={{ color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.8 }}>
          Current cloud resource scheduling relies on <strong style={{ color: 'var(--text-primary)' }}>reactive autoscaling</strong> —
          resources are allocated only <em>after</em> overload is detected, introducing a provisioning lag of 30–120 seconds.
          This project proposes using <strong style={{ color: 'var(--text-primary)' }}>ML-based predictive scheduling</strong> with
          LSTM, ARIMA, and a Combined (LSTM+ARIMA) adaptive ensemble to forecast workload 5 steps ahead across
          <strong style={{ color: 'var(--text-primary)' }}> three resource dimensions</strong> (CPU, Memory, Network I/O) and
          proactively allocate resources <em>before</em> overload occurs on any resource.
        </p>
        <div style={{ marginTop: 16, padding: '12px 16px', background: 'var(--accent-glow)',
                      borderRadius: 'var(--radius-md)', border: '1px solid var(--border)',
                      fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: 'var(--text-secondary)' }}>
          H0: Multi-resource predictive scheduling using LSTM/ARIMA forecasting reduces overload events by &ge;40%
          compared to threshold-based reactive scheduling across gradual, spike, and periodic workload patterns,
          considering CPU, memory, and network I/O simultaneously.
        </div>
      </div>

      {/* Model Status */}
      <div className="card">
        <div className="section-title">Proposed Models — Status</div>
        {loading ? <div className="loading-spinner" /> : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {models.map(({ key, name, desc }) => {
              const info = mlStatus?.[key];
              const ready = s[key];
              return (
                <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                        padding: '14px 18px', background: 'var(--bg-input)',
                                        borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 2 }}>{name}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{desc}</div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontFamily: 'JetBrains Mono, monospace', fontSize: 12 }}>
                    {info?.r2 != null && (
                      <>
                        <span style={{ color: 'var(--text-muted)' }}>R² = <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{fmtR2(info.r2)}</span></span>
                        <span style={{ color: 'var(--text-muted)' }}>RMSE = <span style={{ color: 'var(--text-primary)' }}>{fmtVal(info.rmse)}</span></span>
                        <span style={{ color: 'var(--text-muted)' }}>MAE = <span style={{ color: 'var(--text-primary)' }}>{fmtVal(info.mae)}</span></span>
                      </>
                    )}
                    <span className={`badge ${ready ? 'badge-green' : 'badge-red'}`}>
                      {ready ? 'Trained' : 'Untrained'}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Key Finding */}
      {r && p && (
        <div className="card">
          <div className="section-title">Key Finding — Scheduler Comparison (Aggregated)</div>
          <div className="grid-3">
            <div className="research-card">
              <h3>Reactive (Baseline)</h3>
              <div className="stat">{r.avg_overload?.toFixed(1)}%</div>
              <div className="label">Average overload rate across {r.run_count} runs</div>
            </div>
            <div className="research-card">
              <h3>Predictive (Proposed)</h3>
              <div className="stat">{p.avg_overload?.toFixed(1)}%</div>
              <div className="label">Average overload rate across {p.run_count} runs</div>
            </div>
            <div className="research-card">
              <h3>Overload Reduction</h3>
              <div className="stat" style={{ color: 'var(--green)' }}>
                {r.avg_overload > 0 ? ((r.avg_overload - p.avg_overload) / r.avg_overload * 100).toFixed(0) : '—'}%
              </div>
              <div className="label">Target: 40–60% (Proposal V3, Table 2)</div>
            </div>
          </div>
        </div>
      )}

      {/* Novel Contributions */}
      <div className="card">
        <div className="section-title">Novel Contributions — Phase 2</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {[
            { title: 'Multi-Resource Forecasting',
              desc: 'Extended from single-signal (CPU-only) to 3-resource (CPU, Memory, Network I/O) prediction. All models now accept (20, 3) input windows and produce (5, 3) forecast tensors, enabling correlated resource scheduling.' },
            { title: 'Per-Resource Ensemble Weights',
              desc: 'The Combined model now computes independent inverse-RMSE weights for each resource dimension. LSTM may dominate CPU prediction while ARIMA may be preferred for memory forecasting, enabling fine-grained resource-specific model selection.' },
            { title: 'Anomaly-Aware Scheduling',
              desc: 'Integrated Rolling Z-Score + Isolation Forest anomaly detection. When anomalies are detected, the predictive scheduler temporarily lowers scale-up thresholds by 10%, providing proactive safety margins during unusual workload events.' },
            { title: 'Rate-of-Change Spike Detection',
              desc: 'The predictive scheduler includes a fast-path heuristic that detects spike onset via workload acceleration. If CPU increases by >15% of capacity in 3 steps, it triggers an immediate scale-up — bypassing the ML model for faster response.' },
          ].map(({ title, desc }, i) => (
            <div key={i} style={{ padding: '14px 18px', background: 'var(--bg-input)',
                                  borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
              <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
                            color: 'var(--text-muted)', marginBottom: 4 }}>Contribution {i + 1}</div>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{title}</div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Methodology */}
      <div className="card">
        <div className="section-title">Methodology</div>
        <div className="grid-3">
          {[
            { step: '01', title: 'Workload Simulation',
              desc: 'Generate synthetic 3-resource cloud workloads (CPU, Memory, Network I/O) with 4 patterns. Memory lags CPU by ~3 steps (ρ≈0.75). Network exhibits Poisson-burst behavior.' },
            { step: '02', title: 'Model Training',
              desc: 'Train 4 models: GBR (baseline), LSTM (150 epochs, multi-output), per-channel ARIMA (AIC grid search), and Combined ensemble with per-resource inverse-RMSE weighting.' },
            { step: '03', title: 'Evaluation',
              desc: 'Compare reactive vs ML-predictive scheduling across per-resource overloads, utilisation, and cost. Anomaly-aware threshold adjustment enabled during predictive runs.' },
          ].map(({ step, title, desc }) => (
            <div key={step} style={{ padding: '18px 20px', background: 'var(--bg-input)',
                                     borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
              <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
                            color: 'var(--text-muted)', marginBottom: 8 }}>Step {step}</div>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{title}</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.7 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
