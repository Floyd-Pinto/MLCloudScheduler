// src/pages/DashboardPage.jsx — Research overview
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
    { key: 'lstm',     name: 'LSTM',     desc: 'Long Short-Term Memory — deep learning temporal model' },
    { key: 'arima',    name: 'ARIMA',    desc: 'Auto-Regressive Integrated Moving Average — statistical baseline' },
    { key: 'combined', name: 'Combined', desc: 'LSTM + ARIMA ensemble weighted by inverse-RMSE' },
  ];

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Research Overview</div>
        <div className="page-subtitle">
          ML-Based Adaptive Cloud Resource Scheduling — Predictive vs Reactive Approach
        </div>
      </div>

      {/* Hypothesis */}
      <div className="card">
        <div className="section-title">Research Hypothesis</div>
        <p style={{ color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.8 }}>
          Current cloud resource scheduling relies on <strong style={{ color: 'var(--text-primary)' }}>reactive autoscaling</strong> — 
          resources are allocated only <em>after</em> overload is detected, introducing a provisioning lag of 30–120 seconds.
          This project proposes using <strong style={{ color: 'var(--text-primary)' }}>ML-based predictive scheduling</strong> with 
          LSTM, ARIMA, and a Combined (LSTM+ARIMA) ensemble model to forecast workload 5 steps ahead and 
          proactively allocate resources <em>before</em> overload occurs.
        </p>
        <div style={{ marginTop: 16, padding: '12px 16px', background: 'var(--accent-glow)',
                      borderRadius: 'var(--radius-md)', border: '1px solid var(--border)',
                      fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: 'var(--text-secondary)' }}>
          H₀: Predictive scheduling using LSTM/ARIMA forecasting reduces overload events by ≥40% 
          compared to threshold-based reactive scheduling across gradual, spike, and periodic workload patterns.
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
                        <span style={{ color: 'var(--text-muted)' }}>R² = <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{info.r2.toFixed(4)}</span></span>
                        <span style={{ color: 'var(--text-muted)' }}>RMSE = <span style={{ color: 'var(--text-primary)' }}>{info.rmse?.toFixed(4)}</span></span>
                      </>
                    )}
                    <span className={`badge ${ready ? 'badge-green' : 'badge-red'}`}>
                      {ready ? '✓ Trained' : '✗ Untrained'}
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
          <div className="section-title">Key Finding — Scheduler Comparison</div>
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
              <h3>Improvement</h3>
              <div className="stat" style={{ color: 'var(--green)' }}>
                {r.avg_overload > 0 ? ((r.avg_overload - p.avg_overload) / r.avg_overload * 100).toFixed(0) : '—'}%
              </div>
              <div className="label">Overload reduction (target: 40–60%)</div>
            </div>
          </div>
        </div>
      )}

      {/* Methodology */}
      <div className="card">
        <div className="section-title">Methodology</div>
        <div className="grid-3">
          {[
            { step: '01', title: 'Workload Simulation',
              desc: 'Generate synthetic cloud workloads with 3 patterns: gradual growth, sudden spikes, and periodic/diurnal cycles.' },
            { step: '02', title: 'Model Training',
              desc: 'Train LSTM (deep learning), ARIMA (statistical), and Combined (ensemble) forecasting models on historical telemetry.' },
            { step: '03', title: 'Evaluation',
              desc: 'Compare reactive vs ML-predictive scheduling across overload count, CPU utilisation, response delay, and cost.' },
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
