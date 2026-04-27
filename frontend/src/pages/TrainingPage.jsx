// src/pages/TrainingPage.jsx — Train LSTM, ARIMA, and Combined models
import { useState, useEffect, useCallback } from 'react';
import toast from 'react-hot-toast';
import { mlAPI } from '../services/api';

const MODELS = [
  { key: 'lstm',     name: 'LSTM Neural Network',
    desc: '2-layer LSTM with 128 hidden units, BatchNorm, and 3 fully-connected layers. Input window: 20 steps, horizon: 5 steps.',
    type: 'Deep Learning' },
  { key: 'arima',    name: 'ARIMA',
    desc: 'Auto-selected order via AIC grid search. Walk-forward validation on 300-step segment. Statistical baseline.',
    type: 'Statistical' },
  { key: 'combined', name: 'Combined Hybrid (LSTM + ARIMA)',
    desc: 'Weighted ensemble — blends LSTM and ARIMA predictions using inverse-RMSE weighting so the more accurate model has higher influence.',
    type: 'Ensemble' },
];

export default function TrainingPage() {
  const [status, setStatus]     = useState(null);
  const [history, setHistory]   = useState([]);
  const [training, setTraining] = useState(null);
  const [loading, setLoading]   = useState(true);

  const fetchStatus = useCallback(async () => {
    try {
      const [sRes, hRes] = await Promise.all([mlAPI.status(), mlAPI.history()]);
      setStatus(sRes.data);
      setHistory(hRes.data?.results || hRes.data || []);
    } catch (e) { console.error(e); }
    setLoading(false);
  }, []);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);

  const trainModel = async (mt) => {
    setTraining(mt);
    try {
      await mlAPI.train({ model_type: mt });
      toast.success(`${mt.toUpperCase()} trained successfully`);
      await fetchStatus();
    } catch (e) { toast.error(`Training failed: ${e.message}`); }
    setTraining(null);
  };

  const trainAll = async () => {
    setTraining('all');
    try {
      await mlAPI.trainAll();
      toast.success('All models trained');
      await fetchStatus();
    } catch (e) { toast.error('Training failed'); }
    setTraining(null);
  };

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Model Training</div>
        <div className="page-subtitle">
          Train the three proposed forecasting models — LSTM, ARIMA, and Combined ensemble.
        </div>
      </div>

      {/* Train All */}
      <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4 }}>Train All Models</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Runs GBR (baseline) → LSTM → ARIMA → Combined sequentially. Total time: ~3 minutes.
          </div>
        </div>
        <button className="btn btn-primary btn-lg" disabled={training} onClick={trainAll}>
          {training === 'all' ? 'Training…' : 'Train All Models'}
        </button>
      </div>

      {/* Model Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {MODELS.map(({ key, name, desc, type }) => {
          const info  = status?.[key];
          const ready = status?.statuses?.[key];
          return (
            <div key={key} className="card" style={{ marginBottom: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontWeight: 700, fontSize: 15 }}>{name}</span>
                    <span className="badge badge-blue" style={{ fontFamily: 'JetBrains Mono, monospace' }}>{type}</span>
                    {ready && <span className="badge badge-green">✓ Trained</span>}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4, maxWidth: 600 }}>{desc}</div>
                </div>
                <button className="btn btn-outline btn-sm" disabled={training}
                        onClick={() => trainModel(key)}>
                  {training === key ? 'Training…' : `Train ${key.toUpperCase()}`}
                </button>
              </div>

              {info?.r2 != null && (
                <div style={{ display: 'flex', gap: 24, fontFamily: 'JetBrains Mono, monospace', fontSize: 13 }}>
                  <div>
                    <span style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em' }}>R² Score</span>
                    <div style={{ fontWeight: 700, fontSize: 22, marginTop: 2 }}>{info.r2.toFixed(4)}</div>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em' }}>RMSE</span>
                    <div style={{ fontWeight: 600, fontSize: 18, marginTop: 4, color: 'var(--text-secondary)' }}>{info.rmse?.toFixed(4)}</div>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em' }}>MAE</span>
                    <div style={{ fontWeight: 600, fontSize: 18, marginTop: 4, color: 'var(--text-secondary)' }}>{info.mae?.toFixed(4)}</div>
                  </div>
                  {info.extra_info?.w_lstm != null && (
                    <>
                      <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: 24 }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em' }}>W_LSTM</span>
                        <div style={{ fontWeight: 600, fontSize: 18, marginTop: 4, color: 'var(--text-secondary)' }}>{info.extra_info.w_lstm.toFixed(3)}</div>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em' }}>W_ARIMA</span>
                        <div style={{ fontWeight: 600, fontSize: 18, marginTop: 4, color: 'var(--text-secondary)' }}>{info.extra_info.w_arima.toFixed(3)}</div>
                      </div>
                    </>
                  )}
                  {info.finished_at && (
                    <div style={{ marginLeft: 'auto', alignSelf: 'flex-end' }}>
                      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        Last trained: {new Date(info.finished_at).toLocaleString()}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Training History */}
      {history.length > 0 && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="section-title">
            Training History
            <span className="badge badge-blue" style={{ marginLeft: 10 }}>{history.length} runs</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th><th>Model</th><th>Status</th><th>R²</th><th>RMSE</th><th>MAE</th><th>Finished</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(0, 20).map(h => (
                  <tr key={h.id}>
                    <td className="metric-value">#{h.id}</td>
                    <td><span className="badge badge-blue">{h.model_type?.toUpperCase()}</span></td>
                    <td><span className={`badge ${h.status === 'completed' ? 'badge-green' : 'badge-red'}`}>{h.status}</span></td>
                    <td className="metric-value">{h.r2?.toFixed(4) ?? '—'}</td>
                    <td className="metric-value">{h.rmse?.toFixed(4) ?? '—'}</td>
                    <td className="metric-value">{h.mae?.toFixed(4) ?? '—'}</td>
                    <td style={{ fontSize: 12 }}>{h.finished_at ? new Date(h.finished_at).toLocaleString() : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
