// src/pages/ModelComparisonPage.jsx
// Full model comparison: GBR vs LSTM vs ARIMA vs Combined
import { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { mlAPI } from '../services/api';
import ForecastChart  from '../charts/ForecastChart';
import ModelRadarChart from '../charts/RadarChart';
import BarCompareChart from '../charts/BarCompareChart';

const MODEL_META = {
  gbr:      { label: 'GBR',      color: 'var(--accent)',  badge: 'badge-blue',   icon: '🌲', desc: 'GradientBoosting Regressor' },
  lstm:     { label: 'LSTM',     color: 'var(--purple)',  badge: 'badge-purple', icon: '🧠', desc: 'Long Short-Term Memory (PyTorch)' },
  arima:    { label: 'ARIMA',    color: 'var(--yellow)',  badge: 'badge-yellow', icon: '📈', desc: 'ARIMA (Auto-Regressive Integrated MA)' },
  combined: { label: 'Combined', color: 'var(--green)',   badge: 'badge-green',  icon: '⚡', desc: 'Hybrid LSTM + ARIMA Ensemble' },
};

const BAR_METRICS = [
  { key: 'r2',   label: 'R² Score' },
];
const BAR_METRICS2 = [
  { key: 'rmse', label: 'RMSE' },
  { key: 'mae',  label: 'MAE'  },
];

function MetricCard({ model, metrics }) {
  const m    = MODEL_META[model];
  const data = metrics?.[model];
  const ready = data?.ready !== false && data?.r2 != null;
  return (
    <div className="card" style={{ borderTop: `3px solid ${m.color}` }}>
      <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:12 }}>
        <span style={{ fontSize:22 }}>{m.icon}</span>
        <div>
          <div style={{ fontWeight:800, fontSize:15, color:'var(--text-primary)' }}>{m.label}</div>
          <div style={{ fontSize:11, color:'var(--text-muted)' }}>{m.desc}</div>
        </div>
        <span className={`badge ${m.badge}`} style={{ marginLeft:'auto' }}>
          {ready ? '✓ Ready' : '⚠ Not Ready'}
        </span>
      </div>
      {ready ? (
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:8 }}>
          {[['R²', data.r2?.toFixed(4), m.color], ['RMSE', data.rmse?.toFixed(4), 'var(--text-secondary)'], ['MAE', data.mae?.toFixed(4), 'var(--text-muted)']].map(([lbl, val, col]) => (
            <div key={lbl} style={{ textAlign:'center', background:'var(--bg-input)',
                                    borderRadius:'var(--radius-sm)', padding:'8px 4px' }}>
              <div style={{ fontSize:10, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.07em' }}>{lbl}</div>
              <div style={{ fontSize:18, fontWeight:800, color:col, fontVariantNumeric:'tabular-nums' }}>{val ?? '—'}</div>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ textAlign:'center', padding:'16px 0', color:'var(--text-muted)', fontSize:13 }}>
          Train this model to see metrics
        </div>
      )}
    </div>
  );
}

export default function ModelComparisonPage() {
  const [form,       setForm]       = useState({ pattern:'combined', steps:300, seed:42 });
  const [result,     setResult]     = useState(null);
  const [status,     setStatus]     = useState(null);
  const [history,    setHistory]    = useState([]);
  const [loading,    setLoading]    = useState(false);
  const [trainState, setTrainState] = useState({});
  const [visible,    setVisible]    = useState(['actual','gbr','lstm','arima','combined']);
  const [chart,      setChart]      = useState('forecast'); // forecast | radar | bar

  useEffect(() => {
    fetchStatus();
    mlAPI.comparisonList().then(r => setHistory(r.data)).catch(console.error);
  }, []);

  const fetchStatus = () =>
    mlAPI.status().then(r => setStatus(r.data)).catch(console.error);

  const handleTrainOne = async (mt) => {
    setTrainState(s => ({...s, [mt]: true}));
    toast(`⏳ Training ${mt.toUpperCase()}…`, { duration: 3000 });
    try {
      const res = await mlAPI.train({ model_type: mt });
      const rec = Array.isArray(res.data) ? res.data[0] : res.data;
      if (rec.status === 'completed') {
        toast.success(`✅ ${mt.toUpperCase()} trained — R²=${rec.r2?.toFixed(4)}`);
      } else {
        toast.error(`${mt.toUpperCase()} failed: ${rec.error_msg || '?'}`);
      }
      fetchStatus();
    } catch (e) {
      toast.error(e.response?.data?.error || `${mt} training failed`);
    } finally {
      setTrainState(s => ({...s, [mt]: false}));
    }
  };

  const handleTrainAll = async () => {
    setLoading(true);
    toast('🚀 Training all 4 models… (~90s)', { duration: 6000 });
    try {
      const res = await mlAPI.trainAll();
      const records = Array.isArray(res.data) ? res.data : [res.data];
      const done = records.filter(r => r.status === 'completed').map(r => r.model_type.toUpperCase());
      toast.success(`Trained: ${done.join(', ')}`);
      fetchStatus();
    } catch (e) {
      toast.error(e.response?.data?.error || 'Train all failed');
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    setLoading(true);
    try {
      const res = await mlAPI.compareModels(form);
      setResult(res.data);
      toast.success(`Comparison done — Best: ${res.data.best_model?.toUpperCase()}`);
      mlAPI.comparisonList().then(r => setHistory(r.data)).catch(() => {});
    } catch (e) {
      toast.error(e.response?.data?.error || 'Comparison failed');
    } finally {
      setLoading(false);
    }
  };

  const toggleModel = (m) => setVisible(v =>
    v.includes(m) ? v.filter(x => x !== m) : [...v, m]);

  const statuses = status?.statuses || {};
  const metrics  = result?.metrics;

  return (
    <div>
      <div className="page-header">
        <div className="page-title">🔬 Model Comparison</div>
        <div className="page-subtitle">
          Compare GBR · LSTM · ARIMA · Combined (LSTM+ARIMA Ensemble) on the same workload.
        </div>
      </div>

      {/* Status cards — all 4 models */}
      <div className="grid-2" style={{ marginBottom:24 }}>
        {Object.keys(MODEL_META).map(mt => (
          <div key={mt} style={{ display:'flex', flexDirection:'column', gap:0 }}>
            <MetricCard model={mt} metrics={result?.metrics || {
              [mt]: status?.[mt] ? { r2: status[mt].r2, rmse: status[mt].rmse, mae: status[mt].mae, ready: status[mt].ready } : null
            }} />
            <button className="btn btn-outline btn-sm"
              style={{ marginTop:4 }}
              onClick={() => handleTrainOne(mt)}
              disabled={trainState[mt] || loading}>
              {trainState[mt] ? <><span className="loading-spinner" />&nbsp;Training…</> : `Retrain ${MODEL_META[mt].label}`}
            </button>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="card" style={{ marginBottom:24 }}>
        <div className="section-title">Run Forecast Comparison</div>
        <div style={{ display:'flex', gap:14, flexWrap:'wrap', alignItems:'flex-end' }}>
          <div className="form-group" style={{ flex:1, minWidth:140, marginBottom:0 }}>
            <label className="form-label">Workload Pattern</label>
            <select className="form-control" value={form.pattern}
              onChange={e => setForm(f=>({...f, pattern:e.target.value}))}>
              {['gradual','spike','periodic','combined'].map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="form-group" style={{ flex:1, minWidth:110, marginBottom:0 }}>
            <label className="form-label">Steps</label>
            <input type="number" className="form-control" min={200} max={1000}
              value={form.steps} onChange={e => setForm(f=>({...f, steps:+e.target.value}))} />
          </div>
          <div className="form-group" style={{ flex:1, minWidth:100, marginBottom:0 }}>
            <label className="form-label">Seed</label>
            <input type="number" className="form-control" min={0} max={99999}
              value={form.seed} onChange={e => setForm(f=>({...f, seed:+e.target.value}))} />
          </div>
          <div style={{ display:'flex', gap:8 }}>
            <button className="btn btn-primary" onClick={handleCompare} disabled={loading}>
              {loading ? <><span className="loading-spinner"/>&nbsp;Running…</> : '📊 Run Comparison'}
            </button>
            <button className="btn btn-success" onClick={handleTrainAll} disabled={loading}>
              {loading ? <><span className="loading-spinner"/>&nbsp;Training…</> : '🚀 Train All'}
            </button>
          </div>
        </div>
      </div>

      {result && (
        <>
          {/* Best model banner */}
          {result.best_model && (
            <div style={{
              background:'linear-gradient(135deg,rgba(16,185,129,0.1),rgba(59,130,246,0.1))',
              border:'1px solid rgba(16,185,129,0.3)',
              borderRadius:'var(--radius-lg)', padding:'14px 22px',
              marginBottom:20, display:'flex', alignItems:'center', gap:14,
            }}>
              <span style={{ fontSize:30 }}>🏆</span>
              <div>
                <div style={{ fontWeight:800, fontSize:16, color:'var(--green)' }}>
                  Best Model: {result.best_model.toUpperCase()} — R²={metrics?.[result.best_model]?.r2?.toFixed(4)}
                </div>
                <div style={{ fontSize:13, color:'var(--text-muted)' }}>
                  Evaluated on <strong style={{color:'var(--text-secondary)'}}>{form.pattern}</strong> pattern · {form.steps} steps
                </div>
              </div>
            </div>
          )}

          {/* Metrics table */}
          <div className="card" style={{ marginBottom:24 }}>
            <div className="section-title">Performance Metrics — All Models</div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th><th>Architecture</th>
                    <th>R² Score ↑</th><th>RMSE ↓</th><th>MAE ↓</th><th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(MODEL_META).map(([mt, m]) => {
                    const d = metrics?.[mt];
                    const isBest = mt === result.best_model;
                    return (
                      <tr key={mt} style={{ background: isBest ? 'rgba(16,185,129,0.05)' : '' }}>
                        <td>
                          <span style={{ display:'flex', alignItems:'center', gap:6 }}>
                            {m.icon}
                            <strong style={{ color: m.color }}>{m.label}</strong>
                            {isBest && <span className="badge badge-green" style={{fontSize:10}}>Best</span>}
                          </span>
                        </td>
                        <td style={{ fontSize:12, color:'var(--text-muted)' }}>{m.desc}</td>
                        <td style={{ color: d?.r2 >= 0.9 ? 'var(--green)' : d?.r2 >= 0.7 ? 'var(--yellow)' : 'var(--red)', fontWeight:700 }}>
                          {d?.r2?.toFixed(4) ?? '—'}
                        </td>
                        <td>{d?.rmse?.toFixed(4) ?? '—'}</td>
                        <td>{d?.mae?.toFixed(4) ?? '—'}</td>
                        <td>
                          <span className={`badge ${d?.ready ? 'badge-green' : 'badge-yellow'}`}>
                            {d?.ready ? '✓ evaluated' : '⚠ no data'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Chart tabs */}
          <div className="card" style={{ marginBottom:24 }}>
            <div className="tabs">
              <button className={`tab-btn ${chart==='forecast'?'active':''}`} onClick={()=>setChart('forecast')}>📈 Forecast vs Actual</button>
              <button className={`tab-btn ${chart==='radar'?'active':''}`}    onClick={()=>setChart('radar')}>🕸 Radar Chart</button>
              <button className={`tab-btn ${chart==='bar'?'active':''}`}      onClick={()=>setChart('bar')}>📊 Bar Comparison</button>
            </div>

            {chart === 'forecast' && (
              <>
                {/* Model toggles */}
                <div style={{ display:'flex', gap:8, marginBottom:14, flexWrap:'wrap' }}>
                  {['actual','gbr','lstm','arima','combined'].map(m => (
                    <button key={m}
                      onClick={() => toggleModel(m)}
                      style={{
                        padding:'4px 12px', borderRadius:'99px', border:'1px solid',
                        fontSize:12, fontWeight:600, cursor:'pointer', fontFamily:'inherit',
                        borderColor: visible.includes(m) ? MODEL_META[m]?.color || '#94a3b8' : 'var(--border)',
                        background:  visible.includes(m) ? `${MODEL_META[m]?.color || '#94a3b8'}22` : 'transparent',
                        color:       visible.includes(m) ? MODEL_META[m]?.color || '#94a3b8' : 'var(--text-muted)',
                      }}>
                      {m === 'actual' ? '◦ Actual' : MODEL_META[m].icon + ' ' + MODEL_META[m].label}
                    </button>
                  ))}
                </div>
                <ForecastChart
                  chartData={result.chart}
                  visibleModels={visible}
                  title={`Workload Forecast — ${form.pattern} pattern`}
                />
              </>
            )}

            {chart === 'radar' && (
              <ModelRadarChart metrics={metrics} />
            )}

            {chart === 'bar' && (
              <div className="grid-2">
                <div>
                  <BarCompareChart
                    reactive={{ ...metrics?.gbr,  name:'GBR'      }}
                    predictive={{ ...metrics?.lstm, name:'LSTM'    }}
                    metrics={[{key:'r2',label:'R²'},{key:'rmse',label:'RMSE'},{key:'mae',label:'MAE'}]}
                    title="GBR vs LSTM"
                  />
                </div>
                <div>
                  <BarCompareChart
                    reactive={{ ...metrics?.arima,    name:'ARIMA'    }}
                    predictive={{ ...metrics?.combined, name:'Combined' }}
                    metrics={[{key:'r2',label:'R²'},{key:'rmse',label:'RMSE'},{key:'mae',label:'MAE'}]}
                    title="ARIMA vs Combined"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Architecture explainer */}
          <div className="card">
            <div className="section-title">Model Architecture Notes</div>
            <div className="grid-2">
              {[
                { icon:'🌲', title:'GradientBoosting Regressor', color:'var(--accent)',
                  points:['Ensemble of 200 decision trees','Sliding window (10 steps) → features','Trained on 3200+ samples across all patterns','Training: ~2s  |  Inference: <1ms'] },
                { icon:'🧠', title:'LSTM Neural Network (PyTorch)', color:'var(--purple)',
                  points:['2-layer LSTM + FC head','Input: (batch, seq=15, 1)  →  Output: scalar','Adam optimizer, ReduceLROnPlateau scheduler','Training: ~20s  |  Inference: <5ms'] },
                { icon:'📈', title:'ARIMA', color:'var(--yellow)',
                  points:['Auto-selected order via AIC minimization','Walk-forward validation on test set','Captures linear autocorrelation + trend','Training: ~10s  |  Inference: ~100ms'] },
                { icon:'⚡', title:'Combined Hybrid (LSTM + ARIMA)', color:'var(--green)',
                  points:['Weighted ensemble by inverse RMSE','w_lstm ≈ 0.12, w_arima ≈ 0.88 (data-driven)','ARIMA handles trend, LSTM handles non-linearity','Best stability on periodic + combined patterns'] },
              ].map(({ icon, title, color, points }) => (
                <div key={title} style={{
                  background:'var(--bg-input)', border:'1px solid var(--border-subtle)',
                  borderRadius:'var(--radius-md)', padding:'16px 18px',
                }}>
                  <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:10 }}>
                    <span style={{ fontSize:20 }}>{icon}</span>
                    <div style={{ fontWeight:700, color, fontSize:13 }}>{title}</div>
                  </div>
                  <ul style={{ fontSize:12, color:'var(--text-muted)', lineHeight:1.9,
                               paddingLeft:16, margin:0 }}>
                    {points.map(p => <li key={p}>{p}</li>)}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* History */}
      {history.length > 0 && (
        <div className="card" style={{ marginTop:24 }}>
          <div className="section-title" style={{display:'flex',justifyContent:'space-between'}}>
            <span>Comparison History</span>
            <span className="badge badge-blue">{history.length} runs</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>ID</th><th>Pattern</th><th>GBR R²</th><th>LSTM R²</th><th>ARIMA R²</th><th>Combined R²</th><th>Best</th><th>Created</th></tr>
              </thead>
              <tbody>
                {history.map(h => (
                  <tr key={h.id}>
                    <td>#{h.id}</td>
                    <td><span className={`badge badge-${h.pattern==='spike'?'red':h.pattern==='gradual'?'green':h.pattern==='periodic'?'yellow':'purple'}`}>{h.pattern}</span></td>
                    <td style={{color:'var(--accent)', fontWeight:600}}>{h.gbr_r2?.toFixed(4) ?? '—'}</td>
                    <td style={{color:'var(--purple)', fontWeight:600}}>{h.lstm_r2?.toFixed(4) ?? '—'}</td>
                    <td style={{color:'var(--yellow)', fontWeight:600}}>{h.arima_r2?.toFixed(4) ?? '—'}</td>
                    <td style={{color:'var(--green)', fontWeight:600}}>{h.combined_r2?.toFixed(4) ?? '—'}</td>
                    <td><span className="badge badge-green">{h.best_model?.toUpperCase()}</span></td>
                    <td style={{fontSize:12}}>{new Date(h.created_at).toLocaleString()}</td>
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
