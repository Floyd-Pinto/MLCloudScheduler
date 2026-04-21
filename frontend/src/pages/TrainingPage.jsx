// src/pages/TrainingPage.jsx — Enhanced: train all 4 models individually
import { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { mlAPI } from '../services/api';

const MODELS = [
  { key:'gbr',      icon:'🌲', label:'GradientBoosting (GBR)',       color:'var(--accent)',  badge:'badge-blue',
    desc:'Ensemble of 200 gradient-boosted trees. Sliding window of 10 steps as features. Trains in ~2s. Best overall R².' },
  { key:'lstm',     icon:'🧠', label:'LSTM Neural Network',           color:'var(--purple)', badge:'badge-purple',
    desc:'2-layer PyTorch LSTM with FC head. Input window=15, horizon=5. Adam + ReduceLROnPlateau. ~20s training time.' },
  { key:'arima',    icon:'📈', label:'ARIMA',                         color:'var(--yellow)', badge:'badge-yellow',
    desc:'Auto-selected order via AIC grid search. Walk-forward validation. Best for stationary+trending series.' },
  { key:'combined', icon:'⚡', label:'Combined Hybrid (LSTM+ARIMA)',  color:'var(--green)',  badge:'badge-green',
    desc:'Weighted ensemble: trains LSTM + ARIMA, weights by inverse RMSE. ARIMA handles trend; LSTM handles non-linearity.' },
];

function ModelCard({ model, statusData, onTrain, training }) {
  const { key, icon, label, color, badge, desc } = model;
  const s    = statusData?.[key];
  const ready = s?.ready;

  return (
    <div className="card" style={{ borderTop:`2px solid ${color}` }}>
      <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:10 }}>
        <span style={{ fontSize:22 }}>{icon}</span>
        <div style={{ flex:1 }}>
          <div style={{ fontWeight:800, fontSize:14, color:'var(--text-primary)' }}>{label}</div>
          <div style={{ fontSize:11, color:'var(--text-muted)', marginTop:2 }}>{desc}</div>
        </div>
        <span className={`badge ${ready ? 'badge-green' : 'badge-yellow'}`}>
          {ready ? '✓ Ready' : '⚠ Needs Training'}
        </span>
      </div>

      {ready && s && (
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:8, marginBottom:12 }}>
          {[['R²', s.r2?.toFixed(4), color], ['RMSE', s.rmse?.toFixed(4), 'var(--text-secondary)'], ['MAE', s.mae?.toFixed(4), 'var(--text-muted)']].map(([l,v,c]) => (
            <div key={l} style={{ background:'var(--bg-input)', borderRadius:'var(--radius-sm)',
                                  padding:'8px 6px', textAlign:'center' }}>
              <div style={{ fontSize:10, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.07em' }}>{l}</div>
              <div style={{ fontSize:16, fontWeight:800, color:c }}>{v ?? '—'}</div>
            </div>
          ))}
        </div>
      )}

      {s?.extra_info && Object.keys(s.extra_info).length > 0 && (
        <div style={{ marginBottom:10, display:'flex', gap:6, flexWrap:'wrap' }}>
          {Object.entries(s.extra_info).map(([k,v]) => (
            <span key={k} className="badge badge-purple" style={{fontSize:10}}>
              {k}={typeof v==='number'?v.toFixed(3):v}
            </span>
          ))}
        </div>
      )}

      {s?.finished_at && (
        <div style={{ fontSize:11, color:'var(--text-muted)', marginBottom:10 }}>
          Last trained: {new Date(s.finished_at).toLocaleString()}
        </div>
      )}

      <button className="btn btn-primary btn-sm" style={{ width:'100%' }}
        onClick={() => onTrain(key)} disabled={training[key]}>
        {training[key] ? <><span className="loading-spinner"/>&nbsp;Training…</> : `🚀 Train ${label}`}
      </button>
    </div>
  );
}

export default function TrainingPage() {
  const [status,   setStatus]   = useState(null);
  const [history,  setHistory]  = useState([]);
  const [training, setTraining] = useState({});
  const [trainAll, setTrainAll] = useState(false);

  useEffect(() => { fetchStatus(); fetchHistory(); }, []);

  const fetchStatus  = () => mlAPI.status().then(r => setStatus(r.data)).catch(console.error);
  const fetchHistory = () => mlAPI.history().then(r => setHistory(r.data)).catch(console.error);

  const handleTrain = async (mt) => {
    setTraining(t => ({...t, [mt]: true}));
    toast(`⏳ Training ${mt.toUpperCase()}…`, { duration: 3000 });
    try {
      const res = await mlAPI.train({ model_type: mt });
      const rec = Array.isArray(res.data) ? res.data[0] : res.data;
      if (rec.status === 'completed') {
        toast.success(`✅ ${mt.toUpperCase()} — R²=${rec.r2?.toFixed(4)} RMSE=${rec.rmse?.toFixed(4)}`);
      } else {
        toast.error(`${mt.toUpperCase()} failed: ${rec.error_msg?.slice(0,80) || '?'}`);
      }
      fetchStatus(); fetchHistory();
    } catch (e) {
      toast.error(e.response?.data?.error || `${mt} training failed`);
    } finally {
      setTraining(t => ({...t, [mt]: false}));
    }
  };

  const handleTrainAll = async () => {
    setTrainAll(true);
    toast('🚀 Training all 4 models… (~90s)', { duration: 5000 });
    try {
      const res = await mlAPI.trainAll();
      const records = Array.isArray(res.data) ? res.data : [res.data];
      const done = records.filter(r => r.status === 'completed').map(r => r.model_type.toUpperCase());
      const failed = records.filter(r => r.status === 'failed').map(r => r.model_type.toUpperCase());
      if (done.length) toast.success(`Trained: ${done.join(', ')}`);
      if (failed.length) toast.error(`Failed: ${failed.join(', ')}`);
      fetchStatus(); fetchHistory();
    } catch (e) {
      toast.error(e.response?.data?.error || 'Train-all failed');
    } finally {
      setTrainAll(false);
    }
  };

  const anyTraining = trainAll || Object.values(training).some(Boolean);

  return (
    <div>
      <div className="page-header">
        <div className="page-title">🧠 Model Training</div>
        <div className="page-subtitle">
          Train and manage all 4 forecasting models — GBR · LSTM · ARIMA · Combined.
        </div>
      </div>

      {/* Train all shortcut */}
      <div className="card" style={{ marginBottom:24,
        background:'linear-gradient(135deg,#0f1f3d,#1a2236)', border:'1px solid var(--border)' }}>
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', flexWrap:'wrap', gap:16 }}>
          <div>
            <div style={{ fontWeight:800, fontSize:16, color:'var(--text-primary)', marginBottom:4 }}>
              🚀 Train All Models at Once
            </div>
            <div style={{ fontSize:13, color:'var(--text-muted)', maxWidth:500 }}>
              Runs GBR → LSTM → ARIMA → Combined in sequence. Total time: ~90s.
              This populates all model artifacts needed for the Model Comparison page.
            </div>
          </div>
          <button className="btn btn-success btn-lg"
            onClick={handleTrainAll} disabled={anyTraining}>
            {trainAll ? <><span className="loading-spinner"/>&nbsp;Training all…</> : '⚡ Train All 4 Models'}
          </button>
        </div>
      </div>

      {/* Individual model cards */}
      <div className="grid-2" style={{ marginBottom:24 }}>
        {MODELS.map(m => (
          <ModelCard key={m.key} model={m} statusData={status}
            onTrain={handleTrain} training={{...training, combined: training.combined || trainAll}} />
        ))}
      </div>

      {/* Full training history */}
      <div className="card">
        <div className="section-title" style={{display:'flex',justifyContent:'space-between'}}>
          <span>Training History</span>
          <span className="badge badge-blue">{history.length} records</span>
        </div>
        {history.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-icon">📊</span>
            <span>No training runs yet.</span>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>ID</th><th>Model</th><th>Status</th><th>R²</th><th>RMSE</th><th>MAE</th><th>Extra</th><th>Finished</th></tr>
              </thead>
              <tbody>
                {history.map(h => {
                  const m = MODELS.find(m => m.key === h.model_type);
                  return (
                    <tr key={h.id}>
                      <td>#{h.id}</td>
                      <td>
                        <span style={{ display:'flex', alignItems:'center', gap:5 }}>
                          <span>{m?.icon}</span>
                          <span className={`badge ${m?.badge || 'badge-blue'}`}>
                            {h.model_type.toUpperCase()}
                          </span>
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${h.status==='completed'?'badge-green':h.status==='failed'?'badge-red':'badge-yellow'}`}>
                          {h.status}
                        </span>
                      </td>
                      <td style={{ color:h.r2>=0.9?'var(--green)':h.r2>=0.7?'var(--yellow)':'var(--red)', fontWeight:700 }}>
                        {h.r2?.toFixed(4) ?? '—'}
                      </td>
                      <td>{h.rmse?.toFixed(4) ?? '—'}</td>
                      <td>{h.mae?.toFixed(4) ?? '—'}</td>
                      <td style={{ fontSize:11, color:'var(--text-muted)' }}>
                        {Object.keys(h.extra_info||{}).length > 0
                          ? Object.entries(h.extra_info).map(([k,v])=>`${k}:${typeof v==='number'?v.toFixed(3):v}`).join(' ')
                          : '—'}
                      </td>
                      <td style={{ fontSize:11 }}>{h.finished_at ? new Date(h.finished_at).toLocaleString() : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
