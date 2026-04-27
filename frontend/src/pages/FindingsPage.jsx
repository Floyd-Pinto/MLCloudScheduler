// src/pages/FindingsPage.jsx — Combined findings: model accuracy + scheduler comparison
import { useState } from 'react';
import toast from 'react-hot-toast';
import { mlAPI, schedulerAPI } from '../services/api';
import ForecastChart from '../charts/ForecastChart';

export default function FindingsPage() {
  const [tab, setTab] = useState('scheduler');

  // Scheduler comparison state
  const [pattern, setPattern] = useState('gradual');
  const [steps, setSteps]     = useState(200);
  const [seed, setSeed]       = useState(42);
  const [running, setRunning] = useState(false);
  const [result, setResult]   = useState(null);

  // Model comparison state
  const [mPattern, setMPattern] = useState('combined');
  const [mSteps, setMSteps]     = useState(300);
  const [mSeed, setMSeed]       = useState(42);
  const [mRunning, setMRunning] = useState(false);
  const [mResult, setMResult]   = useState(null);

  const runSchedulerComparison = async () => {
    setRunning(true);
    try {
      const res = await schedulerAPI.compare({ pattern, steps, seed });
      setResult(res.data);
      toast.success('Comparison complete');
    } catch (e) { toast.error('Failed'); }
    setRunning(false);
  };

  const runModelComparison = async () => {
    setMRunning(true);
    try {
      const res = await mlAPI.compareModels({ pattern: mPattern, steps: mSteps, seed: mSeed });
      setMResult(res.data);
      toast.success('Model comparison complete');
    } catch (e) { toast.error('Failed'); }
    setMRunning(false);
  };

  const r = result?.reactive;
  const p = result?.predictive;

  const metricRow = (label, rVal, pVal, unit = '', better = 'lower') => {
    const improved = better === 'lower' ? pVal < rVal : pVal > rVal;
    return (
      <tr key={label}>
        <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{label}</td>
        <td className="metric-value">{rVal != null ? rVal.toFixed(2) : '—'}{unit}</td>
        <td className="metric-value">{pVal != null ? pVal.toFixed(2) : '—'}{unit}</td>
        <td className="metric-value" style={{ color: improved ? 'var(--green)' : 'var(--red)' }}>
          {rVal != null && pVal != null
            ? `${improved ? '↓' : '↑'} ${Math.abs(((rVal - pVal) / Math.max(rVal, 0.01)) * 100).toFixed(1)}%`
            : '—'}
        </td>
      </tr>
    );
  };

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Findings</div>
        <div className="page-subtitle">
          Experimental results — reactive vs predictive scheduling and model accuracy comparison.
        </div>
      </div>

      <div className="tabs">
        <button className={`tab-btn ${tab === 'scheduler' ? 'active' : ''}`} onClick={() => setTab('scheduler')}>
          Scheduler Comparison
        </button>
        <button className={`tab-btn ${tab === 'models' ? 'active' : ''}`} onClick={() => setTab('models')}>
          Model Accuracy
        </button>
      </div>

      {tab === 'scheduler' && (
        <>
          {/* Configuration */}
          <div className="card">
            <div className="section-title">Run Experiment</div>
            <div style={{ display: 'flex', gap: 16, alignItems: 'end', flexWrap: 'wrap' }}>
              <div className="form-group" style={{ marginBottom: 0, flex: 1, minWidth: 150 }}>
                <label className="form-label">Workload Pattern</label>
                <select className="form-control" value={pattern} onChange={e => setPattern(e.target.value)}>
                  <option value="gradual">Gradual Growth</option>
                  <option value="spike">Sudden Spike</option>
                  <option value="periodic">Periodic / Diurnal</option>
                  <option value="combined">Combined</option>
                </select>
              </div>
              <div className="form-group" style={{ marginBottom: 0, width: 120 }}>
                <label className="form-label">Steps</label>
                <input className="form-control" type="number" value={steps} onChange={e => setSteps(+e.target.value)} />
              </div>
              <div className="form-group" style={{ marginBottom: 0, width: 100 }}>
                <label className="form-label">Seed</label>
                <input className="form-control" type="number" value={seed} onChange={e => setSeed(+e.target.value)} />
              </div>
              <button className="btn btn-primary" disabled={running} onClick={runSchedulerComparison}>
                {running ? 'Running…' : 'Run Comparison'}
              </button>
            </div>
          </div>

          {/* Results */}
          {r && p && (
            <>
              <div className="card">
                <div className="section-title">
                  Results — {pattern.charAt(0).toUpperCase() + pattern.slice(1)} Pattern ({steps} steps)
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Reactive (Baseline)</th>
                        <th>Predictive (Proposed)</th>
                        <th>Change</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metricRow('Overload Events', r.overload_events, p.overload_events, '', 'lower')}
                      {metricRow('Overload Rate', r.overload_rate, p.overload_rate, '%', 'lower')}
                      {metricRow('Avg CPU Utilisation', r.avg_cpu, p.avg_cpu, '%', 'lower')}
                      {metricRow('Total Cost', r.total_cost, p.total_cost, '', 'lower')}
                      {metricRow('Scale-Up Actions', r.scale_up_count, p.scale_up_count, '', 'lower')}
                      {metricRow('Scale-Down Actions', r.scale_down_count, p.scale_down_count, '')}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="grid-2">
                <div className="research-card">
                  <h3>Overload Reduction</h3>
                  <div className="stat" style={{ color: 'var(--green)' }}>
                    {r.overload_events > 0
                      ? ((r.overload_events - p.overload_events) / r.overload_events * 100).toFixed(0)
                      : 0}%
                  </div>
                  <div className="label">
                    {r.overload_events} → {p.overload_events} overload events
                  </div>
                </div>
                <div className="research-card">
                  <h3>CPU Stability</h3>
                  <div className="stat">{p.avg_cpu?.toFixed(1)}%</div>
                  <div className="label">
                    Predictive avg CPU (vs {r.avg_cpu?.toFixed(1)}% reactive)
                  </div>
                </div>
              </div>
            </>
          )}
        </>
      )}

      {tab === 'models' && (
        <>
          <div className="card">
            <div className="section-title">Run Model Evaluation</div>
            <div style={{ display: 'flex', gap: 16, alignItems: 'end', flexWrap: 'wrap' }}>
              <div className="form-group" style={{ marginBottom: 0, flex: 1, minWidth: 150 }}>
                <label className="form-label">Workload Pattern</label>
                <select className="form-control" value={mPattern} onChange={e => setMPattern(e.target.value)}>
                  <option value="gradual">Gradual Growth</option>
                  <option value="spike">Sudden Spike</option>
                  <option value="periodic">Periodic / Diurnal</option>
                  <option value="combined">Combined</option>
                </select>
              </div>
              <div className="form-group" style={{ marginBottom: 0, width: 120 }}>
                <label className="form-label">Steps</label>
                <input className="form-control" type="number" value={mSteps} onChange={e => setMSteps(+e.target.value)} />
              </div>
              <div className="form-group" style={{ marginBottom: 0, width: 100 }}>
                <label className="form-label">Seed</label>
                <input className="form-control" type="number" value={mSeed} onChange={e => setMSeed(+e.target.value)} />
              </div>
              <button className="btn btn-primary" disabled={mRunning} onClick={runModelComparison}>
                {mRunning ? 'Running…' : 'Evaluate Models'}
              </button>
            </div>
          </div>

          {mResult && (
            <>
              <div className="card">
                <div className="section-title">Model Accuracy Comparison</div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>R² Score</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {['lstm', 'arima', 'combined'].map(mt => {
                        const m = mResult.metrics?.[mt];
                        const isBest = mt === mResult.best_model;
                        return (
                          <tr key={mt}>
                            <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                              {mt.toUpperCase()}
                              {isBest && <span className="badge badge-green" style={{ marginLeft: 8 }}>Best</span>}
                            </td>
                            <td className="metric-value">{m?.r2?.toFixed(4) ?? '—'}</td>
                            <td className="metric-value">{m?.rmse?.toFixed(4) ?? '—'}</td>
                            <td className="metric-value">{m?.mae?.toFixed(4) ?? '—'}</td>
                            <td>
                              <span className={`badge ${m?.ready ? 'badge-green' : 'badge-red'}`}>
                                {m?.ready ? 'Trained' : 'Untrained'}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {mResult.chart && (
                <div className="card">
                  <div className="section-title">Forecast vs Actual — {mPattern}</div>
                  <ForecastChart data={mResult.chart} />
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
