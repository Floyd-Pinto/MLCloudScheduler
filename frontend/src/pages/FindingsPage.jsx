// src/pages/FindingsPage.jsx — Phase 2: per-resource overload breakdown + PNG export
import { useState, useRef, useCallback } from 'react';
import toast from 'react-hot-toast';
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  BarElement, Legend, Tooltip,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { mlAPI, schedulerAPI } from '../services/api';
import ForecastChart from '../charts/ForecastChart';

ChartJS.register(CategoryScale, LinearScale, BarElement, Legend, Tooltip);

/* ── PNG export helper ─────────────────────────────────────────────────────── */
const exportChart = (chartRef, filename) => {
  if (!chartRef?.current) return;
  const url = chartRef.current.toBase64Image('image/png', 1.0);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || 'chart.png';
  a.click();
};

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

  // Chart refs for PNG export
  const overloadChartRef = useRef(null);
  const forecastChartRef = useRef(null);

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

  /* ── Per-resource overload data for grouped bar chart ───────────────────── */
  const overloadBarData = (r && p) ? {
    labels: ['CPU', 'Memory', 'Network', 'Any Resource'],
    datasets: [
      {
        label: 'Reactive (Baseline)',
        data: [
          r.overload_cpu_count    ?? 0,
          r.overload_memory_count ?? 0,
          r.overload_network_count?? 0,
          r.overload_events       ?? 0,
        ],
        backgroundColor: 'rgba(229, 229, 229, 0.7)',
        borderColor: 'rgba(229, 229, 229, 1)',
        borderWidth: 1,
      },
      {
        label: 'Predictive (Proposed)',
        data: [
          p.overload_cpu_count    ?? 0,
          p.overload_memory_count ?? 0,
          p.overload_network_count?? 0,
          p.overload_events       ?? 0,
        ],
        backgroundColor: 'rgba(120, 120, 120, 0.7)',
        borderColor: 'rgba(120, 120, 120, 1)',
        borderWidth: 1,
      },
    ],
  } : null;

  const overloadBarOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#9a9a9a', font: { size: 11, family: 'Inter' } } },
      tooltip: {
        backgroundColor: '#111', borderColor: '#2a2a2a', borderWidth: 1,
        titleColor: '#e5e5e5', bodyColor: '#9a9a9a',
      },
    },
    scales: {
      x: { grid: { color: '#1a1a1a' }, ticks: { color: '#666', font: { size: 11 } } },
      y: {
        grid: { color: '#1a1a1a' },
        ticks: { color: '#666', font: { size: 11, family: 'JetBrains Mono' } },
        beginAtZero: true,
      },
    },
  };

  /* ── Reduction % for each resource ─────────────────────────────────────── */
  const reduction = (rVal, pVal) =>
    rVal > 0 ? ((rVal - pVal) / rVal * 100).toFixed(0) : '—';

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
          Experimental results — reactive vs predictive scheduling (multi-resource) and model accuracy comparison.
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
              {/* Summary metrics table */}
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
                      {metricRow('Overload Events (Any)', r.overload_events, p.overload_events, '', 'lower')}
                      {metricRow('Overload Rate', r.overload_rate, p.overload_rate, '%', 'lower')}
                      {metricRow('Avg CPU Utilisation', r.avg_cpu, p.avg_cpu, '%', 'lower')}
                      {metricRow('Avg Memory Utilisation', r.avg_memory, p.avg_memory, '%', 'lower')}
                      {metricRow('Avg Network Utilisation', r.avg_network, p.avg_network, '%', 'lower')}
                      {metricRow('Total Cost', r.total_cost, p.total_cost, '', 'lower')}
                      {metricRow('Scale-Up Actions', r.scale_up_count, p.scale_up_count, '', 'lower')}
                      {metricRow('Scale-Down Actions', r.scale_down_count, p.scale_down_count, '')}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Per-resource overload grouped bar chart */}
              <div className="card">
                <div className="section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>Per-Resource Overload Breakdown</span>
                  <button className="btn btn-outline btn-sm"
                    onClick={() => exportChart(overloadChartRef, 'overload-comparison.png')}>
                    ↓ Export PNG
                  </button>
                </div>
                {overloadBarData && (
                  <div style={{ height: 280 }}>
                    <Bar ref={overloadChartRef} data={overloadBarData} options={overloadBarOpts} />
                  </div>
                )}
                {/* Reduction labels */}
                <div style={{ display: 'flex', gap: 16, marginTop: 16, flexWrap: 'wrap' }}>
                  {[
                    ['CPU', r.overload_cpu_count, p.overload_cpu_count],
                    ['Memory', r.overload_memory_count, p.overload_memory_count],
                    ['Network', r.overload_network_count, p.overload_network_count],
                    ['Any', r.overload_events, p.overload_events],
                  ].map(([label, rv, pv]) => (
                    <div key={label} style={{ flex: 1, textAlign: 'center', padding: '10px 14px',
                                              background: 'var(--bg-input)', borderRadius: 'var(--radius-md)',
                                              border: '1px solid var(--border)' }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase',
                                    letterSpacing: '0.07em', marginBottom: 4 }}>{label}</div>
                      <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--green)' }}>
                        {reduction(rv, pv)}%
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        {rv ?? 0} → {pv ?? 0}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Summary cards */}
              <div className="grid-2">
                <div className="research-card">
                  <h3>Overall Overload Reduction</h3>
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
                  <h3>Resource Stability</h3>
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
                      {['gbr', 'lstm', 'arima', 'combined'].map(mt => {
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
                  <div className="section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>Forecast vs Actual — {mPattern}</span>
                    <button className="btn btn-outline btn-sm"
                      onClick={() => exportChart(forecastChartRef, 'forecast-comparison.png')}>
                      ↓ Export PNG
                    </button>
                  </div>
                  <ForecastChart ref={forecastChartRef} data={mResult.chart} />
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
