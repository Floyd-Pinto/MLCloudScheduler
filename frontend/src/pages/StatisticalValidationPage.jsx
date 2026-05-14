// src/pages/StatisticalValidationPage.jsx — 20-Run LSTM vs All Statistical Validation
import { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  BarElement, Legend, Tooltip,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { mlAPI } from '../services/api';

ChartJS.register(CategoryScale, LinearScale, BarElement, Legend, Tooltip);

const MODEL_COLORS = {
  lstm:     'rgba(168, 130, 255, 0.8)',
  gbr:      'rgba(100, 180, 255, 0.8)',
  arima:    'rgba(255, 200, 80, 0.8)',
  combined: 'rgba(100, 220, 160, 0.8)',
};

export default function StatisticalValidationPage() {
  const [running, setRunning]   = useState(false);
  const [result, setResult]     = useState(null);
  const [history, setHistory]   = useState([]);
  const [nRuns, setNRuns]       = useState(20);

  useEffect(() => {
    mlAPI.getStatisticalValidation().then(r => setHistory(r.data)).catch(() => {});
  }, []);

  const runValidation = async () => {
    setRunning(true);
    toast('🔬 Running statistical validation… This may take ~2 minutes.', { duration: 5000 });
    try {
      const res = await mlAPI.runStatisticalValidation({ n_runs: nRuns });
      setResult(res.data);
      toast.success(`Validation complete — LSTM wins ${(res.data.lstm_win_rate * 100).toFixed(0)}% of runs`);
      mlAPI.getStatisticalValidation().then(r => setHistory(r.data)).catch(() => {});
    } catch (e) {
      toast.error(e.response?.data?.error || 'Validation failed');
    }
    setRunning(false);
  };

  const loadHistoryEntry = (entry) => {
    if (entry.full_results) {
      setResult(entry.full_results);
      toast.success('Loaded previous validation');
    }
  };

  const downloadCSV = () => {
    if (!result?.runs) return;
    const headers = ['run', 'seed', 'lstm_r2', 'gbr_r2', 'arima_r2', 'combined_r2', 'winner'];
    const rows = result.runs.map(r =>
      [r.run, r.seed, r.lstm_r2, r.gbr_r2, r.arima_r2, r.combined_r2, r.winner].join(',')
    );
    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'statistical_validation_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const summary = result?.summary || {};
  const pairwise = result?.pairwise || {};

  /* ── Mean R² bar chart data ────────────────────────────────────────────── */
  const barData = summary.lstm ? {
    labels: ['LSTM', 'GBR', 'ARIMA', 'Combined'],
    datasets: [{
      label: 'Mean R² ± Std',
      data: [
        summary.lstm?.mean_r2,
        summary.gbr?.mean_r2,
        summary.arima?.mean_r2,
        summary.combined?.mean_r2,
      ],
      backgroundColor: [
        MODEL_COLORS.lstm, MODEL_COLORS.gbr, MODEL_COLORS.arima, MODEL_COLORS.combined,
      ],
      borderColor: [
        MODEL_COLORS.lstm, MODEL_COLORS.gbr, MODEL_COLORS.arima, MODEL_COLORS.combined,
      ].map(c => c.replace('0.8', '1')),
      borderWidth: 2,
    }],
  } : null;

  const barOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#111', borderColor: '#2a2a2a', borderWidth: 1,
        titleColor: '#e5e5e5', bodyColor: '#9a9a9a',
        callbacks: {
          label: (ctx) => {
            const mt = ['lstm', 'gbr', 'arima', 'combined'][ctx.dataIndex];
            const s = summary[mt];
            return s ? `R²: ${s.mean_r2?.toFixed(4)} ± ${s.std_r2?.toFixed(4)}` : '';
          },
        },
      },
    },
    scales: {
      x: { grid: { color: '#1a1a1a' }, ticks: { color: '#999', font: { size: 12, weight: 'bold' } } },
      y: {
        grid: { color: '#1a1a1a' },
        ticks: { color: '#666', font: { size: 11, family: 'JetBrains Mono' } },
        beginAtZero: false,
        suggestedMin: 0.3,
      },
    },
  };

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Statistical Validation</div>
        <div className="page-subtitle">
          Run multiple independent experiments to statistically validate LSTM superiority across diverse workloads.
        </div>
      </div>

      {/* Controls */}
      <div className="card">
        <div className="section-title">Run Validation Experiment</div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ marginBottom: 0, width: 120 }}>
            <label className="form-label">Number of Runs</label>
            <input type="number" className="form-control" min={5} max={50}
              value={nRuns} onChange={e => setNRuns(+e.target.value)} />
          </div>
          <button className="btn btn-primary btn-lg" disabled={running} onClick={runValidation}>
            {running ? <><span className="loading-spinner" />&nbsp;Running {nRuns} experiments…</> : `Run ${nRuns}-Run Validation`}
          </button>
        </div>
        <div style={{ marginTop: 12, fontSize: 12, color: 'var(--text-muted)' }}>
          Each run generates a new workload with a unique seed, evaluates all 4 models, and records R²/RMSE/MAE.
          Estimated time: ~{Math.ceil(nRuns * 6 / 60)} minutes.
        </div>
      </div>

      {result && (
        <>
          {/* Summary Cards */}
          <div className="grid-2" style={{ marginTop: 0 }}>
            <div className="card" style={{ borderTop: '3px solid rgba(168, 130, 255, 0.8)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                LSTM Win Rate
              </div>
              <div style={{ fontSize: 36, fontWeight: 800, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
                {result.win_counts?.lstm ?? 0}/{result.n_runs}
                <span style={{ fontSize: 18, color: 'var(--text-muted)', marginLeft: 8 }}>
                  ({(result.lstm_win_rate * 100).toFixed(0)}%)
                </span>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                LSTM achieved highest R² in {result.win_counts?.lstm ?? 0} out of {result.n_runs} independent runs
              </div>
            </div>

            <div className="card">
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                Performance Summary
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
                {['lstm', 'gbr', 'arima', 'combined'].map(mt => (
                  <div key={mt} style={{
                    padding: '10px 14px', background: 'var(--bg-input)', borderRadius: 'var(--radius-sm)',
                    borderLeft: `3px solid ${MODEL_COLORS[mt]}`,
                  }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                      {mt.toUpperCase()}
                    </div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
                      {summary[mt]?.mean_r2?.toFixed(4) ?? '—'}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      ± {summary[mt]?.std_r2?.toFixed(4) ?? '—'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Pairwise Comparisons */}
          <div className="card">
            <div className="section-title">Pairwise: LSTM vs Others</div>
            <div className="grid-3">
              {Object.entries(pairwise).map(([key, val]) => {
                const other = key.replace('lstm_vs_', '').toUpperCase();
                return (
                  <div key={key} style={{
                    padding: '14px 18px', background: 'var(--bg-input)', borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border)',
                  }}>
                    <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>
                      LSTM vs {other}
                    </div>
                    <div style={{ fontSize: 22, fontWeight: 800, color: val.lstm_win_pct >= 50 ? 'var(--green)' : 'var(--red)' }}>
                      {val.lstm_win_pct}%
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {val.lstm_wins}/{val.total_runs} runs · Mean gap: {val.mean_diff?.toFixed(4) ?? '—'}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Bar Chart */}
          {barData && (
            <div className="card">
              <div className="section-title">Mean R² Comparison (Error Bars)</div>
              <div style={{ height: 300 }}>
                <Bar data={barData} options={barOpts} />
              </div>
              <div style={{ display: 'flex', gap: 16, marginTop: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
                {['lstm', 'gbr', 'arima', 'combined'].map(mt => (
                  <div key={mt} style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    <span style={{
                      display: 'inline-block', width: 10, height: 10,
                      borderRadius: 2, marginRight: 4,
                      backgroundColor: MODEL_COLORS[mt],
                    }} />
                    {mt.toUpperCase()}: {summary[mt]?.mean_r2?.toFixed(4)} ± {summary[mt]?.std_r2?.toFixed(4)}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Detailed per-run table */}
          <div className="card">
            <div className="section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Detailed Results — {result.n_runs} Runs</span>
              <button className="btn btn-outline btn-sm" onClick={downloadCSV}>
                ↓ Download CSV
              </button>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Run #</th><th>Seed</th>
                    <th>LSTM R²</th><th>GBR R²</th><th>ARIMA R²</th><th>Combined R²</th>
                    <th>Winner</th>
                  </tr>
                </thead>
                <tbody>
                  {(result.runs || []).map(r => (
                    <tr key={r.run}>
                      <td>{r.run}</td>
                      <td style={{ color: 'var(--text-muted)' }}>{r.seed}</td>
                      <td className="metric-value" style={{ color: r.winner === 'lstm' ? 'var(--green)' : 'var(--text-secondary)', fontWeight: r.winner === 'lstm' ? 700 : 400 }}>
                        {r.lstm_r2?.toFixed(4) ?? '—'}
                      </td>
                      <td className="metric-value" style={{ color: r.winner === 'gbr' ? 'var(--green)' : 'var(--text-secondary)', fontWeight: r.winner === 'gbr' ? 700 : 400 }}>
                        {r.gbr_r2?.toFixed(4) ?? '—'}
                      </td>
                      <td className="metric-value" style={{ color: r.winner === 'arima' ? 'var(--green)' : 'var(--text-secondary)', fontWeight: r.winner === 'arima' ? 700 : 400 }}>
                        {r.arima_r2?.toFixed(4) ?? '—'}
                      </td>
                      <td className="metric-value" style={{ color: r.winner === 'combined' ? 'var(--green)' : 'var(--text-secondary)', fontWeight: r.winner === 'combined' ? 700 : 400 }}>
                        {r.combined_r2?.toFixed(4) ?? '—'}
                      </td>
                      <td>
                        <span className={`badge ${r.winner === 'lstm' ? 'badge-purple' : r.winner === 'combined' ? 'badge-green' : 'badge-blue'}`}>
                          {r.winner?.toUpperCase() ?? '—'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: 12, fontSize: 12, color: 'var(--text-muted)', textAlign: 'center' }}>
              Completed in {result.elapsed_seconds?.toFixed(1)}s · Pattern: {result.pattern} · {result.steps} steps per run
            </div>
          </div>
        </>
      )}

      {/* Past validation history */}
      {history.length > 0 && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="section-title">
            Previous Validation Runs
            <span className="badge badge-blue" style={{ marginLeft: 10 }}>{history.length}</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th><th>Runs</th><th>LSTM Win %</th>
                  <th>LSTM R²</th><th>GBR R²</th><th>ARIMA R²</th><th>Combined R²</th>
                  <th>Time</th><th></th>
                </tr>
              </thead>
              <tbody>
                {history.map(h => (
                  <tr key={h.id}>
                    <td>#{h.id}</td>
                    <td>{h.n_runs}</td>
                    <td style={{ fontWeight: 700, color: (h.lstm_win_rate ?? 0) >= 0.8 ? 'var(--green)' : 'var(--yellow)' }}>
                      {h.lstm_win_rate != null ? (h.lstm_win_rate * 100).toFixed(0) + '%' : '—'}
                    </td>
                    <td className="metric-value">{h.lstm_mean_r2?.toFixed(4) ?? '—'}</td>
                    <td className="metric-value">{h.gbr_mean_r2?.toFixed(4) ?? '—'}</td>
                    <td className="metric-value">{h.arima_mean_r2?.toFixed(4) ?? '—'}</td>
                    <td className="metric-value">{h.combined_mean_r2?.toFixed(4) ?? '—'}</td>
                    <td style={{ fontSize: 12 }}>{h.created_at ? new Date(h.created_at).toLocaleString() : '—'}</td>
                    <td>
                      <button className="btn btn-outline btn-sm" onClick={() => loadHistoryEntry(h)}>Load</button>
                    </td>
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
