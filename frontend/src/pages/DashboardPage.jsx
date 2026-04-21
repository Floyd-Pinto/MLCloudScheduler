// src/pages/DashboardPage.jsx
import { useEffect, useState, useCallback } from 'react';
import { metricsAPI, mlAPI, simulationAPI, schedulerAPI } from '../services/api';

const MODEL_COLORS = {
  gbr:      { label: 'GBR',      color: 'var(--accent)',  icon: '🌲' },
  lstm:     { label: 'LSTM',     color: 'var(--purple)', icon: '🧠' },
  arima:    { label: 'ARIMA',    color: 'var(--yellow)', icon: '📈' },
  combined: { label: 'Combined', color: 'var(--green)',  icon: '⚡' },
};

function KpiCard({ label, value, sub, color = 'blue', icon }) {
  return (
    <div className={`kpi-card ${color}`}>
      <div className="kpi-label">{icon} {label}</div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

function SchedulerRow({ label, overload, cpu, cost, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '10px 0', borderBottom: '1px solid var(--border-subtle)' }}>
      <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: 13 }}>{label}</span>
      <div style={{ display: 'flex', gap: 24 }}>
        <span style={{ fontSize: 13, color }}>{overload != null ? overload + '%' : '—'} overload</span>
        <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{cpu != null ? cpu + '%' : '—'} avg CPU</span>
        <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>cost {cost != null ? cost : '—'}</span>
      </div>
    </div>
  );
}

function ModelStatusRow({ modelKey, info }) {
  const meta  = MODEL_COLORS[modelKey];
  const ready = info?.ready;
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '9px 0', borderBottom: '1px solid var(--border-subtle)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span>{meta.icon}</span>
        <span style={{ fontWeight: 600, fontSize: 13, color: meta.color }}>{meta.label}</span>
        <span className={`badge ${ready ? 'badge-green' : 'badge-yellow'}`} style={{ fontSize: 10 }}>
          {ready ? '✓ Ready' : '⚠ Needs Training'}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 18 }}>
        {info?.r2  != null && <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>R²&nbsp;<strong style={{ color: meta.color }}>{info.r2.toFixed(4)}</strong></span>}
        {info?.rmse != null && <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>RMSE&nbsp;<strong style={{ color: 'var(--text-secondary)' }}>{info.rmse.toFixed(4)}</strong></span>}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [summary,  setSummary]  = useState(null);
  const [mlStatus, setMlStatus] = useState(null);
  const [simCount, setSimCount] = useState('—');
  const [runCount, setRunCount] = useState('—');
  const [loading,  setLoading]  = useState(true);
  const [lastRefresh, setLastRefresh] = useState(null);

  const fetchAll = useCallback(() => {
    setLoading(true);
    Promise.all([
      metricsAPI.summary(),
      mlAPI.status(),
      simulationAPI.listRuns(),
      schedulerAPI.listRuns(),
    ]).then(([sum, ml, sims, runs]) => {
      setSummary(sum.data);
      setMlStatus(ml.data);
      setSimCount(sims.data.length);
      setRunCount(runs.data.length);
      setLastRefresh(new Date().toLocaleTimeString());
    }).catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // Fetch on mount + auto-refresh every 30 seconds
  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 30000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  const r = summary?.reactive;
  const p = summary?.predictive;

  // New API shape: mlStatus.statuses.gbr, mlStatus.gbr.r2 etc.
  const statuses  = mlStatus?.statuses || {};
  const readyCount = Object.values(statuses).filter(Boolean).length;
  const bestModel = ['gbr', 'lstm', 'arima', 'combined']
    .filter(m => mlStatus?.[m]?.r2 != null)
    .sort((a, b) => (mlStatus[b].r2 || 0) - (mlStatus[a].r2 || 0))[0];

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <div className="page-title">Dashboard</div>
            <div className="page-subtitle">ML-Based Adaptive Cloud Resource Scheduling — System Overview</div>
          </div>
          <button className="btn btn-outline btn-sm" onClick={fetchAll} disabled={loading}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {loading ? <><span className="loading-spinner" />&nbsp;Refreshing…</> : '↻ Refresh'}
          </button>
        </div>
        {lastRefresh && (
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
            Last updated: {lastRefresh} · auto-refreshes every 30s
          </div>
        )}
      </div>

      {/* Hero banner */}
      <div className="card" style={{
        background: 'linear-gradient(135deg, #0f1f3d 0%, #1a2236 100%)',
        border: '1px solid var(--border)', marginBottom: 24,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexWrap: 'wrap', gap: 16,
      }}>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 800, color: 'var(--text-primary)', marginBottom: 6 }}>
            Adaptive Cloud Resource Scheduler
          </h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', maxWidth: 520 }}>
            Compares <strong style={{ color: 'var(--red)' }}>reactive</strong> threshold-based scheduling
            against <strong style={{ color: 'var(--green)' }}>ML-predictive</strong> scheduling powered by
            GBR · LSTM · ARIMA · Combined ensemble forecasting.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {['gbr', 'lstm', 'arima', 'combined'].map(m => (
            <span key={m}
              className={`badge ${statuses[m] ? 'badge-green' : 'badge-yellow'}`}>
              {MODEL_COLORS[m].icon} {MODEL_COLORS[m].label} {statuses[m] ? 'Ready' : '—'}
            </span>
          ))}
          <span className="badge badge-blue">✓ DRF Backend</span>
        </div>
      </div>

      {/* KPI Row */}
      <div className="kpi-grid">
        <KpiCard icon="🔬" label="Simulations Run"
          value={loading ? '…' : simCount}
          color="blue" sub="workload patterns generated" />
        <KpiCard icon="▶" label="Scheduler Runs"
          value={loading ? '…' : runCount}
          color="purple" sub="reactive + predictive combined" />
        <KpiCard icon="🤖" label="Models Ready"
          value={loading ? '…' : `${readyCount}/4`}
          color={readyCount === 4 ? 'green' : readyCount > 0 ? 'yellow' : 'red'}
          sub={bestModel ? `Best: ${bestModel.toUpperCase()} R²=${mlStatus?.[bestModel]?.r2?.toFixed(3)}` : 'Train models to see metrics'} />
        <KpiCard icon="⚠" label="Avg Overload (Reactive)"
          value={loading ? '…' : (r?.avg_overload != null ? r.avg_overload.toFixed(1) + '%' : '—')}
          color="red" sub="across all reactive runs" />
        <KpiCard icon="✅" label="Avg Overload (Predictive)"
          value={loading ? '…' : (p?.avg_overload != null ? p.avg_overload.toFixed(1) + '%' : '—')}
          color="green" sub="across all predictive runs" />
      </div>

      {/* Scheduler + ML Model panels */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        <div className="card">
          <div className="section-title">Scheduler Performance Summary</div>
          {loading ? <div className="loading-spinner" /> : (
            <>
              <SchedulerRow label="🔴 Reactive"
                overload={r?.avg_overload?.toFixed(1)}
                cpu={r?.avg_cpu?.toFixed(1)}
                cost={r?.total_cost?.toFixed(0)}
                color="var(--red)" />
              <SchedulerRow label="🟢 Predictive"
                overload={p?.avg_overload?.toFixed(1)}
                cpu={p?.avg_cpu?.toFixed(1)}
                cost={p?.total_cost?.toFixed(0)}
                color="var(--green)" />
              {r && p && (
                <div style={{ marginTop: 14, padding: '10px 14px',
                              background: 'var(--accent-glow)', borderRadius: 'var(--radius-md)',
                              fontSize: 12, color: 'var(--text-accent)' }}>
                  💡 Predictive scheduling forecasts 5 steps ahead and scales pre-emptively,
                  reducing overload by {r.avg_overload > 0
                    ? ((r.avg_overload - p.avg_overload) / r.avg_overload * 100).toFixed(1) + '%'
                    : '—'}.
                </div>
              )}
            </>
          )}
        </div>

        <div className="card">
          <div className="section-title">ML Model Status</div>
          {loading ? <div className="loading-spinner" /> : mlStatus ? (
            <>
              {['gbr', 'lstm', 'arima', 'combined'].map(m => (
                <ModelStatusRow key={m} modelKey={m} info={mlStatus[m]} />
              ))}
              {readyCount === 0 && (
                <div className="empty-state" style={{ marginTop: 12 }}>
                  <span className="empty-state-icon">🧠</span>
                  <span>No models trained yet. Go to Model Training.</span>
                </div>
              )}
            </>
          ) : <div className="empty-state"><span>No data yet</span></div>}
        </div>
      </div>

      {/* Architecture summary */}
      <div className="card">
        <div className="section-title">System Architecture</div>
        <div className="grid-3">
          {[
            { icon: '⚡', title: 'Workload Simulator',
              desc: 'Generates gradual, spike, periodic & combined synthetic cloud workload patterns with configurable params.',
              color: 'var(--accent)' },
            { icon: '🧠', title: 'ML Forecasting Engine',
              desc: 'GBR · PyTorch LSTM · ARIMA · Combined ensemble — 4 models compared side-by-side. Predicts load 5 steps ahead.',
              color: 'var(--purple)' },
            { icon: '⚖️', title: 'Scheduler Comparison',
              desc: 'Reactive (threshold-based) vs Predictive (ML-based) tracking overload events, cost, and CPU utilization.',
              color: 'var(--green)' },
          ].map(({ icon, title, desc, color }) => (
            <div key={title} style={{ padding: '18px 20px', background: 'var(--bg-input)',
                                      borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: 24, marginBottom: 10 }}>{icon}</div>
              <div style={{ fontWeight: 700, color, fontSize: 14, marginBottom: 6 }}>{title}</div>
              <div style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.7 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
