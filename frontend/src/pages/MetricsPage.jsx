// src/pages/MetricsPage.jsx
import { useEffect, useState } from 'react';
import { metricsAPI } from '../services/api';

function StatBox({ label, value, color='var(--text-primary)' }) {
  return (
    <div style={{ background:'var(--bg-input)', border:'1px solid var(--border-subtle)',
                  borderRadius:'var(--radius-md)', padding:'14px 18px', textAlign:'center' }}>
      <div style={{ fontSize:11, color:'var(--text-muted)', textTransform:'uppercase',
                    letterSpacing:'0.08em', marginBottom:6 }}>{label}</div>
      <div style={{ fontSize:20, fontWeight:800, color }}>{value ?? '—'}</div>
    </div>
  );
}

export default function MetricsPage() {
  const [records,  setRecords]  = useState([]);
  const [summary,  setSummary]  = useState(null);
  const [filter,   setFilter]   = useState({ type:'', pattern:'' });
  const [loading,  setLoading]  = useState(true);

  useEffect(() => {
    Promise.all([
      metricsAPI.list({}),
      metricsAPI.summary(),
    ]).then(([r, s]) => {
      setRecords(r.data);
      setSummary(s.data);
    }).catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleFilter = () => {
    const params = {};
    if (filter.type)    params.type    = filter.type;
    if (filter.pattern) params.pattern = filter.pattern;
    metricsAPI.list(params).then(r => setRecords(r.data));
  };

  const overall = summary?.overall;
  const react   = summary?.reactive;
  const pred    = summary?.predictive;

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Metrics</div>
        <div className="page-subtitle">
          Aggregated performance metrics across all scheduler runs.
        </div>
      </div>

      {loading ? (
        <div className="empty-state"><span className="loading-spinner"/></div>
      ) : (
        <>
          {/* Summary KPIs */}
          {summary && (
            <div className="card" style={{ marginBottom: 24 }}>
              <div className="section-title">Global Summary</div>
              <div className="grid-3" style={{ marginBottom: 16 }}>
                <StatBox label="Total Scheduler Runs" value={overall?.total_runs} />
                <StatBox label="Avg Overload Rate" value={overall?.avg_overload?.toFixed(2)+'%'} color="var(--red)" />
                <StatBox label="Avg CPU Usage" value={overall?.avg_cpu?.toFixed(2)+'%'} />
              </div>

              <div className="grid-2">
                <div>
                  <div style={{ fontWeight:700, color:'var(--text-secondary)', fontSize:13, marginBottom:8 }}>
                    Reactive (Baseline)
                  </div>
                  <div className="grid-2">
                    <StatBox label="Runs"         value={react?.total_runs} />
                    <StatBox label="Avg Overload" value={react?.avg_overload?.toFixed(2)+'%'} color="var(--red)" />
                    <StatBox label="Avg CPU"      value={react?.avg_cpu?.toFixed(2)+'%'} />
                  </div>
                </div>
                <div>
                  <div style={{ fontWeight:700, color:'var(--text-secondary)', fontSize:13, marginBottom:8 }}>
                    Predictive (Proposed)
                  </div>
                  <div className="grid-2">
                    <StatBox label="Runs"         value={pred?.total_runs} />
                    <StatBox label="Avg Overload" value={pred?.avg_overload?.toFixed(2)+'%'} color="var(--green)" />
                    <StatBox label="Avg CPU"      value={pred?.avg_cpu?.toFixed(2)+'%'} />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Filter + table */}
          <div className="card">
            <div style={{ display:'flex', gap:12, marginBottom:16, alignItems:'flex-end', flexWrap:'wrap' }}>
              <div>
                <label className="form-label">Scheduler Type</label>
                <select className="form-control" style={{width:180}}
                  value={filter.type} onChange={e=>setFilter(f=>({...f, type:e.target.value}))}>
                  <option value="">All</option>
                  <option value="reactive">Reactive</option>
                  <option value="predictive">Predictive</option>
                </select>
              </div>
              <div>
                <label className="form-label">Pattern</label>
                <select className="form-control" style={{width:160}}
                  value={filter.pattern} onChange={e=>setFilter(f=>({...f, pattern:e.target.value}))}>
                  <option value="">All</option>
                  {['gradual','spike','periodic','combined'].map(p =>
                    <option key={p} value={p}>{p}</option>)}
                </select>
              </div>
              <button className="btn btn-outline" onClick={handleFilter}>Apply Filter</button>
            </div>

            <div className="section-title" style={{display:'flex',justifyContent:'space-between'}}>
              <span>Run Records</span>
              <span className="badge badge-blue">{records.length} records</span>
            </div>

            {records.length === 0 ? (
              <div className="empty-state">
                <span className="empty-state-icon">—</span>
                <span>No scheduler runs yet — run a comparison first.</span>
              </div>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>ID</th><th>Type</th><th>Pattern</th><th>Steps</th>
                      <th>Overloads</th><th>Overload%</th><th>Avg CPU</th>
                      <th>Avg Cap</th><th>Cost</th><th>↑ Up</th><th>↓ Down</th>
                    </tr>
                  </thead>
                  <tbody>
                    {records.map(r => (
                      <tr key={r.id}>
                        <td>#{r.id}</td>
                        <td>
                          <span className={`badge ${r.scheduler_type==='reactive'?'badge-red':'badge-green'}`}>
                            {r.scheduler_type}
                          </span>
                        </td>
                        <td>
                          <span className={`badge ${r.pattern==='spike'?'badge-red':r.pattern==='gradual'?'badge-green':r.pattern==='periodic'?'badge-yellow':'badge-purple'}`}>
                            {r.pattern}
                          </span>
                        </td>
                        <td>{r.steps}</td>
                        <td style={{color: r.overload_events>0?'var(--red)':'var(--green)', fontWeight:600}}>
                          {r.overload_events}
                        </td>
                        <td>{r.overload_rate?.toFixed(1)}%</td>
                        <td>{r.avg_cpu?.toFixed(1)}%</td>
                        <td>{r.avg_capacity?.toFixed(1)}</td>
                        <td>{r.total_cost?.toFixed(0)}</td>
                        <td>{r.scale_up_count}</td>
                        <td>{r.scale_down_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
