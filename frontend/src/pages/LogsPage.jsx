// src/pages/LogsPage.jsx
import { useEffect, useState } from 'react';
import { schedulerAPI, evaluationAPI } from '../services/api';

const ACTION_COLOR = {
  scale_up:   'var(--green)',
  scale_down: 'var(--yellow)',
  hold:       'var(--text-muted)',
};

export default function LogsPage() {
  const [runs,     setRuns]     = useState([]);
  const [evals,    setEvals]    = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail,   setDetail]   = useState(null);
  const [loading,  setLoading]  = useState(true);
  const [tab,      setTab]      = useState('runs');

  useEffect(() => {
    Promise.all([
      schedulerAPI.listRuns(),
      evaluationAPI.list(),
    ]).then(([r, e]) => {
      setRuns(r.data);
      setEvals(e.data);
    }).catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleSelectRun = (run) => {
    setSelected(run);
    schedulerAPI.getRun(run.id).then(r => setDetail(r.data)).catch(console.error);
  };

  return (
    <div>
      <div className="page-header">
        <div className="page-title">📋 Run Logs</div>
        <div className="page-subtitle">History of all scheduler runs and step-by-step actions.</div>
      </div>

      <div className="tabs">
        <button className={`tab-btn ${tab==='runs' ?'active':''}`} onClick={()=>setTab('runs')}>Scheduler Runs</button>
        <button className={`tab-btn ${tab==='evals'?'active':''}`} onClick={()=>setTab('evals')}>Evaluation Results</button>
      </div>

      {loading ? <div className="empty-state"><span className="loading-spinner"/></div> : (
        <>
          {/* Scheduler Runs tab */}
          {tab === 'runs' && (
            <div className="grid-2">
              <div className="card" style={{ maxHeight:600, overflow:'auto' }}>
                <div className="section-title" style={{display:'flex',justifyContent:'space-between'}}>
                  <span>All Runs</span>
                  <span className="badge badge-blue">{runs.length}</span>
                </div>
                {runs.length===0 ? (
                  <div className="empty-state">
                    <span className="empty-state-icon">📋</span>
                    <span>No runs yet.</span>
                  </div>
                ) : runs.map(r => (
                  <div key={r.id}
                    onClick={() => handleSelectRun(r)}
                    style={{
                      padding:'10px 14px',
                      borderRadius:'var(--radius-md)',
                      cursor:'pointer',
                      marginBottom:4,
                      background: selected?.id===r.id ? 'var(--bg-card-hover)' : 'var(--bg-input)',
                      border:`1px solid ${selected?.id===r.id ? 'var(--accent)' : 'var(--border-subtle)'}`,
                      transition:'all 0.15s',
                    }}>
                    <div style={{display:'flex', justifyContent:'space-between', marginBottom:4}}>
                      <span style={{fontWeight:600, fontSize:13}}>
                        #{r.id} — <span style={{color: r.scheduler_type==='reactive'?'var(--red)':'var(--green)'}}>
                          {r.scheduler_type}</span>
                      </span>
                      <span className={`badge ${r.pattern==='spike'?'badge-red':r.pattern==='gradual'?'badge-green':r.pattern==='periodic'?'badge-yellow':'badge-purple'}`}>
                        {r.pattern}
                      </span>
                    </div>
                    <div style={{fontSize:12, color:'var(--text-muted)', display:'flex', gap:12}}>
                      <span>⚠ {r.overload_events} overloads</span>
                      <span>💻 {r.avg_cpu?.toFixed(1)}% cpu</span>
                      <span>💰 {r.total_cost?.toFixed(0)} cost</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Detail panel */}
              <div className="card" style={{ maxHeight:600, overflow:'auto' }}>
                <div className="section-title">Step-by-Step Actions</div>
                {!detail ? (
                  <div className="empty-state">
                    <span className="empty-state-icon">👆</span>
                    <span>Select a run to see actions</span>
                  </div>
                ) : (
                  <>
                    <div style={{display:'flex', gap:8, marginBottom:12, flexWrap:'wrap'}}>
                      <span className={`badge ${detail.scheduler_type==='reactive'?'badge-red':'badge-green'}`}>
                        {detail.scheduler_type}
                      </span>
                      <span className="badge badge-blue">{detail.pattern} · {detail.steps} steps</span>
                      <span className="badge badge-yellow">⚠ {detail.overload_events} overloads</span>
                    </div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr><th>t</th><th>Workload</th><th>Cap</th><th>CPU%</th><th>Action</th><th>OL</th></tr>
                        </thead>
                        <tbody>
                          {(detail.actions || []).slice(0,200).map(a => (
                            <tr key={a.time_step}>
                              <td style={{fontSize:11}}>{a.time_step}</td>
                              <td>{a.workload?.toFixed(1)}</td>
                              <td>{a.capacity}</td>
                              <td>{a.cpu_usage?.toFixed(1)}%</td>
                              <td style={{color: ACTION_COLOR[a.action], fontWeight:600, fontSize:11}}>
                                {a.action==='scale_up'?' ↑ UP':a.action==='scale_down'?' ↓ DOWN':' — hold'}
                              </td>
                              <td>{a.overloaded ? <span style={{color:'var(--red)'}}>🔴</span> : ''}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {detail.actions?.length > 200 && (
                        <div style={{fontSize:12, color:'var(--text-muted)', textAlign:'center', padding:8}}>
                          Showing first 200 of {detail.actions.length} steps
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Evaluation Results tab */}
          {tab === 'evals' && (
            <div className="card">
              <div className="section-title" style={{display:'flex',justifyContent:'space-between'}}>
                <span>Saved Evaluation Results</span>
                <span className="badge badge-blue">{evals.length} evaluations</span>
              </div>
              {evals.length === 0 ? (
                <div className="empty-state">
                  <span className="empty-state-icon">📊</span>
                  <span>No evaluations yet — go to Comparison and click "Save Evaluation".</span>
                </div>
              ) : (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>ID</th><th>Pattern</th><th>Steps</th>
                        <th>R-Overloads</th><th>P-Overloads</th>
                        <th>Overload Reduction</th><th>Cost Diff</th><th>Created</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evals.map(e => (
                        <tr key={e.id}>
                          <td>#{e.id}</td>
                          <td><span className={`badge ${e.pattern==='spike'?'badge-red':e.pattern==='gradual'?'badge-green':e.pattern==='periodic'?'badge-yellow':'badge-purple'}`}>{e.pattern}</span></td>
                          <td>{e.steps}</td>
                          <td style={{color:'var(--red)', fontWeight:600}}>{e.r_overload_events}</td>
                          <td style={{color:'var(--green)', fontWeight:600}}>{e.p_overload_events}</td>
                          <td>
                            <span className={`badge ${e.overload_reduction>0?'badge-green':'badge-yellow'}`}>
                              {e.overload_reduction>0?'+':''}{e.overload_reduction}%
                            </span>
                          </td>
                          <td style={{color: e.cost_difference>0?'var(--red)':'var(--green)'}}>
                            {e.cost_difference>0?'+':''}{e.cost_difference?.toFixed(1)}
                          </td>
                          <td style={{fontSize:12}}>{new Date(e.created_at).toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
