// src/pages/AnomalyLogPage.jsx — Phase 2: Anomaly detection log
import { useEffect, useState } from 'react';
import { anomalyAPI } from '../services/api';

export default function AnomalyLogPage() {
  const [logs, setLogs]       = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([anomalyAPI.logs(), anomalyAPI.summary()])
      .then(([l, s]) => {
        setLogs(l.data);
        setSummary(s.data);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Anomaly Detection Log</div>
        <div className="page-subtitle">
          Phase 2 — Rolling Z-Score + Isolation Forest anomaly detection across CPU, memory, and network I/O.
        </div>
      </div>

      {loading ? (
        <div className="empty-state"><span className="loading-spinner" /></div>
      ) : (
        <>
          {/* Summary KPIs */}
          {summary && (
            <div className="card">
              <div className="section-title">Detection Summary</div>
              <div className="grid-3">
                <div style={{ background:'var(--bg-input)', border:'1px solid var(--border-subtle)',
                              borderRadius:'var(--radius-md)', padding:'14px 18px', textAlign:'center' }}>
                  <div style={{ fontSize:11, color:'var(--text-muted)', textTransform:'uppercase',
                                letterSpacing:'0.08em', marginBottom:6 }}>Total Checked</div>
                  <div style={{ fontSize:20, fontWeight:800 }}>{summary.total_checked}</div>
                </div>
                <div style={{ background:'var(--bg-input)', border:'1px solid var(--border-subtle)',
                              borderRadius:'var(--radius-md)', padding:'14px 18px', textAlign:'center' }}>
                  <div style={{ fontSize:11, color:'var(--text-muted)', textTransform:'uppercase',
                                letterSpacing:'0.08em', marginBottom:6 }}>Anomalies Found</div>
                  <div style={{ fontSize:20, fontWeight:800, color:'var(--red)' }}>{summary.total_anomalies}</div>
                </div>
                <div style={{ background:'var(--bg-input)', border:'1px solid var(--border-subtle)',
                              borderRadius:'var(--radius-md)', padding:'14px 18px', textAlign:'center' }}>
                  <div style={{ fontSize:11, color:'var(--text-muted)', textTransform:'uppercase',
                                letterSpacing:'0.08em', marginBottom:6 }}>Anomaly Rate</div>
                  <div style={{ fontSize:20, fontWeight:800, color:'var(--yellow)' }}>{summary.anomaly_rate}%</div>
                </div>
              </div>
            </div>
          )}

          {/* Methodology */}
          <div className="card">
            <div className="section-title">Detection Methodology</div>
            <div className="grid-2">
              <div style={{ padding:'14px 18px', background:'var(--bg-input)',
                            borderRadius:'var(--radius-md)', border:'1px solid var(--border)' }}>
                <div style={{ fontFamily:'JetBrains Mono, monospace', fontSize:11,
                              color:'var(--text-muted)', marginBottom:4 }}>Strategy 1</div>
                <div style={{ fontWeight:700, fontSize:13, marginBottom:6 }}>Rolling Z-Score</div>
                <div style={{ fontSize:12, color:'var(--text-secondary)', lineHeight:1.7 }}>
                  Computes z-scores over a sliding window of 30 steps.
                  Any resource exceeding z &gt; 3.0 triggers an anomaly flag.
                  Effective for sudden point anomalies and contextual deviations.
                </div>
              </div>
              <div style={{ padding:'14px 18px', background:'var(--bg-input)',
                            borderRadius:'var(--radius-md)', border:'1px solid var(--border)' }}>
                <div style={{ fontFamily:'JetBrains Mono, monospace', fontSize:11,
                              color:'var(--text-muted)', marginBottom:4 }}>Strategy 2</div>
                <div style={{ fontWeight:700, fontSize:13, marginBottom:6 }}>Isolation Forest</div>
                <div style={{ fontSize:12, color:'var(--text-secondary)', lineHeight:1.7 }}>
                  Unsupervised ensemble trained on the 3-D feature space (CPU, memory, network).
                  Contamination factor: 5%. 200 estimators. Detects structural
                  anomalies across correlated resource dimensions.
                </div>
              </div>
            </div>
          </div>

          {/* Log entries */}
          <div className="card">
            <div className="section-title" style={{ display:'flex', justifyContent:'space-between' }}>
              <span>Detection Log</span>
              <span className="badge badge-blue">{logs.length} entries</span>
            </div>
            {logs.length === 0 ? (
              <div className="empty-state">
                <span className="empty-state-icon">—</span>
                <span>No anomaly detection logs yet. Run a scheduler comparison to generate entries.</span>
              </div>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>t</th><th>CPU</th><th>Memory</th><th>Network</th>
                      <th>Z-Score</th><th>IForest</th><th>Anomaly</th><th>Pattern</th>
                    </tr>
                  </thead>
                  <tbody>
                    {logs.slice(0, 200).map(l => (
                      <tr key={l.id}>
                        <td style={{ fontSize:11 }}>{l.time_step}</td>
                        <td>{l.cpu_usage?.toFixed(1)}%</td>
                        <td>{l.memory_usage?.toFixed(1)}%</td>
                        <td>{l.network_io?.toFixed(1)}%</td>
                        <td>{l.z_score_flag ? <span style={{ color:'var(--yellow)' }}>●</span> : ''}</td>
                        <td>{l.iforest_flag ? <span style={{ color:'var(--red)' }}>●</span> : ''}</td>
                        <td>
                          <span className={`badge ${l.is_anomaly ? 'badge-red' : 'badge-green'}`}>
                            {l.is_anomaly ? 'Yes' : 'No'}
                          </span>
                        </td>
                        <td><span className="badge badge-blue">{l.pattern}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {logs.length > 200 && (
                  <div style={{ fontSize:12, color:'var(--text-muted)', textAlign:'center', padding:8 }}>
                    Showing first 200 of {logs.length} entries
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
