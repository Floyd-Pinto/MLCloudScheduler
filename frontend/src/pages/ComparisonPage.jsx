// src/pages/ComparisonPage.jsx
// API shape: { reactive: {id, scheduler_type, overload_events, ..., actions:[...]},
//              predictive: {same} }
import { useState } from 'react';
import toast from 'react-hot-toast';
import { schedulerAPI, evaluationAPI } from '../services/api';
import { CpuComparisonChart, CapacityComparisonChart } from '../charts/ComparisonChart';
import BarCompareChart from '../charts/BarCompareChart';

const BAR_METRICS = [
  { key: 'overload_events', label: 'Overloads' },
  { key: 'avg_cpu',         label: 'Avg CPU%'  },
  { key: 'total_cost',      label: 'Total Cost' },
  { key: 'scale_up_count',  label: 'Scale-Ups'  },
];

function SumCard({ label, value, sub, color, icon }) {
  return (
    <div style={{
      background:'var(--bg-input)', border:'1px solid var(--border-subtle)',
      borderRadius:'var(--radius-md)', padding:'14px 18px',
    }}>
      <div style={{fontSize:11, color:'var(--text-muted)', textTransform:'uppercase',
                   letterSpacing:'0.07em', marginBottom:5}}>{icon} {label}</div>
      <div style={{fontSize:22, fontWeight:800, color}}>{value}</div>
      {sub && <div style={{fontSize:11, color:'var(--text-muted)', marginTop:3}}>{sub}</div>}
    </div>
  );
}

export default function ComparisonPage() {
  const [form,       setForm]       = useState({ pattern:'combined', steps:200, seed:42 });
  const [result,     setResult]     = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [activeTab,  setActiveTab]  = useState('cpu');

  const handleCompare = async () => {
    setLoading(true);
    try {
      const res = await schedulerAPI.compare(form);
      setResult(res.data);
      toast.success('Comparison complete!');
    } catch (e) {
      toast.error(e.response?.data?.error || 'Comparison failed');
    } finally { setLoading(false); }
  };

  const handleEval = async () => {
    setLoading(true);
    try {
      await evaluationAPI.run(form);
      toast.success('Evaluation saved to database!');
    } catch (e) {
      toast.error(e.response?.data?.error || 'Evaluation failed');
    } finally { setLoading(false); }
  };

  // API returns flat objects — summary fields are directly on reactive/predictive
  const r = result?.reactive;
  const p = result?.predictive;
  const overloadReduction = r && p && r.overload_events > 0
    ? (((r.overload_events - p.overload_events) / r.overload_events) * 100).toFixed(1)
    : null;

  return (
    <div>
      <div className="page-header">
        <div className="page-title">⚖️ Scheduler Comparison</div>
        <div className="page-subtitle">
          Run reactive vs predictive scheduling on the same workload and compare performance.
        </div>
      </div>

      {/* Controls */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="section-title">Run Configuration</div>
        <div style={{ display:'flex', gap:16, flexWrap:'wrap', alignItems:'flex-end' }}>
          <div className="form-group" style={{flex:1, minWidth:150, marginBottom:0}}>
            <label className="form-label">Pattern</label>
            <select className="form-control" value={form.pattern}
              onChange={e => setForm(f=>({...f, pattern:e.target.value}))}>
              {['gradual','spike','periodic','combined'].map(pat =>
                <option key={pat} value={pat}>{pat}</option>)}
            </select>
          </div>
          <div className="form-group" style={{flex:1, minWidth:120, marginBottom:0}}>
            <label className="form-label">Steps</label>
            <input type="number" className="form-control" min={50} max={1000}
              value={form.steps} onChange={e => setForm(f=>({...f, steps:+e.target.value}))} />
          </div>
          <div className="form-group" style={{flex:1, minWidth:120, marginBottom:0}}>
            <label className="form-label">Seed</label>
            <input type="number" className="form-control" min={0} max={99999}
              value={form.seed} onChange={e => setForm(f=>({...f, seed:+e.target.value}))} />
          </div>
          <div style={{ display:'flex', gap:8, marginBottom:0 }}>
            <button className="btn btn-primary" onClick={handleCompare} disabled={loading}>
              {loading ? <><span className="loading-spinner"/>&nbsp;Running…</> : '▶ Run Comparison'}
            </button>
            {result && (
              <button className="btn btn-outline" onClick={handleEval} disabled={loading}>
                💾 Save Evaluation
              </button>
            )}
          </div>
        </div>
      </div>

      {!result && !loading && (
        <div className="empty-state card" style={{ padding:60 }}>
          <span className="empty-state-icon">⚖️</span>
          <span>Configure and run a comparison above to see results.</span>
        </div>
      )}

      {result && (
        <>
          {/* KPI summary row */}
          {overloadReduction !== null && (
            <div style={{
              background:'linear-gradient(135deg,rgba(16,185,129,0.1),rgba(59,130,246,0.1))',
              border:'1px solid rgba(16,185,129,0.3)', borderRadius:'var(--radius-lg)',
              padding:'16px 24px', marginBottom:20,
              display:'flex', alignItems:'center', gap:16, flexWrap:'wrap',
            }}>
              <span style={{fontSize:28}}>🎯</span>
              <div>
                <div style={{fontWeight:800, fontSize:16, color:'var(--green)'}}>
                  {overloadReduction}% reduction in overload events
                </div>
                <div style={{fontSize:13, color:'var(--text-muted)'}}>
                  Predictive scheduler ({p.overload_events} overloads) vs Reactive ({r.overload_events} overloads)
                  on <strong style={{color:'var(--text-secondary)'}}>{form.pattern}</strong> pattern
                </div>
              </div>
            </div>
          )}

          {/* Summary cards */}
          <div className="grid-2" style={{ marginBottom: 24 }}>
            {[
              { label:'Reactive Scheduler',   sum:r, color:'var(--red)',   icon:'🔴' },
              { label:'Predictive Scheduler', sum:p, color:'var(--green)', icon:'🟢' },
            ].map(({ label, sum, color, icon }) => sum && (
              <div className="card" key={label}>
                <div className="section-title" style={{color}}>
                  {icon} {label}
                </div>
                <div className="grid-2">
                  <SumCard label="Overload Events" value={sum.overload_events} color={color} icon="⚠" />
                  <SumCard label="Overload Rate"   value={`${sum.overload_rate}%`} color={color} icon="📈" />
                  <SumCard label="Avg CPU"         value={`${sum.avg_cpu}%`} color="var(--cyan)" icon="💻" />
                  <SumCard label="Total Cost"      value={sum.total_cost} color="var(--yellow)" icon="💰" />
                </div>
                <div style={{marginTop:12, display:'flex', gap:8, flexWrap:'wrap'}}>
                  <span className="badge badge-blue">↑ {sum.scale_up_count} scale-ups</span>
                  <span className="badge badge-purple">↓ {sum.scale_down_count} scale-downs</span>
                  <span className="badge badge-green">avg capacity: {sum.avg_capacity?.toFixed(1)}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Bar chart */}
          <div className="card" style={{ marginBottom: 24 }}>
            <BarCompareChart
              reactive={r}
              predictive={p}
              metrics={BAR_METRICS}
              title="Side-by-Side Metric Comparison"
            />
          </div>

          {/* Time-series charts */}
          <div className="card">
            <div className="tabs">
              <button className={`tab-btn ${activeTab==='cpu'?'active':''}`} onClick={()=>setActiveTab('cpu')}>CPU Utilization</button>
              <button className={`tab-btn ${activeTab==='cap'?'active':''}`} onClick={()=>setActiveTab('cap')}>Capacity</button>
            </div>
            {activeTab === 'cpu' && (
              <CpuComparisonChart
                reactiveRecords={result.reactive?.actions}
                predictiveRecords={result.predictive?.actions}
              />
            )}
            {activeTab === 'cap' && (
              <CapacityComparisonChart
                reactiveRecords={result.reactive?.actions}
                predictiveRecords={result.predictive?.actions}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}
