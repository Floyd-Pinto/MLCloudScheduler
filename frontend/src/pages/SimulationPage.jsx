// src/pages/SimulationPage.jsx
import { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { simulationAPI } from '../services/api';
import WorkloadChart from '../charts/WorkloadChart';

const PATTERNS = ['gradual', 'spike', 'periodic', 'combined'];

const PATTERN_DESCRIPTIONS = {
  gradual:  'Steadily increasing load — simulates user growth over time.',
  spike:    'Sudden burst — flash sale, viral event, or DDoS scenario.',
  periodic: 'Sinusoidal daily cycle — regular business-hours traffic.',
  combined: 'Mix of all patterns — most realistic production simulation.',
};

export default function SimulationPage() {
  const [form,    setForm]    = useState({ pattern: 'combined', steps: 200, seed: 42, label: '' });
  const [runs,    setRuns]    = useState([]);
  const [loading, setLoading] = useState(false);
  const [selected,setSelected]= useState(null);

  useEffect(() => { fetchRuns(); }, []);

  const fetchRuns = () =>
    simulationAPI.listRuns().then(r => setRuns(r.data)).catch(console.error);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const res = await simulationAPI.generate(form);
      toast.success(`Workload generated — ${res.data.steps} steps (${res.data.pattern})`);
      setSelected(res.data);
      fetchRuns();
    } catch (e) {
      toast.error(e.response?.data?.error || 'Generation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectRun = async (id) => {
    try {
      const res = await simulationAPI.getRun(id);
      setSelected(res.data);
    } catch { toast.error('Failed to load run'); }
  };

  const handleDelete = async (id, e) => {
    e.stopPropagation();
    try {
      await simulationAPI.deleteRun(id);
      toast.success('Run deleted');
      if (selected?.id === id) setSelected(null);
      fetchRuns();
    } catch { toast.error('Delete failed'); }
  };

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Workload Simulation</div>
        <div className="page-subtitle">Generate synthetic cloud workload patterns for scheduler testing.</div>
      </div>

      <div className="grid-2" style={{ marginBottom: 24 }}>
        {/* Control Panel */}
        <div className="card">
          <div className="section-title">Generate Workload</div>

          <div className="form-group">
            <label className="form-label">Pattern Type</label>
            <select className="form-control"
              value={form.pattern}
              onChange={e => setForm(f => ({...f, pattern: e.target.value}))}>
              {PATTERNS.map(p => <option key={p} value={p}>{p.charAt(0).toUpperCase()+p.slice(1)}</option>)}
            </select>
            <div style={{ marginTop: 8, padding: '8px 12px', background:'var(--bg-input)',
                          borderRadius: 'var(--radius-sm)', fontSize: 12, color:'var(--text-muted)' }}>
              {PATTERN_DESCRIPTIONS[form.pattern]}
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Steps (Time Points)</label>
            <input type="number" className="form-control"
              min={50} max={1000} value={form.steps}
              onChange={e => setForm(f => ({...f, steps: +e.target.value}))} />
          </div>

          <div className="form-group">
            <label className="form-label">Random Seed</label>
            <input type="number" className="form-control"
              min={0} max={99999} value={form.seed}
              onChange={e => setForm(f => ({...f, seed: +e.target.value}))} />
          </div>

          <div className="form-group">
            <label className="form-label">Label (optional)</label>
            <input type="text" className="form-control"
              placeholder="e.g. Flash sale test"
              value={form.label}
              onChange={e => setForm(f => ({...f, label: e.target.value}))} />
          </div>

          <button className="btn btn-primary btn-lg" style={{width:'100%'}}
            onClick={handleGenerate} disabled={loading}>
            {loading ? <><span className="loading-spinner"/>&nbsp;Generating…</> : 'Generate Workload'}
          </button>
        </div>

        {/* Pattern preview */}
        <div className="card">
          <div className="section-title">Pattern Preview — {form.pattern}</div>
          {selected?.datapoints ? (
            <WorkloadChart records={selected.datapoints} title={`${selected.pattern} | ${selected.steps} steps`} />
          ) : (
            <div className="empty-state">
              <span className="empty-state-icon">—</span>
              <span>Generate a workload to preview the chart</span>
            </div>
          )}
          {selected && (
            <div style={{ marginTop: 16, display:'flex', gap:8, flexWrap:'wrap' }}>
              {[
                ['Pattern',  selected.pattern],
                ['Steps',    selected.steps],
                ['Seed',     selected.seed],
                ['ID',       `#${selected.id}`],
              ].map(([k,v]) => (
                <span key={k} className="badge badge-blue">{k}: {v}</span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Run history */}
      <div className="card">
        <div className="section-title" style={{ display:'flex', justifyContent:'space-between' }}>
          <span>Workload Run History</span>
          <span className="badge badge-blue">{runs.length} runs</span>
        </div>
        {runs.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-icon">—</span>
            <span>No runs yet — generate a workload above.</span>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th><th>Pattern</th><th>Steps</th><th>Seed</th><th>Label</th><th>Created</th><th></th>
                </tr>
              </thead>
              <tbody>
                {runs.map(r => (
                  <tr key={r.id} onClick={() => handleSelectRun(r.id)}
                    style={{ cursor:'pointer', background: selected?.id===r.id ? 'var(--bg-card-hover)' : '' }}>
                    <td><span className="badge badge-blue">#{r.id}</span></td>
                    <td><span className={`badge badge-${
                      r.pattern==='spike'?'red':r.pattern==='gradual'?'green':r.pattern==='periodic'?'yellow':'purple'}`}>
                      {r.pattern}</span></td>
                    <td>{r.steps}</td>
                    <td>{r.seed}</td>
                    <td style={{ color:'var(--text-muted)', fontStyle: r.label?'':'italic' }}>
                      {r.label || '—'}
                    </td>
                    <td>{new Date(r.created_at).toLocaleString()}</td>
                    <td>
                      <button className="btn btn-danger btn-sm"
                        onClick={e => handleDelete(r.id, e)}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
