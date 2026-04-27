// src/components/Sidebar.jsx — Academic research navigation
import { NavLink } from 'react-router-dom';

const NAV = [
  { section: 'Research', items: [
    { to: '/',           icon: '◈',  label: 'Overview'          },
  ]},
  { section: 'Experiment', items: [
    { to: '/simulation', icon: '▸',  label: 'Workload Simulation'},
    { to: '/training',   icon: '▸',  label: 'Model Training'    },
  ]},
  { section: 'Results', items: [
    { to: '/findings',   icon: '▸',  label: 'Findings'          },
    { to: '/metrics',    icon: '▸',  label: 'Metrics'           },
    { to: '/logs',       icon: '▸',  label: 'Run Logs'          },
  ]},
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-badge">Research Project</div>
        <h2>ML-Based Adaptive<br/>Cloud Resource Scheduling</h2>
        <span>Major Project — 2026</span>
      </div>

      <nav className="sidebar-nav">
        {NAV.map(({ section, items }) => (
          <div key={section}>
            <div className="nav-section-label">{section}</div>
            {items.map(({ to, icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              >
                <span className="nav-icon">{icon}</span>
                {label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <div style={{ padding:'16px 20px', borderTop:'1px solid var(--border)' }}>
        <div style={{ fontSize:'11px', color:'var(--text-muted)', lineHeight:1.7 }}>
          <div style={{ display:'flex', alignItems:'center', gap:6, marginBottom:4 }}>
            <span className="pulse-dot" />
            <span style={{ color:'var(--green)', fontWeight:600, fontSize:11 }}>Backend Online</span>
          </div>
          <div style={{ fontFamily:'JetBrains Mono, monospace', fontSize:10 }}>
            LSTM · ARIMA · Combined
          </div>
        </div>
      </div>
    </aside>
  );
}
