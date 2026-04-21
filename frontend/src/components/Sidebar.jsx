// src/components/Sidebar.jsx
import { NavLink } from 'react-router-dom';

const NAV = [
  { section: 'Overview', items: [
    { to: '/',           icon: '⬡',  label: 'Dashboard'         },
  ]},
  { section: 'ML Models', items: [
    { to: '/training',   icon: '🧠', label: 'Model Training'    },
    { to: '/models',     icon: '🔬', label: 'Model Comparison'  },
  ]},
  { section: 'Scheduler', items: [
    { to: '/simulation', icon: '⚡', label: 'Simulation'         },
    { to: '/comparison', icon: '⚖️', label: 'Scheduler Comparison'},
  ]},
  { section: 'Analytics', items: [
    { to: '/metrics',    icon: '📊', label: 'Metrics'            },
    { to: '/logs',       icon: '📋', label: 'Run Logs'           },
  ]},
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-badge">🚀 ML Scheduler</div>
        <h2>Adaptive Cloud<br/>Resource Scheduler</h2>
        <span>Final Year Major Project</span>
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

      <div style={{ padding:'16px 20px', borderTop:'1px solid var(--border-subtle)' }}>
        <div style={{ fontSize:'11px', color:'var(--text-muted)', lineHeight:1.7 }}>
          <div style={{ display:'flex', alignItems:'center', gap:6, marginBottom:4 }}>
            <span className="pulse-dot" />
            <span style={{ color:'var(--green)', fontWeight:600 }}>Backend Online</span>
          </div>
          <div>GBR · LSTM · ARIMA · Combined</div>
        </div>
      </div>
    </aside>
  );
}
