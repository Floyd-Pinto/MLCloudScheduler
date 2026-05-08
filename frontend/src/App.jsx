// src/App.jsx — Academic research dashboard (Phase 2)
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Sidebar            from './components/Sidebar';
import DashboardPage      from './pages/DashboardPage';
import SimulationPage     from './pages/SimulationPage';
import TrainingPage       from './pages/TrainingPage';
import FindingsPage       from './pages/FindingsPage';
import MetricsPage        from './pages/MetricsPage';
import LogsPage           from './pages/LogsPage';
import AnomalyLogPage     from './pages/AnomalyLogPage';
import ModelComparisonPage from './pages/ModelComparisonPage';
import './styles/global.css';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/"           element={<DashboardPage  />} />
            <Route path="/simulation" element={<SimulationPage />} />
            <Route path="/training"   element={<TrainingPage   />} />
            <Route path="/findings"   element={<FindingsPage   />} />
            <Route path="/comparison" element={<ModelComparisonPage />} />
            <Route path="/metrics"    element={<MetricsPage    />} />
            <Route path="/logs"       element={<LogsPage       />} />
            <Route path="/anomaly"    element={<AnomalyLogPage />} />
          </Routes>
        </main>
      </div>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: '#111',
            color: '#e5e5e5',
            border: '1px solid #2a2a2a',
            fontSize: '13px',
            fontFamily: 'Inter, sans-serif',
          },
        }}
      />
    </BrowserRouter>
  );
}
