import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Analytics from './pages/Analytics'
import ReportPage from './pages/ReportPage'
import { api } from './lib/api'
import { useLiveFeed } from './hooks/useLiveFeed'

function SystemBanner({ health }) {
  const [dismissed, setDismissed] = useState(false)
  if (!health || dismissed || !health.warnings?.length) return null
  return (
    <div className="banner warn">
      <div>
        <strong>System notice</strong>
        <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
          {health.warnings.map((warning, index) => <li key={index} className="small">{warning}</li>)}
        </ul>
      </div>
      <button className="close" onClick={() => setDismissed(true)} aria-label="Dismiss">✕</button>
    </div>
  )
}

export default function App() {
  const [health, setHealth] = useState(null)
  const { connected } = useLiveFeed({})

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/icon.svg" alt="" width="30" height="30" />
          <div>
            CrackCatch
            <small>Road damage detection &amp; repair workflow</small>
          </div>
        </div>

        <nav className="nav">
          <NavLink to="/dashboard" className={({ isActive }) => (isActive ? 'active' : '')}>Dashboard</NavLink>
          <NavLink to="/analytics" className={({ isActive }) => (isActive ? 'active' : '')}>Analytics</NavLink>
          <NavLink to="/report" className={({ isActive }) => (isActive ? 'active' : '')}>Report a defect</NavLink>
        </nav>

        <div className="topbar-right">
          {health && (
            <span className="faint small" title={`database: ${health.database_backend}`}>
              {health.detector_is_trained_model ? 'YOLOv8' : 'CV baseline'} · {health.database_backend}
            </span>
          )}
          <span className={`live-dot${connected ? ' on' : ''}`}>
            <i />{connected ? 'Live' : 'Offline'}
          </span>
        </div>
      </header>

      <main className="content">
        <SystemBanner health={health} />
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/report" element={<ReportPage />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </main>
    </div>
  )
}
