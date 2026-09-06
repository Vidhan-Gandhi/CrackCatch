import { useState } from 'react'
import { api } from '../lib/api'

/**
 * Demo control: run the whole pipeline over a server-side clip and watch the
 * dashboard fill up live. This is the "replay a dashcam video end-to-end
 * during the presentation" feature.
 */
export default function IngestPanel({ onStarted }) {
  const [source, setSource] = useState('data/samples/demo_drive.mp4')
  const [fps, setFps] = useState(2)
  const [roadType, setRoadType] = useState('arterial')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState(null)
  const [error, setError] = useState(null)

  const run = async () => {
    setBusy(true)
    setError(null)
    setMessage(null)
    try {
      const job = await api.runIngest({
        source,
        target_fps: Number(fps),
        road_type: roadType,
        gps_method: 'simulated',
      })
      setMessage(`Job ${job.job_id.slice(0, 8)} started — detections will stream in below.`)
      onStarted?.(job)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="panel">
      <div className="panel-head">
        Run detection pipeline
        <span className="spacer" />
        <span className="faint small">stages 1 → 6</span>
      </div>
      <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div className="filters">
          <div style={{ gridColumn: 'span 2' }}>
            <label htmlFor="src">Source (path inside the project, or a camera index)</label>
            <input id="src" value={source} onChange={(e) => setSource(e.target.value)} />
          </div>
          <div>
            <label htmlFor="fps">Sample FPS</label>
            <input id="fps" type="number" min="0.5" max="30" step="0.5" value={fps}
                   onChange={(e) => setFps(e.target.value)} />
          </div>
          <div>
            <label htmlFor="road">Road type</label>
            <select id="road" value={roadType} onChange={(e) => setRoadType(e.target.value)}>
              <option value="highway">Highway</option>
              <option value="arterial">Arterial</option>
              <option value="collector">Collector</option>
              <option value="residential">Residential</option>
              <option value="service">Service</option>
            </select>
          </div>
        </div>

        <div className="actions">
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? <><span className="spinner" /> Starting…</> : 'Run pipeline'}
          </button>
          <span className="faint small" style={{ alignSelf: 'center' }}>
            Road type feeds the traffic weight in the repair-priority score.
          </span>
        </div>

        {message && <div className="banner info"><div>{message}</div></div>}
        {error && <div className="banner alert"><div>{error}</div></div>}
      </div>
    </div>
  )
}
