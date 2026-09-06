import { CLASS_LABELS, PRIORITY_BANDS, SEVERITY_COLORS, STATUS_COLORS, priorityBand } from '../lib/theme'

export function SeverityChip({ severity }) {
  const color = SEVERITY_COLORS[severity] ?? '#94a3b8'
  return (
    <span className="chip" style={{ color }}>
      <span className="dot" />
      {severity}
    </span>
  )
}

export function StatusChip({ status }) {
  const color = STATUS_COLORS[status] ?? '#94a3b8'
  return <span className="chip" style={{ color }}>{status}</span>
}

export function PriorityChip({ score }) {
  const band = priorityBand(score ?? 0)
  return (
    <span className="chip" style={{ color: PRIORITY_BANDS[band] }}>
      {band} {Math.round(score ?? 0)}
    </span>
  )
}

export function ClassChip({ defectClass }) {
  return <span className="chip faint">{CLASS_LABELS[defectClass] ?? defectClass}</span>
}

/**
 * Flags a coordinate that did not come from a real GPS fix. Demo footage uses
 * a simulated route, and the dashboard says so rather than letting a viewer
 * assume every pin is a surveyed location.
 */
export function GpsSourceChip({ source }) {
  if (!source || source === 'device' || source === 'gpx' || source === 'exif') {
    return <span className="chip faint">{source ?? 'unknown'}</span>
  }
  return (
    <span className="chip" style={{ color: '#fbbf24' }} title="Not a real GPS fix">
      {source}
    </span>
  )
}
