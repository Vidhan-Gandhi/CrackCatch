import { useEffect, useRef, useState } from 'react'
import { format } from 'date-fns'
import { api } from '../lib/api'
import { STATUS_FLOW } from '../lib/theme'
import { ClassChip, GpsSourceChip, PriorityChip, SeverityChip, StatusChip } from './Chips'

const FACTOR_LABELS = {
  size_factor: 'Size',
  shape_factor: 'Shape (AR)',
  confidence_factor: 'Confidence',
}

/**
 * Renders the severity score the way the scoring module computes it, so the
 * number on screen can be defended: each factor, its weight contribution, and
 * the total. This is the "not a black box" requirement made visible.
 */
function SeverityBreakdown({ breakdown, score }) {
  if (!breakdown || Object.keys(breakdown).length === 0) {
    return <p className="faint small">No breakdown was stored for this detection.</p>
  }
  const basis = breakdown._size_basis
  return (
    <div className="breakdown">
      {Object.entries(FACTOR_LABELS).map(([key, label]) => {
        const value = breakdown[key] ?? 0
        const contribution = breakdown[key.replace('_factor', '_contribution')] ?? 0
        return (
          <div className="row" key={key}>
            <span className="faint">{label}</span>
            <span className="bar"><span style={{ width: `${Math.min(100, value * 100)}%` }} /></span>
            <span className="num">{value.toFixed(2)} → {contribution.toFixed(3)}</span>
          </div>
        )
      })}
      <div className="row" style={{ borderTop: '1px solid var(--border)', paddingTop: 7 }}>
        <strong>Total</strong>
        <span />
        <strong className="num">{(score ?? breakdown.total ?? 0).toFixed(3)}</strong>
      </div>
      <p className="faint small" style={{ margin: 0 }}>
        Size measured from <span className="mono">{basis ?? 'unknown'}</span>
        {breakdown.aspect_ratio != null && <> · aspect ratio {breakdown.aspect_ratio.toFixed(2)}</>}
        . Thresholds: &lt;0.35 Minor, &lt;0.62 Moderate, else Severe.
      </p>
    </div>
  )
}

export default function DefectDetail({ defect, onClose, onUpdated }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [note, setNote] = useState('')
  const [showExplain, setShowExplain] = useState(false)
  const [explainMethod, setExplainMethod] = useState(null)
  const fileRef = useRef(null)

  useEffect(() => {
    setNote('')
    setError(null)
    setShowExplain(false)
    setExplainMethod(null)
  }, [defect?._id])

  // Fetch the overlay through the API so the X-Explain-Method header can be
  // read; an <img src> would hide whether it is real Grad-CAM or the fallback.
  useEffect(() => {
    if (!showExplain || !defect) return
    let revoked = null
    let cancelled = false
    ;(async () => {
      try {
        const response = await fetch(api.explainUrl(defect._id))
        if (!response.ok) throw new Error(`${response.status}`)
        setExplainMethod(response.headers.get('X-Explain-Method') ?? 'unknown')
        const blob = await response.blob()
        if (cancelled) return
        revoked = URL.createObjectURL(blob)
        setExplainMethod((m) => m)
        document.getElementById('explain-img')?.setAttribute('src', revoked)
      } catch {
        if (!cancelled) setExplainMethod('unavailable')
      }
    })()
    return () => {
      cancelled = true
      if (revoked) URL.revokeObjectURL(revoked)
    }
  }, [showExplain, defect])

  if (!defect) return null

  const act = async (payload) => {
    setBusy(true)
    setError(null)
    try {
      const updated = await api.updateDefect(defect._id, { ...payload, note: note || payload.note || '' })
      onUpdated(updated)
      setNote('')
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const uploadRepairPhoto = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      onUpdated(await api.uploadRepairPhoto(defect._id, file))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const size = defect.size ?? {}
  const nextStatuses = STATUS_FLOW[defect.status] ?? []
  const snapshot = api.snapshotUrl(defect.snapshot_path)
  const repairPhoto = api.repairPhotoUrl(defect.repair_photo_path)

  return (
    <div className="drawer-backdrop" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <aside className="drawer">
        <div className="drawer-head">
          <ClassChip defectClass={defect.defect_class} />
          <SeverityChip severity={defect.severity} />
          <StatusChip status={defect.status} />
          <span className="spacer" />
          <button className="ghost" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="drawer-body">
          {error && <div className="banner alert"><div>{error}</div></div>}

          {snapshot ? (
            <div>
              <img
                id={showExplain ? 'explain-img' : undefined}
                className="snapshot"
                src={showExplain ? undefined : snapshot}
                alt={`${defect.defect_class} detection snapshot`}
                loading="lazy"
              />
              <div className="actions" style={{ marginTop: 8 }}>
                <button onClick={() => setShowExplain((v) => !v)}>
                  {showExplain ? 'Show snapshot' : 'Explain detection'}
                </button>
                {showExplain && explainMethod && (
                  <span className="faint small" style={{ alignSelf: 'center' }}>
                    method: {explainMethod}
                  </span>
                )}
              </div>
            </div>
          ) : (
            <p className="faint small">No snapshot stored for this detection.</p>
          )}

          {repairPhoto && (
            <div>
              <div className="panel-head" style={{ padding: '0 0 6px', border: 0 }}>After repair</div>
              <img className="snapshot" src={repairPhoto} alt="Post-repair verification" loading="lazy" />
            </div>
          )}

          <dl className="kv">
            <dt>Repair priority</dt>
            <dd><PriorityChip score={defect.priority_score} /></dd>
            <dt>Confidence</dt>
            <dd className="num">{((defect.confidence ?? 0) * 100).toFixed(1)}%</dd>
            <dt>Detected</dt>
            <dd className="num">
              {defect.detected_at ? format(new Date(defect.detected_at), 'dd MMM yyyy HH:mm:ss') : '-'}
            </dd>
            <dt>Coordinates</dt>
            <dd className="num">
              {defect.location?.latitude?.toFixed(6)}, {defect.location?.longitude?.toFixed(6)}{' '}
              <GpsSourceChip source={defect.location?.source} />
            </dd>
            <dt>Estimated size</dt>
            <dd className="num">
              {size.area_m2 != null
                ? `${size.width_m?.toFixed(2)} × ${size.length_m?.toFixed(2)} m  (${size.area_m2.toFixed(2)} m²)`
                : 'not recoverable'}
            </dd>
            <dt>Range / method</dt>
            <dd className="num faint">
              {size.range_m != null ? `${size.range_m.toFixed(1)} m · ` : ''}{size.method ?? '-'}
              {size.reliable === false && <span style={{ color: 'var(--warn)' }}> · unreliable</span>}
            </dd>
            <dt>Road type</dt>
            <dd>{defect.road_type}</dd>
            <dt>Source</dt>
            <dd className="faint small">{defect.source_type} · {defect.source_ref || '-'}</dd>
          </dl>

          {size.confidence_note && (
            <p className="faint small" style={{ margin: 0 }}>{size.confidence_note}</p>
          )}

          <div>
            <div className="panel-head" style={{ padding: '0 0 8px', border: 0 }}>
              Why this severity?
            </div>
            <SeverityBreakdown breakdown={defect.severity_breakdown} score={defect.severity_score} />
          </div>

          <div>
            <label htmlFor="note">Reviewer note (stored for retraining)</label>
            <textarea
              id="note"
              rows={2}
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="e.g. confirmed on site; actually a tar patch"
            />
          </div>

          <div className="actions">
            {nextStatuses.map((status) => (
              <button
                key={status}
                className={status === 'Rejected' ? 'danger' : 'primary'}
                disabled={busy}
                onClick={() => act({ status })}
              >
                Mark {status}
              </button>
            ))}
            <button disabled={busy} onClick={() => act({ review_label: defect.defect_class === 'pothole' ? 'crack' : 'pothole' })}>
              Correct class → {defect.defect_class === 'pothole' ? 'crack' : 'pothole'}
            </button>
          </div>

          {defect.status === 'Repaired' && (
            <div>
              <label htmlFor="after">Attach an after-repair photo</label>
              <input id="after" ref={fileRef} type="file" accept="image/*" onChange={uploadRepairPhoto} disabled={busy} />
            </div>
          )}

          {defect.history?.length > 0 && (
            <div>
              <div className="panel-head" style={{ padding: '0 0 8px', border: 0 }}>Audit trail</div>
              <ul className="small faint" style={{ margin: 0, paddingLeft: 18 }}>
                {defect.history.map((event, index) => (
                  <li key={index}>
                    <strong>{event.status}</strong> by {event.by} ·{' '}
                    {event.at ? format(new Date(event.at), 'dd MMM HH:mm') : ''}
                    {event.note ? ` — ${event.note}` : ''}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}
