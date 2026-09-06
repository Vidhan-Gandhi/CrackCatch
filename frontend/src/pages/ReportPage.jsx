import { useRef, useState } from 'react'
import { api } from '../lib/api'
import { PriorityChip, SeverityChip } from '../components/Chips'

/**
 * Driver / commuter crowdsourcing channel.
 *
 * A named stakeholder group in the project scope, not a nice-to-have: a
 * citizen photographs a pothole with their phone, the browser Geolocation API
 * supplies the coordinates, and the photo enters exactly the same detection
 * pipeline as dashcam footage. Installable as a PWA from the browser menu.
 *
 * Data governance: the single coordinate is sent with the photo and only
 * persisted if a defect is actually confirmed in it. No background location
 * tracking, no trail.
 */
export default function ReportPage() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [position, setPosition] = useState(null)
  const [locating, setLocating] = useState(false)
  const [locationError, setLocationError] = useState(null)
  const [note, setNote] = useState('')
  const [roadType, setRoadType] = useState('arterial')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const chooseFile = (chosen) => {
    if (!chosen) return
    if (!chosen.type.startsWith('image/')) {
      setError('Please choose a photo (JPEG, PNG or WebP).')
      return
    }
    setError(null)
    setResult(null)
    setFile(chosen)
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old)
      return URL.createObjectURL(chosen)
    })
  }

  const locate = () => {
    if (!navigator.geolocation) {
      setLocationError('This browser does not expose location services.')
      return
    }
    setLocating(true)
    setLocationError(null)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setPosition({
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
          accuracy_m: pos.coords.accuracy,
        })
        setLocating(false)
      },
      (err) => {
        setLocationError(
          err.code === err.PERMISSION_DENIED
            ? 'Location permission denied. You can still submit — the photo’s EXIF coordinates will be used if present.'
            : `Could not get a location fix (${err.message}).`,
        )
        setLocating(false)
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 30000 },
    )
  }

  const submit = async (event) => {
    event.preventDefault()
    if (!file) {
      setError('Choose or take a photo first.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      setResult(
        await api.submitCrowdsource({
          file,
          latitude: position?.latitude,
          longitude: position?.longitude,
          accuracy_m: position?.accuracy_m,
          note,
          road_type: roadType,
        }),
      )
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setPreview(null)
    setResult(null)
    setNote('')
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="report-page">
      <div>
        <h2 style={{ margin: '0 0 4px' }}>Report a road defect</h2>
        <p className="muted small" style={{ margin: 0 }}>
          Photograph a pothole or crack. It is analysed by the same detector used on
          municipal survey vehicles and, if confirmed, sent straight to the road authority’s queue.
        </p>
      </div>

      {result ? (
        <div className="result-card">
          <h3 style={{ marginTop: 0 }}>
            {result.defects_found > 0 ? 'Reported — thank you' : 'Photo received'}
          </h3>
          <p className="small">{result.message}</p>

          {result.defects?.length > 0 && (
            <ul style={{ listStyle: 'none', padding: 0, margin: '10px 0 0', display: 'flex', flexDirection: 'column', gap: 10 }}>
              {result.defects.map((defect) => (
                <li key={defect._id} style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                  <SeverityChip severity={defect.severity} />
                  <PriorityChip score={defect.priority_score} />
                  <span className="faint small">
                    {defect.defect_class}
                    {defect.size?.area_m2 != null && ` · ~${defect.size.area_m2.toFixed(2)} m²`}
                  </span>
                </li>
              ))}
            </ul>
          )}

          <div className="actions" style={{ marginTop: 14 }}>
            <button className="primary" onClick={reset}>Report another</button>
          </div>
        </div>
      ) : (
        <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div
            className={`dropzone${dragging ? ' active' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); chooseFile(e.dataTransfer.files?.[0]) }}
          >
            {preview ? (
              <img className="preview" src={preview} alt="Selected road defect" />
            ) : (
              <>
                <div style={{ fontSize: 30 }}>📷</div>
                <strong>Take or choose a photo</strong>
                <div className="faint small">Get close to the defect and keep the road surface in frame</div>
              </>
            )}
          </div>
          {/* `capture` opens the rear camera directly on a phone. */}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            capture="environment"
            style={{ display: 'none' }}
            onChange={(e) => chooseFile(e.target.files?.[0])}
          />

          <div className="panel">
            <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div>
                <label>Location</label>
                {position ? (
                  <div className="small">
                    <span className="num">{position.latitude.toFixed(6)}, {position.longitude.toFixed(6)}</span>
                    <span className="faint"> (±{Math.round(position.accuracy_m)} m)</span>
                  </div>
                ) : (
                  <div className="faint small">
                    Not set — the photo’s embedded coordinates will be used if it has any.
                  </div>
                )}
                <button type="button" onClick={locate} disabled={locating} style={{ marginTop: 6 }}>
                  {locating ? <><span className="spinner" /> Locating…</> : 'Use my current location'}
                </button>
                {locationError && <div className="faint small" style={{ marginTop: 6 }}>{locationError}</div>}
              </div>

              <div>
                <label htmlFor="road">Road type</label>
                <select id="road" value={roadType} onChange={(e) => setRoadType(e.target.value)}>
                  <option value="highway">Highway</option>
                  <option value="arterial">Main road</option>
                  <option value="collector">Connecting road</option>
                  <option value="residential">Residential street</option>
                  <option value="service">Service lane</option>
                </select>
              </div>

              <div>
                <label htmlFor="note">Note (optional)</label>
                <textarea
                  id="note"
                  rows={2}
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  placeholder="e.g. deep hole near the bus stop, two-wheelers swerving"
                />
              </div>
            </div>
          </div>

          {error && <div className="banner alert"><div className="small">{error}</div></div>}

          <button className="primary" type="submit" disabled={busy || !file}>
            {busy ? <><span className="spinner" /> Analysing photo…</> : 'Submit report'}
          </button>

          <p className="faint small" style={{ margin: 0 }}>
            Only the single coordinate you attach is stored, and only if a defect is confirmed in
            the photo. CrackCatch does not track your movement.
          </p>
        </form>
      )}
    </div>
  )
}
