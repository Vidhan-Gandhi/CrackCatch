// Thin API client for the CrackCatch backend.
//
// In dev, Vite proxies /api and /ws to :8000 so this stays same-origin. In
// Docker the base URL is injected as VITE_API_BASE at build time.

const BASE = import.meta.env.VITE_API_BASE ?? ''

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: options.body instanceof FormData
        ? undefined
        : { 'Content-Type': 'application/json', ...(options.headers ?? {}) },
      ...options,
    })
  } catch (cause) {
    // A network-level failure is almost always "the backend isn't running",
    // which is worth saying plainly rather than surfacing "Failed to fetch".
    throw new ApiError(
      'Cannot reach the CrackCatch API. Is the backend running on port 8000?',
      0,
      String(cause),
    )
  }

  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail ?? detail
      if (Array.isArray(detail)) {
        detail = detail.map((d) => `${d.loc?.join('.')}: ${d.msg}`).join('; ')
      }
    } catch {
      /* response body was not JSON */
    }
    throw new ApiError(detail, response.status, detail)
  }

  if (response.status === 204) return null
  const type = response.headers.get('content-type') ?? ''
  return type.includes('application/json') ? response.json() : response
}

// Build a query string, repeating keys for array values so FastAPI reads them
// as `severity=Severe&severity=Minor`.
export function toQuery(params = {}) {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    if (Array.isArray(value)) {
      value.filter((v) => v !== '' && v != null).forEach((v) => search.append(key, v))
    } else {
      search.append(key, value)
    }
  }
  const qs = search.toString()
  return qs ? `?${qs}` : ''
}

export const api = {
  health: () => request('/api/health'),

  listDefects: (params) => request(`/api/defects${toQuery(params)}`),
  getDefect: (id) => request(`/api/defects/${id}`),
  updateDefect: (id, payload) =>
    request(`/api/defects/${id}`, { method: 'PATCH', body: JSON.stringify(payload) }),
  deleteDefect: (id) => request(`/api/defects/${id}`, { method: 'DELETE' }),
  defectsNear: (lat, lon, radius_m = 150) =>
    request(`/api/defects/near${toQuery({ lat, lon, radius_m })}`),

  uploadRepairPhoto: (id, file) => {
    const form = new FormData()
    form.append('file', file)
    return request(`/api/defects/${id}/repair-photo`, { method: 'POST', body: form })
  },

  summary: (params) => request(`/api/analytics/summary${toQuery(params)}`),
  heatmap: (params) => request(`/api/analytics/heatmap${toQuery(params)}`),

  runIngest: (payload, wait = false) =>
    request(`/api/ingest/run${toQuery({ wait })}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  listJobs: () => request('/api/ingest/jobs'),
  getJob: (id) => request(`/api/ingest/jobs/${id}`),

  submitCrowdsource: ({ file, latitude, longitude, accuracy_m, note, road_type }) => {
    const form = new FormData()
    form.append('file', file)
    if (latitude != null) form.append('latitude', latitude)
    if (longitude != null) form.append('longitude', longitude)
    if (accuracy_m != null) form.append('accuracy_m', accuracy_m)
    if (note) form.append('note', note)
    if (road_type) form.append('road_type', road_type)
    return request('/api/ingest/crowdsource', { method: 'POST', body: form })
  },

  // Report downloads are plain links so the browser handles the save dialog.
  reportUrl: (format, params) => `${BASE}/api/reports/${format}${toQuery(params)}`,
  snapshotUrl: (name) => (name ? `${BASE}/media/snapshots/${name}` : null),
  repairPhotoUrl: (name) => (name ? `${BASE}/media/repairs/${name}` : null),
  explainUrl: (id) => `${BASE}/api/defects/${id}/explain`,
}

export function websocketUrl() {
  if (BASE) {
    return `${BASE.replace(/^http/, 'ws')}/ws/defects`
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws/defects`
}

export { ApiError }
