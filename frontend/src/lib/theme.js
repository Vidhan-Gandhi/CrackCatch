// Shared visual language. These severity colours are the same ones the Python
// annotator burns into snapshot images (model/crackcatch_model/visualize.py),
// so a map pin, a table chip and a bounding box all agree.

export const SEVERITY_COLORS = {
  Minor: '#38bdf8',     // sky
  Moderate: '#f59e0b',  // amber
  Severe: '#f43f5e',    // rose
}

export const SEVERITY_ORDER = ['Minor', 'Moderate', 'Severe']

export const STATUS_COLORS = {
  New: '#94a3b8',
  Verified: '#38bdf8',
  Scheduled: '#a78bfa',
  Repaired: '#34d399',
  Rejected: '#64748b',
}

// The repair workflow. `next` drives which action buttons appear on a defect,
// and mirrors STATUS_TRANSITIONS in the backend repository - an illegal jump
// is rejected there with a 409, so the UI never offers one.
export const STATUS_FLOW = {
  New: ['Verified', 'Rejected'],
  Verified: ['Scheduled', 'Rejected'],
  Scheduled: ['Repaired', 'Verified'],
  Repaired: ['Scheduled'],
  Rejected: ['New'],
}

export const CLASS_LABELS = {
  pothole: 'Pothole',
  crack: 'Crack',
  manhole: 'Manhole',
}

export const PRIORITY_BANDS = {
  Critical: '#f43f5e',
  High: '#f97316',
  Medium: '#f59e0b',
  Low: '#64748b',
}

export function priorityBand(score) {
  if (score >= 75) return 'Critical'
  if (score >= 55) return 'High'
  if (score >= 35) return 'Medium'
  return 'Low'
}

// Heat ramp for the road-health layer: green (healthy) -> red (failing).
export const HEAT_GRADIENT = {
  0.0: '#22c55e',
  0.35: '#facc15',
  0.65: '#f97316',
  1.0: '#dc2626',
}
