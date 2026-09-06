import { CLASS_LABELS, SEVERITY_ORDER } from '../lib/theme'

const STATUSES = ['New', 'Verified', 'Scheduled', 'Repaired', 'Rejected']
const CLASSES = ['pothole', 'crack', 'manhole']

function Multi({ label, options, selected, onChange, renderLabel }) {
  const toggle = (value) =>
    onChange(selected.includes(value) ? selected.filter((v) => v !== value) : [...selected, value])
  return (
    <div>
      <label>{label}</label>
      <div className="multi">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            className={selected.includes(option) ? 'on' : ''}
            onClick={() => toggle(option)}
          >
            {renderLabel ? renderLabel(option) : option}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function FilterBar({ filters, onChange, onReset, total }) {
  const set = (patch) => onChange({ ...filters, ...patch })

  return (
    <div className="panel">
      <div className="panel-head">
        Filters
        <span className="spacer" />
        {total != null && <span className="faint small">{total} matching</span>}
        <button className="ghost" onClick={onReset}>Reset</button>
      </div>
      <div className="panel-body">
        <div className="filters">
          <Multi
            label="Severity"
            options={SEVERITY_ORDER}
            selected={filters.severity}
            onChange={(severity) => set({ severity })}
          />
          <Multi
            label="Status"
            options={STATUSES}
            selected={filters.status}
            onChange={(status) => set({ status })}
          />
          <Multi
            label="Type"
            options={CLASSES}
            selected={filters.defect_class}
            onChange={(defect_class) => set({ defect_class })}
            renderLabel={(c) => CLASS_LABELS[c] ?? c}
          />
          <div>
            <label htmlFor="from">From</label>
            <input
              id="from"
              type="date"
              value={filters.date_from}
              onChange={(e) => set({ date_from: e.target.value })}
            />
          </div>
          <div>
            <label htmlFor="to">To</label>
            <input
              id="to"
              type="date"
              value={filters.date_to}
              onChange={(e) => set({ date_to: e.target.value })}
            />
          </div>
          <div>
            <label htmlFor="prio">Min priority: {filters.min_priority || 0}</label>
            <input
              id="prio"
              type="range"
              min="0"
              max="100"
              step="5"
              value={filters.min_priority || 0}
              onChange={(e) => set({ min_priority: Number(e.target.value) })}
            />
          </div>
          <div>
            <label htmlFor="search">Search</label>
            <input
              id="search"
              type="search"
              placeholder="source, road type, notes"
              value={filters.search}
              onChange={(e) => set({ search: e.target.value })}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export const EMPTY_FILTERS = {
  severity: [],
  status: [],
  defect_class: [],
  date_from: '',
  date_to: '',
  min_priority: 0,
  search: '',
}

/** Translate the UI filter state into backend query parameters. */
export function toParams(filters) {
  return {
    severity: filters.severity,
    status: filters.status,
    defect_class: filters.defect_class,
    // The backend compares against timestamps, so widen a plain date to the
    // whole day; otherwise "to = today" would exclude everything after 00:00.
    date_from: filters.date_from ? `${filters.date_from}T00:00:00Z` : '',
    date_to: filters.date_to ? `${filters.date_to}T23:59:59Z` : '',
    min_priority: filters.min_priority || '',
    search: filters.search?.trim() || '',
  }
}
