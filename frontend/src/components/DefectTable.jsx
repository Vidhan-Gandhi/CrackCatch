import { format } from 'date-fns'
import { ClassChip, PriorityChip, SeverityChip, StatusChip } from './Chips'

const COLUMNS = [
  { key: 'priority_score', label: 'Priority', sortable: true },
  { key: 'defect_class', label: 'Type' },
  { key: 'severity', label: 'Severity' },
  { key: 'status', label: 'Status' },
  { key: 'size', label: 'Area' },
  { key: 'location', label: 'Location' },
  { key: 'detected_at', label: 'Detected', sortable: true },
]

function areaText(defect) {
  const size = defect.size
  if (!size || size.area_m2 == null) return <span className="faint">n/a</span>
  return (
    <span className="num" title={size.confidence_note || ''}>
      {size.area_m2.toFixed(2)} m²
      {size.reliable === false && (
        <span className="faint" title="Estimate flagged unreliable by the calibration"> *</span>
      )}
    </span>
  )
}

export default function DefectTable({
  defects,
  loading,
  onSelect,
  selectedId,
  sortBy,
  sortDir,
  onSort,
}) {
  if (loading && defects.length === 0) {
    return <div className="empty"><span className="spinner" /> Loading defects…</div>
  }
  if (defects.length === 0) {
    return <div className="empty">No defects match the current filters.</div>
  }

  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            {COLUMNS.map((column) => (
              <th
                key={column.key}
                className={column.sortable ? 'sortable' : undefined}
                onClick={column.sortable ? () => onSort(column.key) : undefined}
              >
                {column.label}
                {sortBy === column.key && (sortDir === 'desc' ? ' ▾' : ' ▴')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {defects.map((defect) => (
            <tr
              key={defect._id}
              className={defect._id === selectedId ? 'selected' : undefined}
              onClick={() => onSelect(defect)}
            >
              <td><PriorityChip score={defect.priority_score} /></td>
              <td><ClassChip defectClass={defect.defect_class} /></td>
              <td><SeverityChip severity={defect.severity} /></td>
              <td><StatusChip status={defect.status} /></td>
              <td>{areaText(defect)}</td>
              <td className="num faint">
                {defect.location?.latitude?.toFixed(4)}, {defect.location?.longitude?.toFixed(4)}
              </td>
              <td className="num faint">
                {defect.detected_at ? format(new Date(defect.detected_at), 'dd MMM HH:mm') : '-'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
