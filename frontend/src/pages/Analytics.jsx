import { useMemo, useState } from 'react'
import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import FilterBar, { EMPTY_FILTERS, toParams } from '../components/FilterBar'
import DefectMap from '../components/DefectMap'
import { StatCards } from '../components/StatCards'
import {
  ClassBars,
  DefectsOverTime,
  HotspotTable,
  SeverityPie,
  StatusBars,
} from '../components/AnalyticsCharts'
import { PriorityChip, SeverityChip, StatusChip } from '../components/Chips'

/**
 * The planning view. Municipal corporations use the Dashboard to work a queue;
 * traffic departments and smart-city planners use this page to see where road
 * health is failing over time. Same data, same filters, different question.
 */
export default function Analytics() {
  const [filters, setFilters] = useState(EMPTY_FILTERS)
  const [cellSize, setCellSize] = useState(0.002)
  const params = useMemo(() => toParams(filters), [filters])

  const summaryQuery = useAsync(() => api.summary({ ...params, days: 60 }), [JSON.stringify(params)])
  const heatQuery = useAsync(
    () => api.heatmap({ ...params, cell_size_deg: cellSize }),
    [JSON.stringify(params), cellSize],
  )

  const summary = summaryQuery.data
  const cells = heatQuery.data?.cells ?? []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <StatCards summary={summary} loading={summaryQuery.loading} />

      <FilterBar filters={filters} onChange={setFilters} onReset={() => setFilters(EMPTY_FILTERS)} />

      <div className="panel">
        <div className="panel-head">
          Road-health heatmap
          <span className="spacer" />
          <label htmlFor="cell" style={{ margin: 0 }}>Grid</label>
          <select
            id="cell"
            value={cellSize}
            onChange={(e) => setCellSize(Number(e.target.value))}
            style={{ width: 150 }}
          >
            <option value={0.001}>~100 m cells</option>
            <option value={0.002}>~200 m cells</option>
            <option value={0.005}>~550 m cells</option>
            <option value={0.01}>~1.1 km cells</option>
          </select>
        </div>
        <DefectMap heatCells={cells} defects={[]} tall defaultLayer="heat" />
      </div>

      <div className="grid charts">
        <DefectsOverTime data={summary?.over_time} />
        <SeverityPie data={summary?.by_severity} />
        <StatusBars data={summary?.by_status} />
        <ClassBars data={summary?.by_class} />
      </div>

      <div className="grid two">
        <HotspotTable cells={cells} />

        <div className="panel">
          <div className="panel-head">
            Repair priority queue
            <span className="spacer" />
            <span className="faint small">highest-scoring open defects</span>
          </div>
          {summary?.top_priority?.length ? (
            <div className="table-scroll" style={{ maxHeight: 320 }}>
              <table className="data">
                <thead>
                  <tr><th>Priority</th><th>Type</th><th>Severity</th><th>Status</th><th>Location</th></tr>
                </thead>
                <tbody>
                  {summary.top_priority.map((defect) => (
                    <tr key={defect._id}>
                      <td><PriorityChip score={defect.priority_score} /></td>
                      <td className="faint">{defect.defect_class}</td>
                      <td><SeverityChip severity={defect.severity} /></td>
                      <td><StatusChip status={defect.status} /></td>
                      <td className="num faint">
                        {defect.location?.latitude?.toFixed(4)}, {defect.location?.longitude?.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty">No open defects</div>
          )}
        </div>
      </div>

      <p className="faint small" style={{ margin: 0 }}>
        Heatmap intensity is a severity-weighted damage density (Minor 1, Moderate 2, Severe 4),
        normalised against the busiest cell in the current filter — not a raw record count.
        Coordinates from demo footage are simulated and are labelled as such on each defect.
      </p>
    </div>
  )
}
