import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { useLiveFeed } from '../hooks/useLiveFeed'
import FilterBar, { EMPTY_FILTERS, toParams } from '../components/FilterBar'
import DefectMap from '../components/DefectMap'
import DefectTable from '../components/DefectTable'
import DefectDetail from '../components/DefectDetail'
import IngestPanel from '../components/IngestPanel'
import { StatCards } from '../components/StatCards'
import { SeverityChip } from '../components/Chips'

const PAGE_SIZE = 100

export default function Dashboard() {
  const [filters, setFilters] = useState(EMPTY_FILTERS)
  const [sortBy, setSortBy] = useState('priority_score')
  const [sortDir, setSortDir] = useState('desc')
  const [selected, setSelected] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [liveCount, setLiveCount] = useState(0)

  const params = useMemo(
    () => ({ ...toParams(filters), page_size: PAGE_SIZE, sort_by: sortBy, sort_dir: sortDir }),
    [filters, sortBy, sortDir],
  )

  const defectsQuery = useAsync(() => api.listDefects(params), [JSON.stringify(params)])
  const summaryQuery = useAsync(() => api.summary(toParams(filters)), [JSON.stringify(toParams(filters))])
  const heatQuery = useAsync(
    () => api.heatmap({ ...toParams(filters), cell_size_deg: 0.002 }),
    [JSON.stringify(toParams(filters))],
  )

  const reloadAll = useCallback(() => {
    defectsQuery.reload()
    summaryQuery.reload()
    heatQuery.reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defectsQuery.reload, summaryQuery.reload, heatQuery.reload])

  // Live feed: new detections are counted immediately, and the full refresh is
  // debounced so a burst from an ingestion run does not trigger one request
  // per detection.
  useLiveFeed({
    'defect.created': () => setLiveCount((n) => n + 1),
    'alert.high_priority': (defect) =>
      setAlerts((current) => [defect, ...current.filter((a) => a._id !== defect._id)].slice(0, 5)),
    'job.completed': reloadAll,
  })

  useEffect(() => {
    if (liveCount === 0) return
    const timer = setTimeout(() => {
      reloadAll()
      setLiveCount(0)
    }, 1200)
    return () => clearTimeout(timer)
  }, [liveCount, reloadAll])

  const defects = defectsQuery.data?.items ?? []
  const total = defectsQuery.data?.total ?? 0

  const onSort = (key) => {
    if (key === sortBy) setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'))
    else {
      setSortBy(key)
      setSortDir('desc')
    }
  }

  const onUpdated = (updated) => {
    setSelected(updated)
    defectsQuery.setData((current) =>
      current
        ? { ...current, items: current.items.map((d) => (d._id === updated._id ? updated : d)) }
        : current,
    )
    summaryQuery.reload()
  }

  const reportParams = toParams(filters)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {alerts.length > 0 && (
        <div className="banner alert">
          <div style={{ flex: 1 }}>
            <strong>High-priority defects detected</strong>
            <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
              {alerts.map((alert) => (
                <li key={alert._id} className="small">
                  <SeverityChip severity={alert.severity} /> {alert.defect_class} · priority{' '}
                  {Math.round(alert.priority_score)} · {alert.location?.latitude?.toFixed(4)},{' '}
                  {alert.location?.longitude?.toFixed(4)}{' '}
                  <button className="ghost small" onClick={() => setSelected(alert)}>Open</button>
                </li>
              ))}
            </ul>
          </div>
          <button className="close" onClick={() => setAlerts([])} aria-label="Dismiss">✕</button>
        </div>
      )}

      {defectsQuery.error && (
        <div className="banner alert">
          <div>
            <strong>Could not load defects.</strong>{' '}
            <span className="small">{defectsQuery.error.message}</span>
          </div>
          <button className="close" onClick={reloadAll}>Retry</button>
        </div>
      )}

      <StatCards summary={summaryQuery.data} loading={summaryQuery.loading} />

      <FilterBar
        filters={filters}
        onChange={setFilters}
        onReset={() => setFilters(EMPTY_FILTERS)}
        total={total}
      />

      <div className="grid two">
        <div className="panel">
          <div className="panel-head">
            Defect map
            <span className="spacer" />
            {liveCount > 0 && <span className="small" style={{ color: 'var(--ok)' }}>+{liveCount} live</span>}
          </div>
          <DefectMap
            defects={defects}
            heatCells={heatQuery.data?.cells ?? []}
            onSelect={setSelected}
            selectedId={selected?._id}
          />
        </div>

        <IngestPanel onStarted={() => setLiveCount((n) => n + 1)} />
      </div>

      <div className="panel">
        <div className="panel-head">
          Defects
          <span className="spacer" />
          <span className="faint small">
            showing {defects.length} of {total}
          </span>
          <a className="ghost" href={api.reportUrl('csv', reportParams)}>
            <button className="ghost">Export CSV</button>
          </a>
          <a href={api.reportUrl('pdf', reportParams)}>
            <button className="ghost">Export PDF</button>
          </a>
        </div>
        <DefectTable
          defects={defects}
          loading={defectsQuery.loading}
          onSelect={setSelected}
          selectedId={selected?._id}
          sortBy={sortBy}
          sortDir={sortDir}
          onSort={onSort}
        />
      </div>

      <DefectDetail defect={selected} onClose={() => setSelected(null)} onUpdated={onUpdated} />
    </div>
  )
}
