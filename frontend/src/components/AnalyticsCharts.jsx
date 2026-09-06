import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { SEVERITY_COLORS, STATUS_COLORS } from '../lib/theme'

const AXIS = { stroke: '#64748b', fontSize: 11 }
const TOOLTIP_STYLE = {
  contentStyle: {
    background: '#111c33',
    border: '1px solid #23324f',
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: '#93a4bf' },
}

function Panel({ title, subtitle, children, height = 240 }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          {title}
          {subtitle && <div className="faint small" style={{ fontWeight: 400 }}>{subtitle}</div>}
        </div>
      </div>
      <div className="panel-body" style={{ height }}>
        {children}
      </div>
    </div>
  )
}

export function SeverityPie({ data }) {
  const rows = (data ?? []).filter((d) => d.count > 0)
  return (
    <Panel title="Severity distribution" subtitle="all defects matching filters">
      {rows.length === 0 ? (
        <div className="empty">No data</div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={rows} dataKey="count" nameKey="severity" innerRadius="48%" outerRadius="78%" paddingAngle={2}>
              {rows.map((row) => (
                <Cell key={row.severity} fill={SEVERITY_COLORS[row.severity]} stroke="#0b1120" />
              ))}
            </Pie>
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
          </PieChart>
        </ResponsiveContainer>
      )}
    </Panel>
  )
}

export function StatusBars({ data }) {
  const rows = data ?? []
  return (
    <Panel title="Repair workflow" subtitle="New → Verified → Scheduled → Repaired">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 6, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1c2b49" vertical={false} />
          <XAxis dataKey="status" tick={AXIS} axisLine={false} tickLine={false} />
          <YAxis tick={AXIS} axisLine={false} tickLine={false} allowDecimals={false} />
          <Tooltip {...TOOLTIP_STYLE} cursor={{ fill: 'rgba(56,189,248,0.08)' }} />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {rows.map((row) => (
              <Cell key={row.status} fill={STATUS_COLORS[row.status] ?? '#94a3b8'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  )
}

export function DefectsOverTime({ data }) {
  const rows = data ?? []
  return (
    <Panel title="Defects detected over time" subtitle="daily counts, severe highlighted">
      {rows.length === 0 ? (
        <div className="empty">No detections in this window</div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 6, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1c2b49" vertical={false} />
            <XAxis dataKey="date" tick={AXIS} axisLine={false} tickLine={false} minTickGap={24} />
            <YAxis tick={AXIS} axisLine={false} tickLine={false} allowDecimals={false} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="count" name="All" stroke="#38bdf8" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="severe" name="Severe" stroke="#f43f5e" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </Panel>
  )
}

export function ClassBars({ data }) {
  const rows = data ?? []
  return (
    <Panel title="Defects by type">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 6, right: 16, left: 10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1c2b49" horizontal={false} />
          <XAxis type="number" tick={AXIS} axisLine={false} tickLine={false} allowDecimals={false} />
          <YAxis type="category" dataKey="defect_class" tick={AXIS} axisLine={false} tickLine={false} width={70} />
          <Tooltip {...TOOLTIP_STYLE} cursor={{ fill: 'rgba(56,189,248,0.08)' }} />
          <Bar dataKey="count" fill="#38bdf8" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  )
}

export function HotspotTable({ cells, onSelect }) {
  const rows = (cells ?? []).slice(0, 12)
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          Hotspots
          <div className="faint small" style={{ fontWeight: 400 }}>
            ~200 m grid cells, ranked by severity-weighted damage density
          </div>
        </div>
      </div>
      {rows.length === 0 ? (
        <div className="empty">No hotspots yet</div>
      ) : (
        <div className="table-scroll" style={{ maxHeight: 320 }}>
          <table className="data">
            <thead>
              <tr><th>#</th><th>Location</th><th>Defects</th><th>Severe</th><th>Mean priority</th><th>Intensity</th></tr>
            </thead>
            <tbody>
              {rows.map((cell, index) => (
                <tr key={`${cell.latitude},${cell.longitude}`} onClick={() => onSelect?.(cell)}>
                  <td className="num faint">{index + 1}</td>
                  <td className="num">{cell.latitude.toFixed(4)}, {cell.longitude.toFixed(4)}</td>
                  <td className="num">{cell.count}</td>
                  <td className="num" style={{ color: cell.severe_count ? '#f43f5e' : undefined }}>
                    {cell.severe_count}
                  </td>
                  <td className="num">{cell.mean_priority.toFixed(1)}</td>
                  <td>
                    <span className="bar" style={{ display: 'block', width: 70 }}>
                      <span style={{ width: `${cell.intensity * 100}%`, background: '#f97316' }} />
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
