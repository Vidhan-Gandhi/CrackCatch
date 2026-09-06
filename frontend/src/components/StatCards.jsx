export function StatCard({ label, value, hint, color }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value" style={color ? { color } : undefined}>{value}</div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  )
}

export function StatCards({ summary, loading }) {
  if (loading && !summary) {
    return (
      <div className="grid cards">
        {[0, 1, 2, 3, 4].map((i) => (
          <div className="stat" key={i}>
            <div className="label">&nbsp;</div>
            <div className="value faint">--</div>
          </div>
        ))}
      </div>
    )
  }
  if (!summary) return null

  const repairRate = summary.total_defects
    ? Math.round((summary.repaired_defects / summary.total_defects) * 100)
    : 0

  return (
    <div className="grid cards">
      <StatCard label="Total defects" value={summary.total_defects} hint="matching current filters" />
      <StatCard
        label="Open"
        value={summary.open_defects}
        hint="New, Verified or Scheduled"
        color="#38bdf8"
      />
      <StatCard
        label="Severe & open"
        value={summary.severe_open}
        hint="needs priority attention"
        color={summary.severe_open > 0 ? '#f43f5e' : undefined}
      />
      <StatCard label="Repaired" value={summary.repaired_defects} hint={`${repairRate}% of all defects`} color="#34d399" />
      <StatCard label="Mean priority" value={summary.mean_priority?.toFixed(1) ?? '0.0'} hint="0-100 repair score" />
    </div>
  )
}
