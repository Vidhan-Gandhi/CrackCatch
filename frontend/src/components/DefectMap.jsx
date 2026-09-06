import { useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import 'leaflet.heat'
import { HEAT_GRADIENT, SEVERITY_COLORS } from '../lib/theme'

/**
 * Leaflet map with two layers over the same filtered data:
 *
 *  - "Pins": one circle marker per defect, coloured by severity and sized by
 *    repair priority, so an operator can click straight through to a defect.
 *  - "Heat": the road-health layer built from the backend's severity-weighted
 *    grid aggregation. This is density of *damage*, not density of *records*:
 *    a cell holding three Severe potholes burns hotter than one holding six
 *    hairline cracks.
 *
 * Leaflet is driven imperatively rather than through react-leaflet components
 * because the heat layer is a vanilla plugin, and mixing the two declaration
 * styles for layers that must stay in sync causes flicker on every update.
 */
export default function DefectMap({
  defects = [],
  heatCells = [],
  onSelect,
  selectedId,
  center = [19.076, 72.8777],
  zoom = 14,
  tall = false,
  defaultLayer = 'pins',
}) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const markerLayerRef = useRef(null)
  const heatLayerRef = useRef(null)
  const [layer, setLayer] = useState(defaultLayer)

  // --- create the map once ---
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return

    const map = L.map(containerRef.current, {
      center,
      zoom,
      zoomControl: true,
      preferCanvas: true, // canvas rendering keeps hundreds of pins smooth
    })

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="https://carto.com/attributions">CARTO</a>',
      maxZoom: 20,
    }).addTo(map)

    markerLayerRef.current = L.layerGroup().addTo(map)

    const legend = L.control({ position: 'bottomright' })
    legend.onAdd = () => {
      const div = L.DomUtil.create('div', 'map-legend')
      div.innerHTML =
        Object.entries(SEVERITY_COLORS)
          .map(([name, colour]) => `<div class="row"><i style="background:${colour}"></i>${name}</div>`)
          .join('') + '<div class="row faint">size = repair priority</div>'
      return div
    }
    legend.addTo(map)

    mapRef.current = map

    // The container is often sized by flexbox after mount; without this the
    // map renders into a zero-height box and shows only grey tiles.
    const resize = () => map.invalidateSize()
    const observer = new ResizeObserver(resize)
    observer.observe(containerRef.current)
    setTimeout(resize, 60)

    return () => {
      observer.disconnect()
      map.remove()
      mapRef.current = null
      markerLayerRef.current = null
      heatLayerRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const points = useMemo(
    () => defects.filter((d) => d?.location?.latitude != null && d?.location?.longitude != null),
    [defects],
  )

  // --- pins ---
  useEffect(() => {
    const group = markerLayerRef.current
    if (!group) return
    group.clearLayers()
    if (layer !== 'pins') return

    points.forEach((defect) => {
      const priority = defect.priority_score ?? 0
      const marker = L.circleMarker(
        [defect.location.latitude, defect.location.longitude],
        {
          radius: 5 + (priority / 100) * 9,
          color: SEVERITY_COLORS[defect.severity] ?? '#94a3b8',
          weight: defect._id === selectedId ? 4 : 2,
          fillColor: SEVERITY_COLORS[defect.severity] ?? '#94a3b8',
          fillOpacity: defect.status === 'Repaired' ? 0.16 : 0.55,
        },
      )
      const area = defect.size?.area_m2
      const areaText =
        area == null ? 'n/a' : `${area.toFixed(2)} m²${defect.size?.reliable === false ? ' (est. unreliable)' : ''}`
      marker.bindPopup(
        `<strong>${defect.defect_class}</strong> &middot; ${defect.severity}<br/>` +
          `Priority ${Math.round(priority)}/100 &middot; ${defect.status}<br/>` +
          `<span class="faint">Area ${areaText}</span><br/>` +
          `<span class="faint">${defect.location.latitude.toFixed(5)}, ${defect.location.longitude.toFixed(5)}` +
          ` (${defect.location.source ?? 'unknown'})</span>`,
      )
      marker.on('click', () => onSelect?.(defect))
      marker.addTo(group)
    })
  }, [points, selectedId, onSelect, layer])

  // --- heat ---
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    if (heatLayerRef.current) {
      map.removeLayer(heatLayerRef.current)
      heatLayerRef.current = null
    }
    if (layer !== 'heat' || heatCells.length === 0) return

    const heatPoints = heatCells.map((cell) => [cell.latitude, cell.longitude, cell.intensity])
    heatLayerRef.current = L.heatLayer(heatPoints, {
      radius: 34,
      blur: 22,
      maxZoom: 17,
      max: 1.0,
      minOpacity: 0.32,
      gradient: HEAT_GRADIENT,
    }).addTo(map)
  }, [heatCells, layer])

  // --- keep the viewport over the data ---
  const fitKey = useMemo(
    () => points.map((d) => d._id).join(',').slice(0, 400) + `|${heatCells.length}|${layer}`,
    [points, heatCells.length, layer],
  )
  const fittedRef = useRef('')
  useEffect(() => {
    const map = mapRef.current
    if (!map || fitKey === fittedRef.current) return

    const source =
      layer === 'heat' && heatCells.length
        ? heatCells.map((c) => [c.latitude, c.longitude])
        : points.map((d) => [d.location.latitude, d.location.longitude])
    if (source.length === 0) return

    fittedRef.current = fitKey
    map.fitBounds(L.latLngBounds(source).pad(0.25), { maxZoom: 17, animate: false })
  }, [fitKey, points, heatCells, layer])

  return (
    <div style={{ position: 'relative' }}>
      <div
        className="map-toggle"
        style={{ position: 'absolute', top: 10, right: 10, zIndex: 500 }}
      >
        <button className={layer === 'pins' ? 'primary' : ''} onClick={() => setLayer('pins')}>
          Pins
        </button>
        <button className={layer === 'heat' ? 'primary' : ''} onClick={() => setLayer('heat')}>
          Heatmap
        </button>
      </div>
      <div ref={containerRef} className={`map-wrap${tall ? ' tall' : ''}`} />
      {points.length === 0 && heatCells.length === 0 && (
        <div
          className="faint small"
          style={{ position: 'absolute', bottom: 12, left: 12, zIndex: 500 }}
        >
          No geo-tagged defects match the current filters.
        </div>
      )}
    </div>
  )
}
