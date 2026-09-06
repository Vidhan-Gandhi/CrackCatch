#!/usr/bin/env node
/**
 * Headless UI smoke test for the CrackCatch dashboard.
 *
 * Drives a local Chromium-family browser (Brave, Chrome or Chromium) over the
 * DevTools Protocol, loads each route against a running stack, and asserts the
 * things a build cannot catch: that the map actually has a non-zero size, that
 * tiles load, that the table renders rows, and that no uncaught exception or
 * blocked request appears in the console.
 *
 * This exists because a CSS regression once collapsed the Leaflet map to zero
 * height. Every unit test passed, the bundle built, and every route returned
 * 200 - the map was simply invisible. Only rendering the page catches that.
 *
 * Usage:
 *   node scripts/ui_smoke_test.mjs [--url http://localhost:5173] [--shots DIR]
 *
 * Exits non-zero if any check fails, so it can gate a release.
 */

import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { spawn } from 'node:child_process'
import { join } from 'node:path'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'

const args = process.argv.slice(2)
const argOf = (flag, fallback) => {
  const i = args.indexOf(flag)
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback
}
const BASE = argOf('--url', 'http://localhost:5173')
const SHOTS = argOf('--shots', null)
const PORT = Number(argOf('--port', '9333'))

const BROWSERS = [
  '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/brave-browser',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
]

const browser = process.env.BROWSER_BIN || BROWSERS.find((p) => existsSync(p))
if (!browser) {
  console.error('No Chromium-family browser found. Set BROWSER_BIN to one.')
  process.exit(2)
}

const profile = mkdtempSync(join(tmpdir(), 'crackcatch-ui-'))
const proc = spawn(browser, [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  `--user-data-dir=${profile}`, '--window-size=1600,1200',
  `--remote-debugging-port=${PORT}`, 'about:blank',
], { stdio: 'ignore' })

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function waitForCdp(timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/version`)
      if (r.ok) return
    } catch { /* not up yet */ }
    await sleep(300)
  }
  throw new Error('browser did not expose a CDP endpoint')
}

function connect(wsUrl) {
  const ws = new WebSocket(wsUrl)
  const state = { id: 0, pending: new Map() }
  const events = { logs: [], failures: [] }
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data)
    if (msg.id && state.pending.has(msg.id)) {
      const { resolve, reject } = state.pending.get(msg.id)
      state.pending.delete(msg.id)
      msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result)
      return
    }
    if (msg.method === 'Runtime.exceptionThrown') {
      events.logs.push(msg.params.exceptionDetails.exception?.description ?? msg.params.exceptionDetails.text)
    }
    if (msg.method === 'Network.loadingFailed' && msg.params.blockedReason) {
      events.failures.push(`${msg.params.type} blocked=${msg.params.blockedReason}`)
    }
  }
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++state.id
      state.pending.set(id, { resolve, reject })
      ws.send(JSON.stringify({ id, method, params }))
      setTimeout(() => state.pending.has(id) && reject(new Error(`timeout: ${method}`)), 30000)
    })
  return { ws, send, events }
}

const PROBE = `(() => {
  const box = (s) => { const e = document.querySelector(s); if (!e) return null
    const r = e.getBoundingClientRect(); return { w: Math.round(r.width), h: Math.round(r.height) } }
  return JSON.stringify({
    map: box('.map-wrap'),
    tiles: document.querySelectorAll('.leaflet-tile').length,
    tilesLoaded: document.querySelectorAll('.leaflet-tile-loaded').length,
    legend: (document.querySelector('.map-legend') || {}).innerText || null,
    rows: document.querySelectorAll('table.data tbody tr').length,
    stats: document.querySelectorAll('.stat').length,
    heading: (document.querySelector('h2') || {}).innerText || null,
  })
})()`

const checks = []
const record = (name, ok, detail = '') => {
  checks.push({ name, ok, detail })
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`)
}

try {
  await waitForCdp()
  const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()
  const target = list.find((t) => t.type === 'page')
  const { ws, send, events } = connect(target.webSocketDebuggerUrl)
  await new Promise((r) => (ws.onopen = r))
  await send('Runtime.enable')
  await send('Network.enable')
  await send('Page.enable')

  if (SHOTS) mkdirSync(SHOTS, { recursive: true })

  const visit = async (path, waitMs = 9000) => {
    await send('Page.navigate', { url: `${BASE}${path}` })
    await sleep(waitMs)
    const { result } = await send('Runtime.evaluate', { expression: PROBE, returnByValue: true })
    if (SHOTS) {
      const { data } = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true })
      writeFileSync(join(SHOTS, `${path.replace(/\W+/g, '_') || 'root'}.png`), Buffer.from(data, 'base64'))
    }
    return JSON.parse(result.value)
  }

  console.log(`\nCrackCatch UI smoke test  (${browser.split('/').pop()})  ${BASE}\n`)

  const dash = await visit('/dashboard')
  record('dashboard: map has non-zero height', (dash.map?.h ?? 0) > 100, `${dash.map?.w}x${dash.map?.h}`)
  record('dashboard: basemap tiles loaded', dash.tilesLoaded > 0, `${dash.tilesLoaded}/${dash.tiles}`)
  record('dashboard: defect table rendered', dash.rows > 0, `${dash.rows} rows`)
  record('dashboard: stat cards rendered', dash.stats === 5, `${dash.stats} cards`)
  record('dashboard: pin legend shown', /Severe/.test(dash.legend ?? ''), (dash.legend ?? '').split('\n')[0])

  const analytics = await visit('/analytics', 10000)
  record('analytics: heat map has non-zero height', (analytics.map?.h ?? 0) > 100, `${analytics.map?.w}x${analytics.map?.h}`)
  record('analytics: heat legend shown', /Road health/.test(analytics.legend ?? ''), (analytics.legend ?? '').split('\n')[0])

  const report = await visit('/report', 5000)
  record('report: citizen page rendered', /Report a road defect/.test(report.heading ?? ''), report.heading)

  record('no uncaught exceptions', events.logs.length === 0, events.logs.slice(0, 2).join(' | '))
  record('no blocked requests', events.failures.length === 0, [...new Set(events.failures)].slice(0, 2).join(' | '))

  ws.close()
} catch (err) {
  record('smoke test completed', false, String(err))
} finally {
  proc.kill()
}

const failed = checks.filter((c) => !c.ok)
console.log(`\n${checks.length - failed.length}/${checks.length} checks passed`)
if (SHOTS) console.log(`screenshots: ${SHOTS}`)
process.exit(failed.length ? 1 : 0)
