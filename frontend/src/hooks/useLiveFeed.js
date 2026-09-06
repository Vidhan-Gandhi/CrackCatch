import { useCallback, useEffect, useRef, useState } from 'react'
import { websocketUrl } from '../lib/api'

/**
 * Subscribe to the backend's live defect feed.
 *
 * Reconnects with exponential backoff so a backend restart during a demo
 * heals itself instead of leaving a dead dashboard. Handlers are held in a ref
 * so that a caller passing an inline arrow function does not tear down and
 * rebuild the socket on every render.
 */
export function useLiveFeed(handlers = {}) {
  const [connected, setConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState(null)
  const socketRef = useRef(null)
  const handlersRef = useRef(handlers)
  const retryRef = useRef(0)
  const timerRef = useRef(null)
  const closedByUs = useRef(false)

  handlersRef.current = handlers

  const connect = useCallback(() => {
    if (socketRef.current) return
    let socket
    try {
      socket = new WebSocket(websocketUrl())
    } catch {
      return
    }
    socketRef.current = socket

    socket.onopen = () => {
      retryRef.current = 0
      setConnected(true)
    }

    socket.onmessage = (event) => {
      let message
      try {
        message = JSON.parse(event.data)
      } catch {
        return
      }
      if (message.type === 'pong') return
      setLastEvent(message)
      const handler = handlersRef.current[message.type]
      if (handler) handler(message.payload, message)
      handlersRef.current.onAny?.(message)
    }

    socket.onclose = () => {
      setConnected(false)
      socketRef.current = null
      if (closedByUs.current) return
      // 1s, 2s, 4s ... capped at 15s.
      const delay = Math.min(15000, 1000 * 2 ** retryRef.current)
      retryRef.current += 1
      timerRef.current = setTimeout(connect, delay)
    }

    socket.onerror = () => socket.close()
  }, [])

  useEffect(() => {
    closedByUs.current = false
    connect()

    // A keepalive stops intermediaries idling the socket out mid-demo.
    const ping = setInterval(() => {
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send('ping')
      }
    }, 25000)

    return () => {
      closedByUs.current = true
      clearInterval(ping)
      if (timerRef.current) clearTimeout(timerRef.current)
      socketRef.current?.close()
      socketRef.current = null
    }
  }, [connect])

  return { connected, lastEvent }
}
