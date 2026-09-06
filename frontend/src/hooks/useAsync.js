import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Run an async loader, tracking loading/error state and discarding results
 * from superseded calls so fast filter changes cannot render stale data.
 */
export function useAsync(loader, deps = []) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const requestId = useRef(0)

  const run = useCallback(async () => {
    const id = ++requestId.current
    setLoading(true)
    try {
      const result = await loader()
      if (id === requestId.current) {
        setData(result)
        setError(null)
      }
    } catch (err) {
      if (id === requestId.current) setError(err)
    } finally {
      if (id === requestId.current) setLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    run()
  }, [run])

  return { data, error, loading, reload: run, setData }
}
