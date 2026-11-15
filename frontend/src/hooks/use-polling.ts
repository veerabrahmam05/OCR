"use client"

import { useEffect, useRef, useState } from "react"

import { ApiError } from "@/lib/api"

/**
 * Calls `fetcher` immediately and then every `intervalMs` while `active(data)` is true.
 * Requests never overlap; the latest error is kept until the next success.
 */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  active: (data: T | undefined) => boolean = () => true,
  deps: unknown[] = [],
) {
  const [data, setData] = useState<T>()
  const [error, setError] = useState<Error>()
  const [tick, setTick] = useState(0)
  const fetcherRef = useRef(fetcher)
  const activeRef = useRef(active)
  // Keep the latest callbacks without restarting the polling loop. Declared before the
  // polling effect, so it runs first in the same commit.
  useEffect(() => {
    fetcherRef.current = fetcher
    activeRef.current = active
  })

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const run = async () => {
      let latest: T | undefined
      try {
        latest = await fetcherRef.current()
        if (cancelled) return
        setData(latest)
        setError(undefined)
      } catch (e) {
        if (cancelled) return
        setError(e as Error)
        // Client errors (e.g. 404: job gone after a server restart) won't fix themselves.
        if (e instanceof ApiError && e.status >= 400 && e.status < 500) return
      }
      if (!cancelled && activeRef.current(latest)) timer = setTimeout(run, intervalMs)
    }
    run()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, tick, ...deps])

  return { data, error, refresh: () => setTick((t) => t + 1) }
}
