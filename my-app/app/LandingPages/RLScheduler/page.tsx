'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import MenuBar from '@/app/components/MenuBar'
import Sidebar from '@/app/components/Sidebar'
import styles from './RLScheduler.module.css'

interface QiaRulesResponse {
  hard_penalty: number | null
  hard_constraints: string[]
  soft_constraints: string[]
  source?: string
}

interface QiaRunSampleResponse {
  run_count: number
  final_cost: number | null
  result: unknown
}

interface QiaBestResponse {
  run_count: number
  best_cost: number | null
  best_result: unknown
}

export default function RLScheduler() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const [qiaRules, setQiaRules] = useState<QiaRulesResponse | null>(null)
  const [qiaBest, setQiaBest] = useState<QiaBestResponse | null>(null)
  const [qiaLastCost, setQiaLastCost] = useState<number | null>(null)

  const [isRunning, setIsRunning] = useState(false)
  const [targetCost, setTargetCost] = useState<number | ''>('')
  const [loopCount, setLoopCount] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const runningRef = useRef(false)
  const targetCostRef = useRef<number | null>(null)

  const apiUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
  const qiaRulesUrl = useMemo(() => `${apiUrl}/api/rl/qia-rules`, [apiUrl])
  const qiaRunUrl = useMemo(() => `${apiUrl}/api/rl/qia/run-sample`, [apiUrl])
  const qiaBestUrl = useMemo(() => `${apiUrl}/api/rl/qia/best`, [apiUrl])
  const qiaResetUrl = useMemo(() => `${apiUrl}/api/rl/qia/reset`, [apiUrl])

  const getFriendlyFetchError = (rawMessage: string) => {
    if (typeof window === 'undefined') return rawMessage

    const isLocalFrontend = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    const isLocalBackend = apiUrl.includes('localhost') || apiUrl.includes('127.0.0.1')

    if (rawMessage === 'Failed to fetch' && isLocalBackend && !isLocalFrontend) {
      return 'Backend not reachable. This page is on Vercel (HTTPS) but the backend URL is still set to localhost. Set NEXT_PUBLIC_BACKEND_URL to your deployed backend HTTPS URL and redeploy.'
    }

    if (rawMessage === 'Failed to fetch' && isLocalBackend && isLocalFrontend) {
      return 'Backend not reachable at http://localhost:8000. Start the FastAPI backend (or set NEXT_PUBLIC_BACKEND_URL).' 
    }

    return rawMessage
  }

  useEffect(() => {
    if (typeof window === 'undefined') return
    const isLocalFrontend = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    const isLocalBackend = apiUrl.includes('localhost') || apiUrl.includes('127.0.0.1')
    if (!isLocalFrontend && isLocalBackend) {
      setError('Backend URL is not configured for production. Set NEXT_PUBLIC_BACKEND_URL to your deployed backend HTTPS URL and redeploy.')
    }
  }, [apiUrl])

  useEffect(() => {
    targetCostRef.current = targetCost === '' ? null : targetCost
  }, [targetCost])

  const fetchQiaRules = async () => {
    try {
      const response = await fetch(qiaRulesUrl)
      const data = (await response.json()) as QiaRulesResponse
      setQiaRules(data)
      return data
    } catch (e) {
      console.error('Error fetching QIA rules:', e)
      return null
    }
  }

  const fetchQiaBest = async () => {
    try {
      const response = await fetch(qiaBestUrl)
      const data = (await response.json()) as QiaBestResponse
      setQiaBest(data)
      return data
    } catch (e) {
      console.error('Error fetching QIA best result:', e)
      return null
    }
  }

  useEffect(() => {
    fetchQiaRules()
    fetchQiaBest()
  }, [qiaRulesUrl, qiaBestUrl])

  const runQiaOnce = async () => {
    const response = await fetch(qiaRunUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        max_iterations: 600,
        low_resource_mode: true,
        lunch_mode: 'auto',
      }),
    })

    if (!response.ok) {
      throw new Error('QIA run request failed')
    }

    const data = (await response.json()) as QiaRunSampleResponse
    setQiaLastCost(data.final_cost ?? null)
    await fetchQiaBest()
    return data
  }

  const start = async () => {
    if (runningRef.current) return

    setError(null)
    runningRef.current = true
    setIsRunning(true)
    setLoopCount(0)

    try {
      let localLoops = 0
      while (runningRef.current) {
        await runQiaOnce()
        localLoops += 1
        setLoopCount(localLoops)

        const latestBest = await fetchQiaBest()
        const latestBestCost = latestBest?.best_cost ?? null
        const target = targetCostRef.current

        if (target !== null && latestBestCost !== null && latestBestCost <= target) {
          break
        }

        await new Promise((r) => setTimeout(r, 250))
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Unexpected error'
      setError(getFriendlyFetchError(message))
    } finally {
      runningRef.current = false
      setIsRunning(false)
    }
  }

  const stop = () => {
    runningRef.current = false
    setIsRunning(false)
  }

  const resetOutcome = async () => {
    if (!confirm('Reset the in-memory best outcome tracking?')) return

    try {
      const response = await fetch(qiaResetUrl, { method: 'POST' })
      if (!response.ok) {
        throw new Error('Reset request failed')
      }

      setError(null)
      setLoopCount(0)
      setQiaLastCost(null)
      await fetchQiaBest()
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Unexpected error'
      setError(getFriendlyFetchError(message))
    }
  }

  return (
    <div className={styles.page} data-page="admin">
      <MenuBar
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        showSidebarToggle={true}
        showAccountIcon={true}
        setSidebarOpen={setSidebarOpen}
      />
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <main className={`${styles.main} ${!sidebarOpen ? styles.fullWidth : ''}`}>
        <div className={styles.container}>
          <div className={styles.header}>
            <div>
              <h1>QIA Outcome Dashboard</h1>
              <p>Runs the QIA scheduler continuously until you stop (or hit your target cost)</p>
            </div>
          </div>

          {error && <div className={styles.errorBanner}>{error}</div>}

          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2>Run Controls</h2>
              <div className={styles.actions}>
                {isRunning ? (
                  <button className={styles.dangerButton} onClick={stop}>
                    Stop
                  </button>
                ) : (
                  <button className={styles.primaryButton} onClick={start}>
                    Start
                  </button>
                )}
                <button className={styles.secondaryButton} onClick={resetOutcome} disabled={isRunning}>
                  Reset Outcome
                </button>
              </div>
            </div>

            <div className={styles.formRow}>
              <div className={styles.field}>
                <label className={styles.label} htmlFor="targetCost">
                  Target Cost (optional)
                </label>
                <input
                  id="targetCost"
                  className={styles.input}
                  type="number"
                  min={0}
                  placeholder="e.g. 0"
                  value={targetCost}
                  onChange={(e) => setTargetCost(e.target.value === '' ? '' : Number(e.target.value))}
                />
              </div>

              <div className={styles.fieldInline}>
                <div className={styles.inlineStat}>
                  <span className={styles.inlineLabel}>Loops</span>
                  <span className={styles.inlineValue}>{loopCount}</span>
                </div>
              </div>
            </div>
          </div>

          <div className={styles.statsGrid}>
            <div className={styles.statCard}>
              <p className={styles.statLabel}>Best Cost</p>
              <h3 className={styles.statValue}>{qiaBest?.best_cost ?? '—'}</h3>
            </div>
            <div className={styles.statCard}>
              <p className={styles.statLabel}>Runs</p>
              <h3 className={styles.statValue}>{qiaBest?.run_count ?? 0}</h3>
            </div>
            <div className={styles.statCard}>
              <p className={styles.statLabel}>Last Cost</p>
              <h3 className={styles.statValue}>{qiaLastCost ?? '—'}</h3>
            </div>
          </div>

          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2>Best Result (JSON)</h2>
            </div>
            <pre className={styles.pre}>{JSON.stringify(qiaBest?.best_result ?? null, null, 2)}</pre>
          </div>

          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2>QIA Rules (Implemented Constraints)</h2>
            </div>
            {qiaRules ? (
              <>
                <p className={styles.disclaimer}>
                  Hard penalty: {qiaRules.hard_penalty ?? 'N/A'} • Source: {qiaRules.source ?? 'N/A'}
                </p>
                <ul className={styles.rulesList}>
                  {qiaRules.hard_constraints.map((c) => (
                    <li key={`hard-${c}`}>
                      <strong>HARD:</strong> {c}
                    </li>
                  ))}
                  {qiaRules.soft_constraints.map((c) => (
                    <li key={`soft-${c}`}>
                      <strong>SOFT:</strong> {c}
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className={styles.disclaimer}>Loading QIA rules…</p>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
