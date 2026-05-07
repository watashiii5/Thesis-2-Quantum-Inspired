'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import styles from './RLScheduler.module.css';

interface QiaRulesResponse {
  hard_penalty: number | null;
  hard_constraints: string[];
  soft_constraints: string[];
  source?: string;
}

interface TrainingStats {
  episodes_trained: number;
  avg_reward: number;
  best_reward: number;
  worst_reward: number;
  agent_stats: {
    q_table_size: number;
    experience_buffer_size: number;
    exploration_rate: number;
    learning_rate: number;
  };
}

interface BestResultResponse {
  has_result: boolean;
  best_reward: number | null;
  best_cost: number | null;
  best_schedule: Record<string, { room_id: string; time_slot: string }> | null;
  best_violations: Record<string, number>;
}

interface QiaRunSampleResponse {
  run_count: number;
  final_cost: number | null;
  result: unknown;
}

interface QiaBestResponse {
  run_count: number;
  best_cost: number | null;
  best_result: unknown;
}

export default function RLScheduler() {
  const [trainingStats, setTrainingStats] = useState<TrainingStats | null>(null);
  const [loading, setLoading] = useState(true);

  const [qiaRules, setQiaRules] = useState<QiaRulesResponse | null>(null);
  const [qiaBest, setQiaBest] = useState<QiaBestResponse | null>(null);
  const [qiaLastCost, setQiaLastCost] = useState<number | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [targetCost, setTargetCost] = useState<number | ''>('');
  const [loopCount, setLoopCount] = useState(0);

  const runningRef = useRef(false);
  const targetCostRef = useRef<number | null>(null);

  const apiUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  const qiaRunUrl = useMemo(() => `${apiUrl}/api/rl/qia/run-sample`, [apiUrl]);
  const qiaBestUrl = useMemo(() => `${apiUrl}/api/rl/qia/best`, [apiUrl]);

  // Fetch outcome + metadata on mount
  useEffect(() => {
    fetchQiaRules();
    fetchStats();
    fetchQiaBest();
  }, []);

  useEffect(() => {
    targetCostRef.current = targetCost === '' ? null : targetCost;
  }, [targetCost]);

  const fetchQiaRules = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/rl/qia-rules`);
      const data = (await response.json()) as QiaRulesResponse;
      setQiaRules(data);
    } catch (error) {
      console.error('Error fetching QIA rules:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/rl/stats`);
      const data = await response.json();
      setTrainingStats(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setLoading(false);
    }
  };

  const fetchQiaBest = async () => {
    try {
      const response = await fetch(qiaBestUrl);
      const data = (await response.json()) as QiaBestResponse;
      setQiaBest(data);
      return data;
    } catch (error) {
      console.error('Error fetching QIA best result:', error);
      return null;
    }
  };

  const runQiaOnce = async () => {
    try {
      const response = await fetch(qiaRunUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          // Keep this tiny and stable for the dashboard demo.
          max_iterations: 600,
          low_resource_mode: true,
          lunch_mode: 'auto',
        }),
      });

      if (!response.ok) {
        throw new Error('QIA run request failed');
      }

      const data = (await response.json()) as QiaRunSampleResponse;
      setQiaLastCost(data.final_cost ?? null);

      await fetchStats();
      await fetchQiaBest();
      return data;
    } catch (error) {
      console.error('Error running QIA sample:', error);
      throw error;
    }
  };

  const start = async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    setIsRunning(true);
    setLoopCount(0);

    try {
      let localLoops = 0;
      while (runningRef.current) {
        await runQiaOnce();
        localLoops += 1;
        setLoopCount(localLoops);

        const latestBest = await fetchQiaBest();
        const latestBestCost = latestBest?.best_cost ?? null;
        const target = targetCostRef.current;

        if (target !== null && latestBestCost !== null && latestBestCost <= target) {
          break;
        }

        await new Promise((r) => setTimeout(r, 250));
      }
    } catch {
      // handled via console
    } finally {
      runningRef.current = false;
      setIsRunning(false);
    }
  };

  const stop = () => {
    runningRef.current = false;
    setIsRunning(false);
  };

  const resetAgent = async () => {
    if (confirm('Are you sure? This will clear all learned data.')) {
      try {
        const response = await fetch(`${apiUrl}/api/rl/qia/reset`, { method: 'POST' });
        if (response.ok) {
          alert('Outcome reset successfully!');
          setQiaLastCost(null);
          fetchQiaBest();
        }
      } catch (error) {
        console.error('Error resetting outcome tracker:', error);
      }
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1>RL Scheduler - AI Learning Dashboard</h1>
        <p>Train your scheduler to learn custom rules and improve over time</p>
      </div>

      <div className={styles.tabContent}>
        <div className={styles.section}>
          <h2>Outcome Dashboard</h2>
          <p className={styles.subtitle}>Run the scheduler loop until you stop it or it reaches your target cost</p>

          {/* QIA Rules */}
          <div className={styles.card}>
            <h3>QIA Rules (Implemented Constraints)</h3>
            {qiaRules ? (
              <>
                <p className={styles.disclaimer}>
                  Hard penalty: {qiaRules.hard_penalty ?? 'N/A'} • Source: {qiaRules.source ?? 'N/A'}
                </p>
                <ul className={styles.exampleList}>
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
              <p className={styles.empty}>Loading QIA rules…</p>
            )}
          </div>

          {/* Controls */}
          <div className={styles.card}>
            <h3>Run Controls</h3>
            <div className={styles.formGrid}>
              <div className={styles.formGroup}>
                <label>Target Cost (optional)</label>
                <input
                  type="number"
                  min="0"
                  placeholder="e.g. 0"
                  value={targetCost}
                  onChange={(e) => setTargetCost(e.target.value === '' ? '' : Number(e.target.value))}
                />
              </div>
            </div>

            <div className={styles.trainingBox}>
              <div className={styles.trainingInfo}>
                <p>
                  This runs training continuously and tracks the best (lowest-cost) schedule seen so far.
                </p>
                <p className={styles.disclaimer}>Loops run: {loopCount}</p>
              </div>

              {isRunning ? (
                <button className={styles.btnDanger} onClick={stop}>
                  ⏹ Stop
                </button>
              ) : (
                <button className={styles.btnTrain} onClick={start}>
                  ▶ Start
                </button>
              )}

              <div style={{ marginTop: '12px' }}>
                <button className={styles.btnPrimary} onClick={resetAgent} disabled={isRunning}>
                  Reset Outcome
                </button>
              </div>
            </div>
          </div>

          {/* Outcome */}
          <div className={styles.card}>
            <h3>Current Best Outcome</h3>
            {qiaBest?.best_cost !== null ? (
              <>
                <div className={styles.statsGrid}>
                  <div className={styles.statCard}>
                    <h4>Best Cost</h4>
                    <p className={styles.statValue}>{qiaBest.best_cost ?? 0}</p>
                    <p className={styles.statLabel}>Lower is better</p>
                  </div>
                  <div className={styles.statCard}>
                    <h4>Runs</h4>
                    <p className={styles.statValue}>{qiaBest.run_count ?? 0}</p>
                    <p className={styles.statLabel}>Total QIA runs</p>
                  </div>
                  <div className={styles.statCard}>
                    <h4>Last Cost</h4>
                    <p className={styles.statValue}>{qiaLastCost ?? 0}</p>
                    <p className={styles.statLabel}>Most recent run</p>
                  </div>
                </div>

                <div className={styles.chart}>
                  <p className={styles.chartLabel}>Best Result (JSON)</p>
                  <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 }}>
                    {JSON.stringify(qiaBest.best_result, null, 2)}
                  </pre>
                </div>
              </>
            ) : (
              <p className={styles.empty}>No outcome yet. Click Start to begin.</p>
            )}
          </div>
              <h3>Advanced Options</h3>
              <button className={styles.btnDanger} onClick={resetAgent}>
                🔄 Reset Agent (Clear Learning)
              </button>
              <p className={styles.disclaimer}>
                Warning: This will erase all learned knowledge. Use only if you want to start fresh.
              </p>
            </div>
          </div>
        )}

        {/* Statistics Tab */}
        {activeTab === 'stats' && (
          <div className={styles.section}>
            <h2>Learning Progress</h2>
            <p className={styles.subtitle}>Monitor how well your AI is learning</p>

            {loading ? (
              <p>Loading statistics...</p>
            ) : trainingStats ? (
              <>
                {/* Stats Grid */}
                <div className={styles.statsGrid}>
                  <div className={styles.statCard}>
                    <h4>Episodes Trained</h4>
                    <p className={styles.statValue}>{trainingStats.episodes_trained}</p>
                    <p className={styles.statLabel}>training runs completed</p>
                  </div>
                  <div className={styles.statCard}>
                    <h4>Average Reward</h4>
                    <p className={styles.statValue}>{trainingStats.avg_reward.toFixed(2)}</p>
                    <p className={styles.statLabel}>higher is better</p>
                  </div>
                  <div className={styles.statCard}>
                    <h4>Best Reward</h4>
                    <p className={styles.statValue}>{trainingStats.best_reward.toFixed(2)}</p>
                    <p className={styles.statLabel}>peak performance</p>
                  </div>
                  <div className={styles.statCard}>
                    <h4>Q-Table Size</h4>
                    <p className={styles.statValue}>{trainingStats.agent_stats.q_table_size}</p>
                    <p className={styles.statLabel}>learned states</p>
                  </div>
                </div>

                {/* Learning Graph */}
                <div className={styles.card}>
                  <h3>Reward Over Time</h3>
                  {trainingHistory.length > 0 ? (
                    <div className={styles.chart}>
                      <div className={styles.chartBars}>
                        {trainingHistory.map((entry, idx) => (
                          <div key={idx} className={styles.bar}>
                            <div
                              className={styles.barFill}
                              style={{
                                height: `${Math.max(
                                  10,
                                  (entry.avg_reward / (trainingStats.best_reward || 100)) * 100
                                )}%`,
                              }}
                              title={`Episode ${entry.iteration}: ${entry.avg_reward.toFixed(2)}`}
                            ></div>
                            <p>{entry.iteration}</p>
                          </div>
                        ))}
                      </div>
                      <p className={styles.chartLabel}>Episode Progress</p>
                    </div>
                  ) : (
                    <p className={styles.empty}>No training history yet. Train the agent to see progress!</p>
                  )}
                </div>

                {/* Agent Details */}
                <div className={styles.card}>
                  <h3>Agent Configuration</h3>
                  <div className={styles.agentDetails}>
                    <div>
                      <strong>Exploration Rate:</strong> {trainingStats.agent_stats.exploration_rate}
                    </div>
                    <div>
                      <strong>Learning Rate:</strong> {trainingStats.agent_stats.learning_rate}
                    </div>
                    <div>
                      <strong>Experience Buffer:</strong>{' '}
                      {trainingStats.agent_stats.experience_buffer_size} transitions stored
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <p>No statistics available. Train the agent to see results.</p>
            )}
          </div>
        )}

        {/* Schedule Comparison Tab */}
        {activeTab === 'schedule' && (
          <div className={styles.section}>
            <h2>Schedule Comparison</h2>
            <p className={styles.subtitle}>Compare QIA vs RL-optimized schedules</p>

            <div className={styles.card}>
              <h3>QIA vs Reinforcement Learning</h3>
              <div className={styles.comparison}>

                    <div className={styles.chart}>
