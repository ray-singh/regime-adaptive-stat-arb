import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  getDrawdown,
  getEquity,
  getHealth,
  getPairs,
  getRegime,
  getRegimePerformance,
  getRollingSharpe,
  getSummary,
  getTrades,
  runBacktest,
} from "./api";

// ── Regime metadata ──────────────────────────────────────────────────────────
const REGIME_META = {
  0: { label: "Bull / Low-Vol",       color: "#51cf66", bg: "rgba(81,207,102,0.18)"  },
  1: { label: "Neutral / Mid-Vol",    color: "#ffd166", bg: "rgba(255,209,102,0.14)" },
  2: { label: "Bear / High-Vol",      color: "#ff8fa3", bg: "rgba(255,143,163,0.22)" },
  3: { label: "Crisis / Extreme-Vol", color: "#ff6b6b", bg: "rgba(255,107,107,0.30)" },
};

const DEFAULT_CONTROLS = {
  initialCapital: 1_000_000,
  trainPct: 0.5,
  maxPairs: 10,
  reselectionInterval: 63,
  reselectionEnabled: true,
  useRisk: true,
  entryZ: 2.0,
  exitZ: 0.5,
  stopZ: 3.5,
  nStates: 3,
};

// ── Helpers ──────────────────────────────────────────────────────────────────
function formatDateLabel(dateString) {
  const dt = new Date(dateString);
  if (Number.isNaN(dt.getTime())) return dateString;
  return dt.toISOString().slice(0, 10);
}

function fmt(n, decimals = 2) {
  return Number(n ?? 0).toFixed(decimals);
}

// ── Small components ─────────────────────────────────────────────────────────
function KpiCard({ label, value, hint, accent }) {
  return (
    <div className="kpi-card" style={accent ? { borderColor: accent } : {}}>
      <p className="kpi-label">{label}</p>
      <p className="kpi-value" style={accent ? { color: accent } : {}}>{value}</p>
      {hint ? <p className="kpi-hint">{hint}</p> : null}
    </div>
  );
}

function RegimeLegend() {
  return (
    <div className="regime-legend">
      {Object.values(REGIME_META).map((m) => (
        <span key={m.label} className="regime-dot" style={{ "--dot-color": m.color }}>
          {m.label}
        </span>
      ))}
    </div>
  );
}

// ── Regime-timeline overlay on equity chart ───────────────────────────────────
// We merge equity + regime series by date so Recharts can render both.
function buildEquityRegimeData(equityArr, regimeArr) {
  const regimeMap = {};
  for (const r of regimeArr) regimeMap[r.date?.slice(0, 10)] = r.value;

  return equityArr.map((row) => ({
    date:   row.date,
    equity: row.value,
    regime: regimeMap[row.date?.slice(0, 10)] ?? null,
  }));
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [health,           setHealth]           = useState(null);
  const [summary,          setSummary]          = useState(null);
  const [equity,           setEquity]           = useState([]);
  const [drawdown,         setDrawdown]         = useState([]);
  const [rollingSharpe,    setRollingSharpe]    = useState([]);
  const [trades,           setTrades]           = useState([]);
  const [pairs,            setPairs]            = useState([]);
  const [regimeSeries,     setRegimeSeries]     = useState([]);
  const [regimePerf,       setRegimePerf]       = useState([]);
  const [controls,         setControls]         = useState(() => {
    try {
      const saved = window.localStorage.getItem("dashboard_controls");
      return saved ? JSON.parse(saved) : DEFAULT_CONTROLS;
    } catch { return DEFAULT_CONTROLS; }
  });
  const [running,  setRunning]  = useState(false);
  const [error,    setError]    = useState("");

  const refreshData = async () => {
    const [h, s, e, dd, rs, t, p, reg, rp] = await Promise.all([
      getHealth(),
      getSummary().catch(() => ({ ok: false })),
      getEquity().catch(() => ({ ok: false, equity: [] })),
      getDrawdown().catch(() => ({ ok: false, drawdown: [] })),
      getRollingSharpe(60).catch(() => ({ ok: false, rollingSharpe: [] })),
      getTrades(120).catch(() => ({ ok: false, trades: [] })),
      getPairs().catch(() => ({ ok: false, pairs: [] })),
      getRegime().catch(() => ({ ok: false, regime: [] })),
      getRegimePerformance().catch(() => ({ ok: false, regimePerformance: [] })),
    ]);

    setHealth(h);
    if (s.ok)   setSummary(s);
    if (e.ok)   setEquity(e.equity || []);
    if (dd.ok)  setDrawdown(dd.drawdown || []);
    if (rs.ok)  setRollingSharpe(rs.rollingSharpe || []);
    if (t.ok)   setTrades(t.trades || []);
    if (p.ok)   setPairs(p.pairs || []);
    if (reg.ok) setRegimeSeries(reg.regime || []);
    if (rp.ok)  setRegimePerf(rp.regimePerformance || []);
  };

  useEffect(() => { refreshData(); }, []);
  useEffect(() => {
    window.localStorage.setItem("dashboard_controls", JSON.stringify(controls));
  }, [controls]);

  const onRun = async () => {
    setRunning(true);
    setError("");
    try {
      const payload = {
        useRisk: controls.useRisk,
        overrides: {
          initialCapital:       Number(controls.initialCapital),
          trainPct:             Number(controls.trainPct),
          maxPairs:             Number(controls.maxPairs),
          reselectionInterval:  Number(controls.reselectionInterval),
          reselectionEnabled:   Boolean(controls.reselectionEnabled),
          entryZ:               Number(controls.entryZ),
          exitZ:                Number(controls.exitZ),
          stopZ:                Number(controls.stopZ),
          nStates:              Number(controls.nStates),
        },
      };
      const result = await runBacktest(payload);
      if (!result.ok) setError(result.error || "Backtest failed");
      await refreshData();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setRunning(false);
    }
  };

  const stats = summary?.stats ?? {};
  const risk  = summary?.risk  ?? {};

  // Merged equity + regime for timeline chart
  const equityRegimeData = useMemo(
    () => buildEquityRegimeData(equity, regimeSeries),
    [equity, regimeSeries],
  );

  // Color-stop stripes behind equity curve (rendered as SVG linear gradient segments)
  // We mark regime changes so they can be rendered as reference areas.
  const regimeBands = useMemo(() => {
    if (!regimeSeries.length) return [];
    const bands = [];
    let cur = regimeSeries[0];
    for (let i = 1; i < regimeSeries.length; i++) {
      if (regimeSeries[i].value !== cur.value) {
        bands.push({ start: cur.date, end: regimeSeries[i - 1].date, regime: cur.value });
        cur = regimeSeries[i];
      }
    }
    bands.push({ start: cur.date, end: regimeSeries[regimeSeries.length - 1].date, regime: cur.value });
    return bands;
  }, [regimeSeries]);

  return (
    <main className="app-shell">
      {/* ── Hero ── */}
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Regime-Adaptive Statistical Arbitrage</p>
          <h1>Trading Intelligence Dashboard</h1>
          <p className="subtitle">
            Regime-aware backtest analytics — run strategies, inspect risk, monitor pair re-selection, and diagnose behaviour by market regime.
          </p>
        </div>
        <div className="hero-actions">
          <button className="run-button" onClick={onRun} disabled={running}>
            {running ? "Running Backtest…" : "▶ Run Backtest"}
          </button>
          <button className="ghost-button" onClick={refreshData}>⟳ Refresh</button>
        </div>
      </section>

      {/* ── Controls ── */}
      <section className="control-grid">
        <h2>Strategy Controls</h2>
        <div className="controls">
          <label>Initial Capital
            <input type="number" value={controls.initialCapital} onChange={(e) => setControls({ ...controls, initialCapital: e.target.value })} />
          </label>
          <label>Train %
            <input type="number" step="0.01" min="0.2" max="0.9" value={controls.trainPct} onChange={(e) => setControls({ ...controls, trainPct: e.target.value })} />
          </label>
          <label>Max Pairs
            <input type="number" min="1" max="30" value={controls.maxPairs} onChange={(e) => setControls({ ...controls, maxPairs: e.target.value })} />
          </label>
          <label>Re-selection Interval (days)
            <input type="number" min="21" max="252" value={controls.reselectionInterval} onChange={(e) => setControls({ ...controls, reselectionInterval: e.target.value })} />
          </label>
          <label>Entry Z (baseline)
            <input type="number" step="0.1" value={controls.entryZ} onChange={(e) => setControls({ ...controls, entryZ: e.target.value })} />
          </label>
          <label>Exit Z (baseline)
            <input type="number" step="0.1" value={controls.exitZ} onChange={(e) => setControls({ ...controls, exitZ: e.target.value })} />
          </label>
          <label>Stop Z
            <input type="number" step="0.1" value={controls.stopZ} onChange={(e) => setControls({ ...controls, stopZ: e.target.value })} />
          </label>
          <label>HMM States (n)
            <input type="number" min="2" max="4" value={controls.nStates} onChange={(e) => setControls({ ...controls, nStates: e.target.value })} />
          </label>
          <label className="toggle-row">
            <input type="checkbox" checked={controls.useRisk} onChange={(e) => setControls({ ...controls, useRisk: e.target.checked })} />
            Enable Risk Manager
          </label>
          <label className="toggle-row">
            <input type="checkbox" checked={controls.reselectionEnabled} onChange={(e) => setControls({ ...controls, reselectionEnabled: e.target.checked })} />
            Enable Pair Re-selection
          </label>
        </div>
        {error ? <p className="error-text">⚠ {error}</p> : null}
      </section>

      {/* ── KPIs ── */}
      <section className="kpi-grid">
        <KpiCard label="Final Equity"    value={`$${Number(stats.final_equity || 0).toLocaleString()}`}        hint="Portfolio end value" />
        <KpiCard label="CAGR"            value={`${fmt(stats.cagr_pct)}%`}                                      hint="Annualized growth" accent={Number(stats.cagr_pct) >= 0 ? "#51cf66" : "#ff6b6b"} />
        <KpiCard label="Sharpe"          value={fmt(stats.sharpe_ratio, 3)}                                     hint="Risk-adjusted return" accent={Number(stats.sharpe_ratio) >= 1 ? "#51cf66" : Number(stats.sharpe_ratio) >= 0 ? "#ffd166" : "#ff6b6b"} />
        <KpiCard label="Max Drawdown"    value={`${fmt(stats.max_drawdown_pct)}%`}                              hint="Worst peak-to-trough" accent="#ff8fa3" />
        <KpiCard label="Ann. Volatility" value={`${fmt(stats.ann_vol_pct)}%`}                                   hint="Daily vol × √252" />
        <KpiCard label="Sortino"         value={fmt(stats.sortino_ratio, 3)}                                    hint="Downside-only risk" />
        <KpiCard label="Risk Rejections" value={String(risk.orders_rejected ?? 0)}                              hint={`Rate ${fmt(risk.rejection_rate_pct)}%`} />
        <KpiCard label="Pair Re-selections" value={String(summary?.pairReselections ?? 0)}                      hint="Dynamic universe updates" />
      </section>

      {/* ── Equity curve with regime timeline (spec §4.1) ── */}
      <section className="card chart-card full-width">
        <div className="chart-header">
          <h3>Equity Curve &amp; Regime Timeline</h3>
          <RegimeLegend />
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={equityRegimeData} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="date" tickFormatter={formatDateLabel} hide />
            <YAxis tickFormatter={(v) => `$${(v / 1_000_000).toFixed(2)}M`} width={72} />
            <Tooltip
              labelFormatter={formatDateLabel}
              formatter={(value, name) => {
                if (name === "equity") return [`$${Number(value).toLocaleString()}`, "Equity"];
                if (name === "regime") return [REGIME_META[value]?.label ?? value, "Regime"];
                return [value, name];
              }}
            />
            {/* Regime bands as thin colored bars on a secondary 0-3 axis */}
              {/* hidden Y axis used to render regime bars correctly */}
              <YAxis yAxisId="regime" hide domain={[0, 3]} />
              <Bar dataKey="regime" yAxisId="regime" fill="transparent" isAnimationActive={false} />
            <Line
              type="monotone"
              dataKey="equity"
              stroke="#63e6be"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
        {/* Lightweight regime band strip below chart */}
        {regimeBands.length > 0 && (
          <div className="regime-strip" aria-label="Regime band strip">
            {regimeBands.map((band, i) => (
              <div
                key={i}
                className="regime-strip-seg"
                style={{ background: REGIME_META[band.regime]?.color ?? "#888", flex: 1 }}
                title={`${REGIME_META[band.regime]?.label ?? band.regime}: ${formatDateLabel(band.start)} – ${formatDateLabel(band.end)}`}
              />
            ))}
          </div>
        )}
      </section>

      {/* ── Drawdown + Rolling Sharpe ── */}
      <section className="chart-grid">
        <article className="card chart-card">
          <h3>Drawdown (%)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={drawdown} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="date" tickFormatter={formatDateLabel} hide />
              <YAxis domain={["dataMin", 0]} tickFormatter={(v) => `${v.toFixed(1)}%`} width={52} />
              <Tooltip labelFormatter={formatDateLabel} formatter={(v) => [`${Number(v).toFixed(2)}%`, "Drawdown"]} />
              <Area type="monotone" dataKey="value" stroke="#ff8fa3" fill="rgba(255,143,163,0.35)" />
            </AreaChart>
          </ResponsiveContainer>
        </article>

        <article className="card chart-card">
          <h3>Rolling 60-Day Sharpe</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={rollingSharpe} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="date" tickFormatter={formatDateLabel} hide />
              <YAxis width={40} />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" strokeDasharray="4 4" />
              <Tooltip labelFormatter={formatDateLabel} formatter={(v) => [Number(v).toFixed(2), "Sharpe"]} />
              <Line type="monotone" dataKey="value" stroke="#ffd166" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </article>
      </section>

      {/* ── Regime Performance Analysis (spec §4.4) ── */}
      {regimePerf.length > 0 && (
        <section className="card regime-perf-section">
          <h3>Regime Performance Analysis</h3>
          <div className="regime-perf-grid">
            {/* Table */}
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Regime</th>
                    <th>Days</th>
                    <th>Ann Return %</th>
                    <th>Sharpe</th>
                    <th>Ann Vol %</th>
                    <th>Trades</th>
                  </tr>
                </thead>
                <tbody>
                  {regimePerf.map((row, idx) => {
                    const sharpe = Number(row.Sharpe ?? 0);
                    const ret    = Number(row["Ann Return %"] ?? 0);
                    const color  = sharpe >= 0.5 ? "#51cf66" : sharpe >= 0 ? "#ffd166" : "#ff6b6b";
                    return (
                      <tr key={idx}>
                        <td>
                          <span className="regime-dot-inline" style={{ "--dot-color": Object.values(REGIME_META).find((m) => m.label === row.Regime)?.color ?? "#888" }} />
                          {row.Regime}
                        </td>
                        <td>{row.Days}</td>
                        <td style={{ color: ret >= 0 ? "#51cf66" : "#ff6b6b" }}>{fmt(ret)}%</td>
                        <td style={{ color }}>{fmt(sharpe, 2)}</td>
                        <td>{fmt(row["Ann Vol %"])}%</td>
                        <td>{row.Trades}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {/* Sharpe bar chart per regime */}
            <ResponsiveContainer width="100%" height={180}>
              <BarChart
                data={regimePerf}
                layout="vertical"
                margin={{ top: 4, right: 24, bottom: 0, left: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" horizontal={false} />
                <XAxis type="number" domain={["dataMin", "dataMax"]} />
                <YAxis type="category" dataKey="Regime" width={120} tick={{ fontSize: 11 }} />
                <ReferenceLine x={0} stroke="rgba(255,255,255,0.4)" />
                <Tooltip formatter={(v) => [Number(v).toFixed(2), "Sharpe"]} />
                <Bar dataKey="Sharpe" radius={[0, 4, 4, 0]}>
                  {regimePerf.map((row, idx) => (
                    <Cell
                      key={idx}
                      fill={Number(row.Sharpe) >= 0 ? "#51cf66" : "#ff6b6b"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {/* ── Pairs + Trades tables ── */}
      <section className="tables-grid">
        <article className="card table-card">
          <h3>Selected Pairs</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Pair</th>
                  <th>P-Value</th>
                  <th>Hedge Ratio</th>
                  <th>Half-Life (d)</th>
                </tr>
              </thead>
              <tbody>
                {pairs.slice(0, 20).map((row, idx) => (
                  <tr key={`${row.ticker1}-${row.ticker2}-${idx}`}>
                    <td><strong>{row.ticker1}/{row.ticker2}</strong></td>
                    <td>{fmt(row.pvalue, 4)}</td>
                    <td>{fmt(row.hedge_ratio, 3)}</td>
                    <td>{fmt(row.half_life_days, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="card table-card">
          <h3>Recent Trades</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Pair</th>
                  <th>Ticker</th>
                  <th>Qty</th>
                  <th>Price</th>
                  <th>Commission</th>
                </tr>
              </thead>
              <tbody>
                {trades.slice(-30).reverse().map((row, idx) => (
                  <tr key={`${row.date}-${row.ticker}-${idx}`}>
                    <td>{formatDateLabel(row.date)}</td>
                    <td>{row.pair_id}</td>
                    <td>{row.ticker}</td>
                    <td style={{ color: Number(row.quantity) > 0 ? "#51cf66" : "#ff8fa3" }}>
                      {fmt(row.quantity, 2)}
                    </td>
                    <td>{fmt(row.fill_price, 2)}</td>
                    <td>{fmt(row.commission, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </section>

      {/* ── Footer ── */}
      <footer className="status-footer">
        <span>API: {health?.ok ? "✓ Online" : "✗ Offline"}</span>
        <span>Backtest running: {String(health?.running ?? false)}</span>
        <span>Has result: {String(health?.hasResult ?? false)}</span>
        {health?.lastError && <span className="footer-error">Last error: {health.lastError}</span>}
      </footer>
    </main>
  );
}
