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
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  Brush,
  YAxis,
} from "recharts";
import {
  getDrawdown,
  getEquity,
  getHealth,
  getHmmInfo,
  getPairs,
  getRegime,
  getRegimePerformance,
  getRollingSharpe,
  getSummary,
  getTradesWithPnL,
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
  useMacroTickers: false,
  // universe choices: 'top200' (all), or sector names matching src/data/universe.SECTOR_MAPPING
  universe: "top200",
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
function formatCurrency(n) {
  if (n == null || Number.isNaN(Number(n))) return "";
  return `$${Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function buildLegsSummary(legs = []) {
  try {
    return legs.map((l) => `${l.ticker}: ${Number(l.quantity).toFixed(2)} @ ${Number(l.fill_price).toFixed(2)}`).join("; ");
  } catch (e) {
    return "";
  }
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
  const [closedTrades,     setClosedTrades]     = useState([]);
  const [pairs,            setPairs]            = useState([]);
  const [regimeSeries,     setRegimeSeries]     = useState([]);
  const [regimePerf,       setRegimePerf]       = useState([]);
  const [controls,         setControls]         = useState(() => {
    try {
      const saved = window.localStorage.getItem("dashboard_controls");
      return saved ? JSON.parse(saved) : DEFAULT_CONTROLS;
    } catch { return DEFAULT_CONTROLS; }
  });
  const [hmmInfo,          setHmmInfo]          = useState(null);
  const [running,  setRunning]  = useState(false);
  const [error,    setError]    = useState("");
  const [chartRange, setChartRange] = useState({ start: null, end: null });

  const refreshData = async () => {
    const [h, s, e, dd, rs, p, reg, rp, tp, hmm] = await Promise.all([
      getHealth(),
      getSummary().catch(() => ({ ok: false })),
      getEquity().catch(() => ({ ok: false, equity: [] })),
      getDrawdown().catch(() => ({ ok: false, drawdown: [] })),
      getRollingSharpe(60).catch(() => ({ ok: false, rollingSharpe: [] })),
      getPairs().catch(() => ({ ok: false, pairs: [] })),
      getRegime().catch(() => ({ ok: false, regime: [] })),
      getRegimePerformance().catch(() => ({ ok: false, regimePerformance: [] })),
      getTradesWithPnL(500).catch(() => ({ ok: false, trades: [], closedTrades: [] })),
      getHmmInfo().catch(() => ({ ok: false })),
    ]);

    setHealth(h);
    if (s.ok)   setSummary(s);
    if (e.ok)   setEquity(e.equity || []);
    if (dd.ok)  setDrawdown(dd.drawdown || []);
    if (rs.ok)  setRollingSharpe(rs.rollingSharpe || []);
    if (tp.ok)  {
      setTrades(tp.trades || []);
      setClosedTrades(tp.closedTrades || []);
    }
    if (p.ok)   setPairs(p.pairs || []);
    if (reg.ok) setRegimeSeries(reg.regime || []);
    if (rp.ok)  setRegimePerf(rp.regimePerformance || []);
    if (hmm.ok) setHmmInfo(hmm.hmm_info || null);
  };

  useEffect(() => { refreshData(); }, []);
  useEffect(() => {
    window.localStorage.setItem("dashboard_controls", JSON.stringify(controls));
  }, [controls]);

  const onRun = async () => {
    setRunning(true);
    setError("");
    try {
      const overrides = {
        initialCapital:       Number(controls.initialCapital),
        trainPct:             Number(controls.trainPct),
        maxPairs:             Number(controls.maxPairs),
        reselectionInterval:  Number(controls.reselectionInterval),
        reselectionEnabled:   Boolean(controls.reselectionEnabled),
        entryZ:               Number(controls.entryZ),
        exitZ:                Number(controls.exitZ),
        stopZ:                Number(controls.stopZ),
        nStates:              Number(controls.nStates),
      };

      if (controls.useMacroTickers) {
        // Default macro tickers recommended by the training guide
        overrides.macroTickers = ["^VIX", "GLD", "TLT", "USO"];
      }

      const payload = {
        useRisk: controls.useRisk,
        overrides,
        universe: controls.universe,
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

  // HMM diagnostics derived client-side from `regimeSeries`
  const hmmDiagnostics = useMemo(() => {
    if (!regimeSeries || regimeSeries.length === 0) return null;
    // sort by date
    const rows = regimeSeries.slice().sort((a, b) => new Date(a.date) - new Date(b.date));
    const states = Array.from(new Set(rows.map((r) => r.value))).sort((a, b) => a - b);
    const n = states.length;
    // map state -> index
    const idx = {};
    states.forEach((s, i) => (idx[s] = i));

    // transition counts
    const counts = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
    for (let i = 1; i < rows.length; i++) {
      const a = idx[rows[i - 1].value];
      const b = idx[rows[i].value];
      if (a != null && b != null) counts[a][b] += 1;
    }

    // convert to probabilities by row
    const probs = counts.map((row) => {
      const s = row.reduce((acc, v) => acc + v, 0) || 1;
      return row.map((v) => v / s);
    });

    // run-lengths / durations
    const runs = [];
    let cur = rows[0].value;
    let len = 1;
    for (let i = 1; i < rows.length; i++) {
      if (rows[i].value === cur) len += 1;
      else {
        runs.push({ state: cur, length: len });
        cur = rows[i].value;
        len = 1;
      }
    }
    runs.push({ state: cur, length: len });

    const durationBars = states.map((s) => ({ state: s, avg_duration: (runs.filter((r) => r.state === s).reduce((a, b) => a + b.length, 0) / Math.max(1, runs.filter((r) => r.state === s).length)) }))
      .map((d) => ({ state: d.state, value: Number(d.avg_duration.toFixed(1)) }));

    return { states, probs, counts, durationBars };
  }, [regimeSeries]);

  const closedTradesById = useMemo(() => {
    const m = new Map();
    for (const c of closedTrades) {
      if (c?.roundtrip_id != null) m.set(c.roundtrip_id, c);
    }
    return m;
  }, [closedTrades]);

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
          <label>Universe
            <select value={controls.universe} onChange={(e) => setControls({ ...controls, universe: e.target.value })}>
              <option value="top200">Top 200 (All)</option>
              <option value="Technology">Technology</option>
              <option value="Financials">Financials</option>
              <option value="Healthcare">Healthcare</option>
              <option value="Consumer">Consumer</option>
              <option value="Energy">Energy</option>
              <option value="Industrials">Industrials</option>
            </select>
          </label>
          {controls.universe === "top200" && (
            <div className="warning-text">⚠ Running on the full universe may be slow. Expect longer backtest times.</div>
          )}
          <label className="toggle-row">
            <input type="checkbox" checked={controls.useRisk} onChange={(e) => setControls({ ...controls, useRisk: e.target.checked })} />
            Enable Risk Manager
          </label>
          <label className="toggle-row">
            <input type="checkbox" checked={controls.useMacroTickers} onChange={(e) => setControls({ ...controls, useMacroTickers: e.target.checked })} />
            Use Macro Tickers (VIX, GLD, TLT, USO)
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
          <div>
            <h3>Equity Curve &amp; Regime Timeline</h3>
            <p className="chart-subtitle">Colored bands show detected market regime. Use the brush below to zoom into any period.</p>
          </div>
          <RegimeLegend />
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart data={equityRegimeData} margin={{ top: 8, right: 20, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#63e6be" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#63e6be" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
            <XAxis
              dataKey="date"
              tickFormatter={formatDateLabel}
              tick={{ fontSize: 11, fill: "#9ab8cc" }}
              tickLine={false}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
            />
            <YAxis
              tickFormatter={(v) => `$${(v / 1_000_000).toFixed(2)}M`}
              width={76}
              tick={{ fontSize: 11, fill: "#9ab8cc" }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              contentStyle={{ background: "rgba(8,22,36,0.95)", border: "1px solid rgba(99,230,190,0.3)", borderRadius: 10, fontSize: 13 }}
              labelFormatter={(d) => <span style={{ color: "#63e6be", fontWeight: 600 }}>{formatDateLabel(d)}</span>}
              formatter={(value, name) => {
                if (name === "equity") return [`$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`, "Portfolio Value"];
                return [value, name];
              }}
            />
            {/* Regime colored background bands */}
            {regimeBands.map((band, i) => (
              <ReferenceArea
                key={i}
                x1={band.start}
                x2={band.end}
                fill={REGIME_META[band.regime]?.bg ?? "transparent"}
                ifOverflow="hidden"
                label={i === 0 || band.regime !== regimeBands[i - 1]?.regime ? undefined : undefined}
              />
            ))}
            {/* Initial capital baseline */}
            {stats.initial_capital ? (
              <ReferenceLine
                y={Number(stats.initial_capital)}
                stroke="rgba(255,255,255,0.25)"
                strokeDasharray="5 4"
                label={{ value: "Initial capital", position: "insideTopLeft", fontSize: 11, fill: "rgba(255,255,255,0.35)" }}
              />
            ) : null}
            <Area
              type="monotone"
              dataKey="equity"
              stroke="#63e6be"
              strokeWidth={2.5}
              fill="url(#equityGrad)"
              dot={false}
              activeDot={{ r: 5, stroke: "#63e6be", strokeWidth: 2, fill: "#0e1b2a" }}
              isAnimationActive={false}
            />
            <Brush
              dataKey="date"
              height={28}
              fill="rgba(14,27,42,0.8)"
              stroke="rgba(99,230,190,0.35)"
              travellerWidth={8}
              tickFormatter={formatDateLabel}
              onChange={(range) => {
                try {
                  if (!range) return;
                  const startIndex = range.startIndex ?? range.startIndex === 0 ? range.startIndex : null;
                  const endIndex = range.endIndex ?? range.endIndex === 0 ? range.endIndex : null;
                  if (startIndex == null || endIndex == null) return;
                  const start = equityRegimeData[startIndex]?.date ?? null;
                  const end = equityRegimeData[endIndex]?.date ?? null;
                  setChartRange({ start, end });
                } catch (e) { /* ignore */ }
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
        {chartRange.start && chartRange.end ? (
          <div className="chart-range">Selection: <strong>{formatDateLabel(chartRange.start)}</strong> — <strong>{formatDateLabel(chartRange.end)}</strong></div>
        ) : null}
        {/* Regime colour strip */}
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
          <div className="chart-header">
            <div>
              <h3>Drawdown from Peak</h3>
              <p className="chart-subtitle">How far the portfolio has fallen from its all-time high at each point in time.</p>
            </div>
            {stats.max_drawdown_pct != null && (
              <span className="stat-badge danger">Max {fmt(stats.max_drawdown_pct)}%</span>
            )}
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={drawdown} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ff8fa3" stopOpacity={0.5} />
                  <stop offset="100%" stopColor="#ff8fa3" stopOpacity={0.04} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
              <XAxis
                dataKey="date"
                tickFormatter={formatDateLabel}
                tick={{ fontSize: 11, fill: "#9ab8cc" }}
                tickLine={false}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              />
              <YAxis
                domain={["dataMin", 0]}
                tickFormatter={(v) => `${v.toFixed(0)}%`}
                width={44}
                tick={{ fontSize: 11, fill: "#9ab8cc" }}
                tickLine={false}
                axisLine={false}
              />
              {stats.max_drawdown_pct != null && (
                <ReferenceLine
                  y={Number(stats.max_drawdown_pct)}
                  stroke="rgba(255,107,107,0.6)"
                  strokeDasharray="5 3"
                  label={{ value: `Max DD ${fmt(stats.max_drawdown_pct)}%`, position: "insideBottomRight", fontSize: 11, fill: "rgba(255,107,107,0.8)" }}
                />
              )}
              <Tooltip
                contentStyle={{ background: "rgba(8,22,36,0.95)", border: "1px solid rgba(255,143,163,0.3)", borderRadius: 10, fontSize: 13 }}
                labelFormatter={(d) => <span style={{ color: "#ff8fa3", fontWeight: 600 }}>{formatDateLabel(d)}</span>}
                formatter={(v) => [`${Number(v).toFixed(2)}%`, "Drawdown"]}
              />
              <Area type="monotone" dataKey="value" stroke="#ff8fa3" strokeWidth={1.8} fill="url(#ddGrad)" dot={false} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </article>

        <article className="card chart-card">
          <div className="chart-header">
            <div>
              <h3>Rolling 60-Day Sharpe</h3>
              <p className="chart-subtitle">Green = positive risk-adjusted alpha. Red = strategy underperforming risk-free rate on this window.</p>
            </div>
            {stats.sharpe_ratio != null && (
              <span className={`stat-badge ${Number(stats.sharpe_ratio) >= 1 ? "success" : Number(stats.sharpe_ratio) >= 0 ? "warn" : "danger"}`}>
                Overall {fmt(stats.sharpe_ratio, 2)}
              </span>
            )}
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={rollingSharpe} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="sharpeGradPos" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#51cf66" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#51cf66" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="sharpeGradNeg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ff6b6b" stopOpacity={0.0} />
                  <stop offset="100%" stopColor="#ff6b6b" stopOpacity={0.35} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
              <XAxis
                dataKey="date"
                tickFormatter={formatDateLabel}
                tick={{ fontSize: 11, fill: "#9ab8cc" }}
                tickLine={false}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              />
              <YAxis
                width={40}
                tick={{ fontSize: 11, fill: "#9ab8cc" }}
                tickLine={false}
                axisLine={false}
              />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.4)" strokeDasharray="4 4" label={{ value: "0", position: "insideTopLeft", fontSize: 10, fill: "rgba(255,255,255,0.4)" }} />
              <ReferenceLine y={1} stroke="rgba(81,207,102,0.3)" strokeDasharray="3 4" label={{ value: "1.0 target", position: "insideTopLeft", fontSize: 10, fill: "rgba(81,207,102,0.5)" }} />
              <Tooltip
                contentStyle={{ background: "rgba(8,22,36,0.95)", border: "1px solid rgba(255,209,102,0.3)", borderRadius: 10, fontSize: 13 }}
                labelFormatter={(d) => <span style={{ color: "#ffd166", fontWeight: 600 }}>{formatDateLabel(d)}</span>}
                formatter={(v) => {
                  const n = Number(v);
                  return [<span style={{ color: n >= 0 ? "#51cf66" : "#ff6b6b" }}>{n.toFixed(2)}</span>, "Sharpe (60d)"];
                }}
              />
              <Area type="monotone" dataKey="value" stroke="none" fill="url(#sharpeGradPos)" dot={false} isAnimationActive={false} baseLine={0} />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#ffd166"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 5, stroke: "#ffd166", strokeWidth: 2, fill: "#0e1b2a" }}
                isAnimationActive={false}
              />
            </ComposedChart>
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

      {/* ── HMM Diagnostics ── */}
      {hmmDiagnostics && (
        <section className="card full-width hmm-section">
          <div className="chart-header" style={{ marginBottom: 16 }}>
            <div>
              <h3>HMM Regime Diagnostics</h3>
              <p className="chart-subtitle">Walk-forward trained Gaussian HMM with {controls.nStates} states. States ordered by ascending realised volatility (State 0 = low-vol/bull, State {controls.nStates - 1} = high-vol/bear).</p>
            </div>
          </div>

          <div className="hmm-grid">
            {/* Left: model or empirical transition matrix */}
            <div className="hmm-panel">
              <h4 className="hmm-panel-title">
                Transition Matrix
                <span className="hmm-source-badge">{hmmInfo?.transition_matrix ? "model" : "empirical"}</span>
              </h4>
              <p className="chart-subtitle" style={{ marginTop: 0 }}>Rows = from state · Cols = to state · Values = probability per bar. Diagonal shows regime persistence.</p>
              {/* Use model matrix from /api/hmm if available, else fall back to empirical */}
              {(() => {
                const modelMat = hmmInfo?.transition_matrix;
                const states = hmmDiagnostics.states;
                const mat = modelMat
                  ? modelMat
                  : hmmDiagnostics.probs;
                return (
                  <div style={{ overflowX: "auto" }}>
                    <table className="matrix-table">
                      <thead>
                        <tr>
                          <th style={{ width: 110 }}>From \ To</th>
                          {states.map((s) => (
                            <th key={`hcol-${s}`} style={{ textAlign: "center" }}>
                              <span className="state-chip" style={{ "--chip": REGIME_META[s]?.color ?? "#888" }}>
                                {REGIME_META[s]?.label ?? `State ${s}`}
                              </span>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {mat.map((row, i) => (
                          <tr key={`hrow-${i}`}>
                            <td>
                              <span className="state-chip" style={{ "--chip": REGIME_META[states[i]]?.color ?? "#888" }}>
                                {REGIME_META[states[i]]?.label ?? `State ${states[i]}`}
                              </span>
                            </td>
                            {row.map((p, j) => {
                              const isDiag = i === j;
                              const alpha = Math.min(0.92, p * 1.1 + 0.04);
                              return (
                                <td
                                  key={`cell-${i}-${j}`}
                                  className={isDiag ? "matrix-diag" : ""}
                                  style={{ background: `rgba(99,230,190,${isDiag ? alpha : alpha * 0.4})`, textAlign: "center", fontWeight: isDiag ? 600 : 400 }}
                                >
                                  {Number(p).toFixed(3)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                );
              })()}
            </div>

            {/* Centre: state durations */}
            <div className="hmm-panel">
              <h4 className="hmm-panel-title">Avg State Duration <span className="hmm-source-badge">days</span></h4>
              <p className="chart-subtitle" style={{ marginTop: 0 }}>How many consecutive bars the model tends to stay in each regime before switching.</p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={hmmDiagnostics.durationBars} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 11, fill: "#9ab8cc" }} tickLine={false} axisLine={false} />
                  <YAxis
                    type="category"
                    dataKey="state"
                    width={115}
                    tick={{ fontSize: 11, fill: "#9ab8cc" }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(s) => REGIME_META[s]?.label ?? `State ${s}`}
                  />
                  <Tooltip
                    contentStyle={{ background: "rgba(8,22,36,0.95)", border: "1px solid rgba(255,209,102,0.3)", borderRadius: 10, fontSize: 13 }}
                    formatter={(v) => [<strong>{v} days</strong>, "Avg duration"]}
                    labelFormatter={(s) => REGIME_META[s]?.label ?? `State ${s}`}
                  />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]} isAnimationActive={false}>
                    {hmmDiagnostics.durationBars.map((d, i) => (
                      <Cell key={i} fill={REGIME_META[d.state]?.color ?? "#ffd166"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              {/* Walk-forward bar counts from model */}
              {hmmInfo?.walkforward_counts && (
                <>
                  <h4 className="hmm-panel-title" style={{ marginTop: 14 }}>Walk-forward Label Distribution</h4>
                  <p className="chart-subtitle" style={{ marginTop: 0 }}>Bars assigned to each regime across the full walk-forward training period.</p>
                  <ResponsiveContainer width="100%" height={130}>
                    <BarChart
                      layout="vertical"
                      data={Object.entries(hmmInfo.walkforward_counts).map(([k, v]) => ({ state: Number(k), count: v }))}
                      margin={{ top: 4, right: 24, bottom: 4, left: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" horizontal={false} />
                      <XAxis type="number" tick={{ fontSize: 11, fill: "#9ab8cc" }} tickLine={false} axisLine={false} />
                      <YAxis
                        type="category"
                        dataKey="state"
                        width={115}
                        tick={{ fontSize: 11, fill: "#9ab8cc" }}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(s) => REGIME_META[s]?.label ?? `State ${s}`}
                      />
                      <Tooltip
                        contentStyle={{ background: "rgba(8,22,36,0.95)", border: "1px solid rgba(99,230,190,0.3)", borderRadius: 10, fontSize: 13 }}
                        formatter={(v) => [<strong>{v} bars</strong>, "Total days"]}
                        labelFormatter={(s) => REGIME_META[s]?.label ?? `State ${s}`}
                      />
                      <Bar dataKey="count" radius={[0, 6, 6, 0]} isAnimationActive={false}>
                        {Object.keys(hmmInfo.walkforward_counts).map((k, i) => (
                          <Cell key={i} fill={REGIME_META[Number(k)]?.color ?? "#63e6be"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </>
              )}
            </div>

            {/* Right: emission means per state feature */}
            <div className="hmm-panel">
              <h4 className="hmm-panel-title">State Emission Profiles <span className="hmm-source-badge">model means</span></h4>
              <p className="chart-subtitle" style={{ marginTop: 0 }}>Scaled feature means for each state. Positive logret + low rv_20 = bull; negative logret + high rv_20 = bear.</p>
              {hmmInfo?.emission_means && hmmInfo?.feature_cols ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={hmmInfo.feature_cols.map((feat, fi) => {
                      const row = { feature: feat };
                      hmmInfo.emission_means.forEach((means, si) => {
                        row[`state_${si}`] = Number(means[fi]?.toFixed(4) ?? 0);
                      });
                      return row;
                    })}
                    margin={{ top: 4, right: 8, bottom: 8, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                    <XAxis
                      dataKey="feature"
                      tick={{ fontSize: 11, fill: "#9ab8cc" }}
                      tickLine={false}
                      axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "#9ab8cc" }}
                      tickLine={false}
                      axisLine={false}
                      width={44}
                    />
                    <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" strokeDasharray="4 3" />
                    <Tooltip
                      contentStyle={{ background: "rgba(8,22,36,0.95)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 10, fontSize: 13 }}
                      formatter={(v, name) => {
                        const si = Number(name.replace("state_", ""));
                        return [Number(v).toFixed(4), REGIME_META[si]?.label ?? name];
                      }}
                    />
                    {hmmInfo.emission_means.map((_, si) => (
                      <Bar key={si} dataKey={`state_${si}`} name={`state_${si}`} fill={REGIME_META[si]?.color ?? "#888"} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p style={{ color: "#9ab8cc", fontSize: "0.82rem" }}>Run a backtest to populate HMM model internals.</p>
              )}
            </div>
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
                  <th>Round PnL</th>
                </tr>
              </thead>
              <tbody>
                  {trades.slice(-30).reverse().map((row, idx) => {
                    // find closed trade PnL if available
                    const rt = row.roundtrip_id;
                    const closed = rt != null ? closedTradesById.get(rt) : null;
                    return (
                      <tr key={`${row.date}-${row.ticker}-${idx}`}>
                        <td>{formatDateLabel(row.date)}</td>
                        <td>{row.pair_id}</td>
                        <td>{row.ticker}</td>
                        <td style={{ color: Number(row.quantity) > 0 ? "#51cf66" : "#ff8fa3" }}>
                          {fmt(row.quantity, 2)}
                        </td>
                        <td>{fmt(row.fill_price, 2)}</td>
                        <td>{fmt(row.commission, 2)}</td>
                        <td
                          style={{ color: closed && closed.realized_pnl >= 0 ? "#51cf66" : "#ff6b6b" }}
                          title={closed ? `Round-trip PnL: ${formatCurrency(closed.realized_pnl)}\nStart: ${formatDateLabel(closed.start_date)}\nEnd: ${formatDateLabel(closed.end_date)}\nLegs: ${buildLegsSummary(closed.legs)}` : ""}
                        >
                          {closed ? formatCurrency(closed.realized_pnl) : ""}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </article>
      </section>

      {/* ── Closed Round-Trips Panel ── */}
      <section className="card table-card full-width closed-trades-panel">
        <h3>Closed Round-Trips</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Round ID</th>
                <th>Pair</th>
                <th>Start</th>
                <th>End</th>
                <th>PnL</th>
                <th>Duration (d)</th>
                <th>Legs</th>
              </tr>
            </thead>
            <tbody>
              {closedTrades.slice().sort((a, b) => new Date(b.end_date) - new Date(a.end_date)).slice(0, 200).map((c, idx) => {
                const duration = c.start_date && c.end_date ? Math.round((new Date(c.end_date) - new Date(c.start_date)) / (1000 * 60 * 60 * 24)) : "-";
                return (
                  <tr key={`${c.roundtrip_id}-${idx}`} title={`Legs: ${buildLegsSummary(c.legs)}`}>
                    <td>{c.roundtrip_id}</td>
                    <td>{c.pair_id}</td>
                    <td>{formatDateLabel(c.start_date)}</td>
                    <td>{formatDateLabel(c.end_date)}</td>
                    <td style={{ color: c.realized_pnl >= 0 ? "#51cf66" : "#ff6b6b" }}>{formatCurrency(c.realized_pnl)}</td>
                    <td>{duration}</td>
                    <td>{(c.legs || []).length}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
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
