import { useEffect, useMemo, useState, useRef } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  Brush,
  YAxis,
  ZAxis,
} from "recharts";
import {
  getEquity,
  getHealth,
  getHmmInfo,
  getNetworkGraph,
  getPairs,
  getPairsByRegime,
  getPairSpread,
  getRankedPairs,
  getRegime,
  getRegimePerformance,
  getSummary,
  runDiscovery,
  getDiscoveryStatus,
} from "./api";

import { REGIME_META, DEFAULT_CONTROLS } from "./controls";
// Link to upstream repository (opens in new tab)
const GITHUB_REPO_URL = "https://github.com/ray-singh/regime-adaptive-stat-arb";
// ── Helpers ──────────────────────────────────────────────────────────────────
function formatDateLabel(dateString) {
  const dt = new Date(dateString);
  if (Number.isNaN(dt.getTime())) return dateString;
  return dt.toISOString().slice(0, 10);
}

// ── Small components ─────────────────────────────────────────────────────────
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

// ── Network View component (circular layout, SVG-based) ──────────────────────
function NetworkGraph({ nodes, edges, onSelectPair }) {
  const W = 700, H = 480, CX = W / 2, CY = H / 2;
  const R = Math.min(CX, CY) - 60;
  // Interactive pan/zoom state
  const [scale, setScale] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [panning, setPanning] = useState(false);
  const panLast = useRef(null);
  // Place nodes on circle
  const positions = useMemo(() => {
    const pos = {};
    nodes.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
      pos[n.id] = { x: CX + R * Math.cos(angle), y: CY + R * Math.sin(angle) };
    });
    return pos;
  }, [nodes, W, H]);

  // Colour edges by weight
  const maxWeight = useMemo(() => Math.max(...edges.map((e) => e.weight), 0.01), [edges]);

  // Wheel zoom handler (zoom toward cursor)
  const onWheel = (e) => {
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    const newScale = Math.max(0.2, Math.min(6, scale * factor));
    // adjust translation so zoom centers on cursor
    const dx = cx - (cx - tx) * (newScale / scale);
    const dy = cy - (cy - ty) * (newScale / scale);
    setScale(newScale);
    setTx(dx);
    setTy(dy);
  };

  const onMouseDown = (e) => {
    setPanning(true);
    panLast.current = { x: e.clientX, y: e.clientY };
  };
  const onMouseMove = (e) => {
    if (!panning || !panLast.current) return;
    const dx = e.clientX - panLast.current.x;
    const dy = e.clientY - panLast.current.y;
    panLast.current = { x: e.clientX, y: e.clientY };
    setTx((t) => t + dx);
    setTy((t) => t + dy);
  };
  const onMouseUp = () => { setPanning(false); panLast.current = null; };

  return (
    <div style={{ overflow: "hidden", touchAction: "none" }}>
      <svg
        width="100%"
        viewBox={`0 0 ${W} ${H}`}
        style={{ maxWidth: W, display: "block", margin: "0 auto", cursor: panning ? 'grabbing' : 'grab' }}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        <g transform={`translate(${tx},${ty}) scale(${scale})`}>
        {edges.map((e, i) => {
          const s = positions[e.source];
          const t = positions[e.target];
          if (!s || !t) return null;
          const opacity = 0.18 + 0.72 * (e.weight / maxWeight);
          return (
            <line
              key={i}
              x1={s.x} y1={s.y} x2={t.x} y2={t.y}
              stroke="#63e6be"
              strokeOpacity={opacity}
              strokeWidth={1 + 2.5 * (e.weight / maxWeight)}
              style={{ cursor: "pointer" }}
              onClick={() => onSelectPair && onSelectPair(e.pair_id)}
            >
              <title>{e.pair_id} · corr {Number(e.corr).toFixed(3)} · weight {Number(e.weight).toFixed(3)}</title>
            </line>
          );
        })}
        {/* Edge labels at midpoints */}
        {edges.map((e, i) => {
          const s = positions[e.source];
          const t = positions[e.target];
          if (!s || !t) return null;
          const mx = (s.x + t.x) / 2;
          const my = (s.y + t.y) / 2;
          return (
            <text key={`lbl-${i}`} x={mx} y={my} fontSize={10} fill="#c8dde8" textAnchor="middle" pointerEvents="none">
              {Number(e.corr).toFixed(2)}
            </text>
          );
        })}
        {nodes.map((n) => {
          const p = positions[n.id];
          if (!p) return null;
          const deg = edges.filter((e) => e.source === n.id || e.target === n.id).length;
          const r = 6 + Math.min(deg * 2, 12);
          return (
            <g key={n.id} transform={`translate(${p.x},${p.y})`}>
              <circle r={r} fill="rgba(99,230,190,0.18)" stroke="#63e6be" strokeWidth={1.5} />
              <text
                textAnchor="middle"
                dy={-r - 4}
                fontSize={10}
                fill="#c8dde8"
                fontWeight={600}
                pointerEvents="none"
              >{n.label}</text>
            </g>
          );
        })}
        </g>
      </svg>
      <p className="chart-subtitle" style={{ textAlign: "center", marginTop: 6 }}>
        Node size = degree (# of pair relationships). Click an edge to open Pair Explorer.
      </p>
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
  const [closedTrades,     setClosedTrades]     = useState([]);
  const [regimeSeries,     setRegimeSeries]     = useState([]);
  const [controls,         setControls]         = useState(() => {
    try {
      const saved = window.localStorage.getItem("dashboard_controls");
      if (!saved) return DEFAULT_CONTROLS;
      const parsed = JSON.parse(saved);
      const merged = { ...DEFAULT_CONTROLS, ...parsed };
      // Keep discovery in 4-state mode so crisis (regime 3) is available.
      merged.nStates = Math.max(4, Number(merged.nStates) || 4);
      return merged;
    } catch { return DEFAULT_CONTROLS; }
  });

  // ── Discovery state ──────────────────────────────────────────────────────
  const [discoveryRunning,  setDiscoveryRunning]  = useState(false);
  const [discoveryError,    setDiscoveryError]    = useState("");
  const [rankedPairs,       setRankedPairs]       = useState([]);
  const [pairsByRegime,     setPairsByRegime]     = useState({});
  const [selectedPair,      setSelectedPair]      = useState(null);
  const [pairSpread,        setPairSpread]        = useState(null);
  const [networkData,       setNetworkData]       = useState(null);
  const [networkRegime,     setNetworkRegime]     = useState("");
  const [topPairsRegimeFilter, setTopPairsRegimeFilter] = useState("");
  const [topPairsSortBy, setTopPairsSortBy] = useState("score"); // "score" | "stability"
  const [showAllPairs, setShowAllPairs] = useState(false);
  const [spreadViewMode,       setSpreadViewMode]       = useState("z"); // "z" | "spread"
  const [spreadHedgeMode,      setSpreadHedgeMode]      = useState("kalman"); // "kalman" | "static"

  const applyScenario = (s) => {
    if (!s) { setActiveScenario(null); return; }
    setActiveScenario(s.id);
    setControls((prev) => ({ ...prev, ...s.controls }));
  };

  // ── Discovery handlers ───────────────────────────────────────────────────
  const onDiscover = async () => {
    setDiscoveryRunning(true);
    setDiscoveryError("");
    try {
      // build tickers payload according to selected universe
      let tickersPayload;
      if (controls.universe === "top200") tickersPayload = [];
      else if (controls.universe === "custom") {
        tickersPayload = (controls.customTickers || "").split(",").map((s) => s.trim().toUpperCase()).filter(Boolean);
      } else tickersPayload = undefined;

      const res = await runDiscovery({
        tickers: tickersPayload,
        nStates: Math.max(4, Number(controls.nStates) || 4),
        minRegimeBars: Math.max(126, Number(controls.minRegimeBars) || 126),
      });
      if (!res.ok) { setDiscoveryError(res.error || "Discovery failed"); setDiscoveryRunning(false); return; }
      // Poll until done
      const poll = setInterval(async () => {
        try {
          const status = await getDiscoveryStatus();
          if (!status.running) {
            clearInterval(poll);
            setDiscoveryRunning(false);
            if (status.error) { setDiscoveryError(status.error); return; }
            if (status.hasResult) await refreshDiscovery();
          }
        } catch { clearInterval(poll); setDiscoveryRunning(false); }
      }, 2000);
    } catch (exc) {
      setDiscoveryError(String(exc));
      setDiscoveryRunning(false);
    }
  };

  const refreshDiscovery = async () => {
    setNetworkRegime("");
    const [rp, pbr, net] = await Promise.all([
      getRankedPairs().catch(() => ({ ok: false, pairs: [] })),
      getPairsByRegime().catch(() => ({ ok: false, byRegime: {} })),
      getNetworkGraph().catch(() => ({ ok: false, nodes: [], edges: [] })),
    ]);
    if (rp.ok)  setRankedPairs(rp.pairs || []);
    if (pbr.ok) setPairsByRegime(pbr.byRegime || {});
    if (net.ok) setNetworkData({ nodes: net.nodes || [], edges: net.edges || [] });
    else setNetworkData({ nodes: [], edges: [] });
  };

  const onSelectPair = async (pairId) => {
    setSelectedPair(pairId);
    setPairSpread(null);
    setSpreadViewMode("z");
    try {
      const res = await getPairSpread(pairId, spreadHedgeMode);
      if (res.ok) setPairSpread(res);
    } catch { /* ignore */ }
  };

  useEffect(() => {
    if (!selectedPair) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await getPairSpread(selectedPair, spreadHedgeMode);
        if (!cancelled && res.ok) setPairSpread(res);
      } catch {
        // ignore
      }
    })();
    return () => { cancelled = true; };
  }, [selectedPair, spreadHedgeMode]);

  const onNetworkRegimeChange = async (regime) => {
    setNetworkRegime(regime);
    try {
      const res = await getNetworkGraph(regime !== "" ? regime : undefined);
      if (res.ok) setNetworkData({ nodes: res.nodes || [], edges: res.edges || [] });
      else setNetworkData({ nodes: [], edges: [] });
    } catch {
      setNetworkData({ nodes: [], edges: [] });
    }
  };

  const refreshData = async () => {
    const [h, s, e, dd, rs, p, reg, rp, tp, hmm] = await Promise.all([
      getHealth(),
      getSummary().catch(() => ({ ok: false })),
      getEquity().catch(() => ({ ok: false, equity: [] })),
      getPairs().catch(() => ({ ok: false, pairs: [] })),
      getRegime().catch(() => ({ ok: false, regime: [] })),
      getRegimePerformance().catch(() => ({ ok: false, regimePerformance: [] })),
      getHmmInfo().catch(() => ({ ok: false })),
    ]);

    setHealth(h);
    if (s.ok)   setSummary(s);
    if (p.ok)   setPairs(p.pairs || []);
    if (reg.ok) setRegimeSeries(reg.regime || []);
    if (rp.ok)  setRegimePerf(rp.regimePerformance || []);
    if (hmm.ok) setHmmInfo(hmm.hmm_info || null);
  };

  useEffect(() => { refreshData(); }, []);
  useEffect(() => { refreshDiscovery(); }, []);
  useEffect(() => {
    window.localStorage.setItem("dashboard_controls", JSON.stringify(controls));
  }, [controls]);

  const stats = summary?.stats ?? {};
  const risk  = summary?.risk  ?? {};

  // Merged equity + regime for timeline chart
  const equityRegimeData = useMemo(
    () => buildEquityRegimeData(equity, regimeSeries),
    [equity, regimeSeries],
  );

  // Which pairs to show in Top Pairs Panel (optionally filter by regime)
  const displayedPairs = useMemo(() => {
    // Build a map: pair_id -> list of regimes where it was discovered
    const activeMap = {};
    Object.entries(pairsByRegime || {}).forEach(([regime, prs]) => {
      (prs || []).forEach((p) => {
        if (!p?.pair_id) return;
        if (!activeMap[p.pair_id]) activeMap[p.pair_id] = [];
        if (!activeMap[p.pair_id].includes(Number(regime))) activeMap[p.pair_id].push(Number(regime));
      });
    });

    let basePairs;
    if (!topPairsRegimeFilter) {
      basePairs = (rankedPairs || []).map((p, i) => {
        // Prefer directly-embedded stable/unstable lists from the ranking payload
        // (covers ALL pairs, not just the top-20 per regime in pairsByRegime).
        // Fall back to the activeMap derived from pairsByRegime, then to empty.
        let active_regimes;
        if (Array.isArray(p.stable_regimes) || Array.isArray(p.unstable_regimes)) {
          active_regimes = [
            ...(p.stable_regimes || []),
            ...(p.unstable_regimes || []),
          ].map(Number).sort((a, b) => a - b);
        } else {
          active_regimes = activeMap[p.pair_id] || [];
        }
        return {
          pair_id: p.pair_id,
          rank: p.rank ?? i + 1,
          score: p.score ?? 0,
          stability_score: p.stability_score ?? 0,
          stability_label: p.stability_label ?? "—",
          regime_sensitive: !!p.regime_sensitive,
          best_half_life_days: p.best_half_life_days,
          best_coint_pvalue: p.best_coint_pvalue ?? p.coint_pvalue,
          n_regimes_active: active_regimes.length || p.n_regimes_active || 0,
          active_regimes,
        };
      });
    } else {
      const arr = pairsByRegime?.[topPairsRegimeFilter] || [];
      basePairs = arr.map((p, i) => ({
        pair_id: p.pair_id,
        rank: i + 1,
        score: p.score ?? 0,
        stability_score: p.stability_score ?? 0,
        stability_label: p.stability_label ?? "—",
        regime_sensitive: !!p.regime_sensitive,
        best_half_life_days: p.half_life_days ?? p.best_half_life_days,
        best_coint_pvalue: p.coint_pvalue ?? p.best_coint_pvalue,
        n_regimes_active: (activeMap[p.pair_id]?.length) ?? p.n_regimes_active ?? 0,
        active_regimes: activeMap[p.pair_id] || [Number(topPairsRegimeFilter)],
      }));
    }

    const sorted = [...basePairs].sort((a, b) => {
      if (topPairsSortBy === "stability") {
        return Number(b.stability_score ?? 0) - Number(a.stability_score ?? 0);
      }
      return Number(b.score ?? 0) - Number(a.score ?? 0);
    });
    return sorted.map((p, i) => ({ ...p, rank: i + 1 }));
  }, [rankedPairs, pairsByRegime, topPairsRegimeFilter, topPairsSortBy]);

  // Limit used by the Top Pairs panel; default top-N when not showing all
  const TOP_PAIRS_DEFAULT = 20;
  const displayedLimit = showAllPairs ? (displayedPairs || []).length : TOP_PAIRS_DEFAULT;


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
      {/* ═══════════════════════════════════════════════════════════════════
          PAIR RELATIONSHIP DISCOVERY  (spec §4–6)
          ═══════════════════════════════════════════════════════════════════ */}
      <section className="card discovery-hero">
        <div className="discovery-hero-text">
          <p className="eyebrow">Pair Relationship Discovery</p>
          <h2>Regime-Aware Pair Browser</h2>
          <p className="subtitle">
            Discover cointegrated asset pairs, analyse how relationships shift across market regimes,
            and explore the most statistically interesting pairs.
          </p>
          <br/>
          <p className="subtitle">
            <strong>Warning:</strong> Discovery pipeline duration depends on VM availability, cache state, and selected universe size. If machines have recently restarted or cache is empty, the service must re-download price data and recompute HMM/feature caches, which can take ~10 minutes to run.
          </p>          
        </div>
        <div className="hero-actions">
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
              Universe:
              <select
                value={controls.universe}
                onChange={(e) => setControls((c) => ({ ...c, universe: e.target.value }))}
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, color: "#e0eaf0", padding: "4px 8px" }}
              >
                <option value="top100">Top 100 US equities</option>
                <option value="top200">Top 200 US equities</option>
                <option value="custom">Custom tickers</option>
              </select>
            </label>
            {controls.universe === "custom" && (
              <input
                placeholder="Comma-separated tickers (e.g. AAPL,MSFT,GOOG)"
                value={controls.customTickers || ""}
                onChange={(e) => setControls((c) => ({ ...c, customTickers: e.target.value }))}
                style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", color: "#e0eaf0", padding: "6px 8px", borderRadius: 6, minWidth: 280 }}
              />
            )}
            <button className="run-button" onClick={onDiscover} disabled={discoveryRunning}>
              {discoveryRunning ? "Discovering…" : "▶ Run Discovery"}
            </button>
            <button
              className="ghost-button"
              onClick={() => window.open(GITHUB_REPO_URL, "_blank")}
              title="Open project on GitHub"
              style={{ marginLeft: 6 }}
            >
              GitHub
            </button>
          </div>
        </div>
        {discoveryError && <p className="error-text" style={{ marginTop: 10 }}>⚠ {discoveryError}</p>}
        {discoveryRunning && (
          <p className="chart-subtitle" style={{ marginTop: 8, color: "#ffd166" }}>
            Running pipeline: fetching prices → features → regime detection → pair discovery…
          </p>
        )}
      </section>

      {/* ── Top Pairs Panel (spec §6 — Pair Ranking) ─────────────────────── */}
      {rankedPairs.length > 0 && (
        <section className="card table-card full-width">
              <div className="chart-header">
                <div>
                  <h3>Top Pairs Panel</h3>
                  <p className="chart-subtitle">
                    Pairs ranked by regime sensitivity and mean-reversion strength. Click a row to explore the spread.
                  </p>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    Regime:
                    <select
                      value={topPairsRegimeFilter}
                      onChange={(e) => setTopPairsRegimeFilter(e.target.value)}
                      style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 6, color: "#e0eaf0", padding: "4px 8px" }}
                    >
                      <option value="">All</option>
                      {Object.keys(REGIME_META).sort((a,b) => Number(a) - Number(b)).map((r) => (
                        <option key={r} value={r}>{`${REGIME_META[Number(r)]?.label ?? `Regime ${r}`} (${(pairsByRegime?.[r]?.length) ?? 0})`}</option>
                      ))}
                    </select>
                  </label>
                  <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    Sort:
                    <select
                      value={topPairsSortBy}
                      onChange={(e) => setTopPairsSortBy(e.target.value)}
                      style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 6, color: "#e0eaf0", padding: "4px 8px" }}
                    >
                      <option value="score">Score</option>
                      <option value="stability">Stability Score</option>
                    </select>
                  </label>
                  <span className="stat-badge">{(displayedPairs || []).length} pairs</span>
                  <button className="ghost-button" onClick={() => setShowAllPairs((s) => !s)} style={{ marginLeft: 8 }}>
                    {showAllPairs ? `Show Top ${TOP_PAIRS_DEFAULT}` : "Show All"}
                  </button>
                </div>
              </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Pair</th>
                  <th>Score</th>
                  <th>Stability</th>
                  <th>Regime Sensitive</th>
                  <th>Best Half-Life (d)</th>
                  <th>Best p-value</th>
                  <th>Active Regimes</th>
                </tr>
              </thead>
              <tbody>
                {displayedPairs.slice(0, displayedLimit).map((p, i) => (
                  <tr
                    key={p.pair_id}
                    onClick={() => onSelectPair(p.pair_id)}
                    className={selectedPair === p.pair_id ? "selected-row" : "clickable-row"}
                    title="Click to open Pair Explorer"
                  >
                    <td>{p.rank ?? i + 1}</td>
                    <td><strong>{p.pair_id}</strong></td>
                    <td style={{ color: "#ffd166" }}>{Number(p.score ?? 0).toFixed(3)}</td>
                    <td>
                      <span className={`stability-badge stability-${p.stability_label}`}>
                        {p.stability_label ?? "—"}
                      </span>
                    </td>
                    <td style={{ color: p.regime_sensitive ? "#ff8fa3" : "#51cf66" }}>
                      {p.regime_sensitive ? "Yes" : "No"}
                    </td>
                    <td>{p.best_half_life_days != null ? Number(p.best_half_life_days).toFixed(1) : "—"}</td>
                    <td style={{ color: Number(p.best_coint_pvalue) < 0.01 ? "#51cf66" : "#ffd166" }}>
                      {p.best_coint_pvalue != null ? Number(p.best_coint_pvalue).toExponential(2) : "—"}
                    </td>
                    <td>
                      {((p.active_regimes || [])?.length > 0) ? (
                        (p.active_regimes || []).map((r) => (
                          <span
                            key={`${p.pair_id}-r-${r}`}
                            className="regime-badge"
                            style={{
                              background: REGIME_META[Number(r)]?.color ?? "#888",
                              color: "#071829",
                              padding: "2px 8px",
                              borderRadius: 8,
                              fontSize: 11,
                              marginRight: 6,
                              display: "inline-block",
                              fontWeight: 700,
                            }}
                            title={REGIME_META[Number(r)]?.label ?? `Regime ${r}`}
                          >
                            {REGIME_META[Number(r)]?.label?.split(" ")?.[0] ?? `R${r}`}
                          </span>
                        ))
                      ) : (
                        <span>{p.n_regimes_active ?? "—"}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ── Pair Analytics: Score Bar Chart + Half-Life Scatter ─────────── */}
      {rankedPairs.length > 2 && (
        <section className="card full-width" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32, alignItems: "start" }}>
          {/* Score bar chart */}
          <div>
            <div className="chart-header" style={{ marginBottom: 12 }}>
              <div>
                <h3>Pair Scores</h3>
                <p className="chart-subtitle">Click a bar to open its spread. Colour = stability.</p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={Math.max(180, Math.min(displayedLimit, 18) * 26)}>
              <BarChart
                data={[...displayedPairs].slice(0, Math.min(displayedLimit, 18)).reverse().map((p) => ({ ...p, displayScore: Number(p.score ?? 0) }))}
                layout="vertical"
                margin={{ top: 0, right: 36, bottom: 0, left: 82 }}
                onClick={(d) => d?.activePayload?.[0] && onSelectPair(d.activePayload[0].payload.pair_id)}
              >
                <CartesianGrid strokeDasharray="2 2" stroke="rgba(255,255,255,0.04)" horizontal={false} />
                <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11, fill: "#98a6b3" }} tickLine={false} tickFormatter={(v) => v.toFixed(2)} />
                <YAxis type="category" dataKey="pair_id" tick={{ fontSize: 11, fill: "#c8dde8" }} tickLine={false} width={80} />
                <Tooltip
                  contentStyle={{ background: "rgba(8,22,36,0.95)", border: "1px solid rgba(99,230,190,0.18)", borderRadius: 10, fontSize: 12 }}
                  formatter={(v, _name, props) => {
                    const stab = props.payload?.stability_label;
                    const col = stab === "high" ? "#51cf66" : stab === "medium" ? "#ffd166" : "#ff8fa3";
                    return [
                      <span key="sv"><span style={{ color: "#ffffff" }}>{Number(v).toFixed(3)}</span> <span style={{ color: col }}>({stab})</span></span>,
                      <span key="sname" style={{ color: "#ffffff" }}>Score</span>
                    ];
                  }}
                />
                  <Bar dataKey="displayScore" radius={[0, 4, 4, 0]} cursor="pointer" isAnimationActive={false}>
                  {[...displayedPairs].slice(0, Math.min(displayedLimit, 18)).reverse().map((p, i) => (
                    <Cell
                      key={`sc-${i}`}
                      fill={p.stability_label === "high" ? "#51cf66" : p.stability_label === "medium" ? "#ffd166" : "#ff8fa3"}
                      opacity={selectedPair === p.pair_id ? 1 : 0.72}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ display: "flex", gap: 14, padding: "6px 0 0", fontSize: 11, color: "#98a6b3" }}>
              {[["high", "#51cf66"], ["medium", "#ffd166"], ["low", "#ff8fa3"]].map(([l, c]) => (
                <span key={l} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 9, height: 9, borderRadius: 2, background: c, display: "inline-block" }} />
                  {l}
                </span>
              ))}
            </div>
          </div>

          {/* Half-life vs Score scatter */}
          <div>
            <div className="chart-header" style={{ marginBottom: 12 }}>
              <div>
                <h3>Half-Life vs Score</h3>
                <p className="chart-subtitle">Bubble size = active regimes. Click to explore spread.</p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={Math.max(180, Math.min(displayedLimit, 18) * 26)}>
              <ScatterChart margin={{ top: 10, right: 24, bottom: 28, left: 10 }}>
                <CartesianGrid strokeDasharray="2 2" stroke="rgba(255,255,255,0.04)" />
                <XAxis
                  dataKey="best_half_life_days" type="number" name="Half-Life"
                  tick={{ fontSize: 11, fill: "#98a6b3" }} tickLine={false}
                  label={{ value: "Half-Life (days)", position: "insideBottom", offset: -14, fill: "#98a6b3", fontSize: 11 }}
                />
                <YAxis
                  dataKey="score" type="number" name="Score" domain={[0, 1]}
                  tick={{ fontSize: 11, fill: "#98a6b3" }} tickLine={false}
                  label={{ value: "Score", angle: -90, position: "insideLeft", fill: "#98a6b3", fontSize: 11 }}
                />
                <ZAxis dataKey="n_regimes_active" range={[40, 180]} name="Active Regimes" />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3", stroke: "rgba(255,255,255,0.15)" }}
                  contentStyle={{ background: "rgba(8,22,36,0.95)", border: "1px solid rgba(99,230,190,0.18)", borderRadius: 10, fontSize: 12 }}
                  formatter={(v, name) => {
                    if (name === "Score") {
                      return [<span style={{ color: "#ffffff" }}>{Number(v).toFixed(3)}</span>, <span style={{ color: "#ffffff" }}>Score</span>];
                    }
                    return [name === "Half-Life" ? `${Number(v).toFixed(1)}d` : v, name];
                  }}
                  labelFormatter={(_lbl, payload) => <strong style={{ color: "#63e6be" }}>{payload?.[0]?.payload?.pair_id}</strong>}
                />
                {["high", "medium", "low"].map((stab) => (
                  <Scatter
                    key={stab}
                    name={`${stab[0].toUpperCase()}${stab.slice(1)} stability`}
                    data={displayedPairs.slice(0, displayedLimit).filter((p) => (p.stability_label ?? "low") === stab && p.best_half_life_days != null)}
                    fill={stab === "high" ? "#51cf66" : stab === "medium" ? "#ffd166" : "#ff8fa3"}
                    opacity={0.85}
                    cursor="pointer"
                    onClick={(d) => onSelectPair(d.pair_id)}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {/* ── Pair Explorer (spec — spread chart + regime overlay) ────────── */}
      {selectedPair && (
        <section className="card chart-card full-width">
          <div className="chart-header">
            <div>
              <h3>Pair Explorer — {selectedPair}</h3>
              <p className="chart-subtitle">
                {spreadViewMode === "z" ? "Z-score with regime bands and ±2σ entry / exit thresholds." : "Raw spread value with regime overlays."}
                {pairSpread && ` Hedge: ${pairSpread.hedge_mode === "kalman" ? "Kalman (strategy-consistent)" : "Static OLS"}.`}
                {pairSpread && ` Current hedge ratio: ${Number(pairSpread.hedge_ratio).toFixed(4)}`}
              </p>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ display: "flex", gap: 4 }}>
                {["kalman", "static"].map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setSpreadHedgeMode(mode)}
                    style={{
                      padding: "3px 10px", borderRadius: 6, fontSize: 12, cursor: "pointer",
                      background: spreadHedgeMode === mode ? "rgba(255,209,102,0.16)" : "transparent",
                      border: `1px solid ${spreadHedgeMode === mode ? "#ffd166" : "rgba(255,255,255,0.1)"}`,
                      color: spreadHedgeMode === mode ? "#ffd166" : "#98a6b3",
                    }}
                  >
                    {mode === "kalman" ? "Kalman Hedge" : "Static Hedge"}
                  </button>
                ))}
              </div>
              <div style={{ display: "flex", gap: 4 }}>
                {["z", "spread"].map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setSpreadViewMode(mode)}
                    style={{
                      padding: "3px 10px", borderRadius: 6, fontSize: 12, cursor: "pointer",
                      background: spreadViewMode === mode ? "rgba(99,230,190,0.15)" : "transparent",
                      border: `1px solid ${spreadViewMode === mode ? "#63e6be" : "rgba(255,255,255,0.1)"}`,
                      color: spreadViewMode === mode ? "#63e6be" : "#98a6b3",
                    }}
                  >
                    {mode === "z" ? "Z-Score" : "Raw Spread"}
                  </button>
                ))}
              </div>
              <button className="ghost-button" onClick={() => { setSelectedPair(null); setPairSpread(null); }}>✕ Close</button>
            </div>
          </div>
          {!pairSpread && <p className="chart-subtitle" style={{ padding: "20px 0", color: "#98a6b3" }}>Loading spread data…</p>}
          {pairSpread && pairSpread.spread && (
            <>
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart
                  data={pairSpread.spread}
                  margin={{ top: 8, right: 20, bottom: 0, left: 0 }}
                >
                  <defs>
                    <linearGradient id="spreadGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#63e6be" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#63e6be" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="2 2" stroke="rgba(255,255,255,0.04)" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={formatDateLabel}
                    tick={{ fontSize: 11, fill: "#98a6b3", fontWeight: 600 }}
                    tickLine={false}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                  />
                  <YAxis
                    width={46}
                    tick={{ fontSize: 11, fill: "#98a6b3" }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => spreadViewMode === "z" ? v.toFixed(1) : Math.abs(v) >= 1000 ? `${(v/1000).toFixed(1)}k` : v.toFixed(2)}
                  />
                  <Tooltip
                    contentStyle={{ background: "rgba(8,22,36,0.95)", border: "1px solid rgba(99,230,190,0.18)", borderRadius: 10, fontSize: 13 }}
                    labelFormatter={(d) => <span style={{ color: "#63e6be", fontWeight: 700 }}>{formatDateLabel(d)}</span>}
                    formatter={(v, name) => [Number(v).toFixed(4), name === "z" ? "Z-score" : "Spread"]}
                  />
                  {/* Regime bands */}
                  {(() => {
                    if (!pairSpread.regime || !pairSpread.regime.length) return null;
                    const bands = [];
                    // If regime series starts after spread history, add an "unknown" band
                    const spreadStart = (pairSpread.spread && pairSpread.spread.length) ? pairSpread.spread[0].date : null;
                    const firstRegimeDate = pairSpread.regime[0].date;
                    if (spreadStart && firstRegimeDate && new Date(firstRegimeDate) > new Date(spreadStart)) {
                      bands.push({ start: spreadStart, end: firstRegimeDate, regime: null });
                    }

                    let cur = { ...pairSpread.regime[0], regime: Number(pairSpread.regime[0].regime) };
                    for (let i = 1; i < pairSpread.regime.length; i++) {
                      const nextRegime = Number(pairSpread.regime[i].regime);
                      if (nextRegime !== cur.regime) {
                        bands.push({ start: cur.date, end: pairSpread.regime[i - 1].date, regime: cur.regime });
                        cur = { ...pairSpread.regime[i], regime: nextRegime };
                      }
                    }
                    bands.push({ start: cur.date, end: pairSpread.regime[pairSpread.regime.length - 1].date, regime: cur.regime });

                    return bands.map((b, i) => {
                      if (b.regime == null) {
                        return (
                          <ReferenceArea
                            key={`unknown-${i}`}
                            x1={b.start}
                            x2={b.end}
                            fill="rgba(255,255,255,0.02)"
                            ifOverflow="hidden"
                          />
                        );
                      }
                      return (
                        <ReferenceArea
                          key={i}
                          x1={b.start}
                          x2={b.end}
                          fill={REGIME_META[b.regime]?.bg ?? "transparent"}
                          ifOverflow="hidden"
                        />
                      );
                    });
                  })()}
                  {spreadViewMode === "z" && (
                    <>
                      <ReferenceArea y1={1} y2={2}  fill="rgba(255,107,107,0.07)" ifOverflow="hidden" />
                      <ReferenceArea y1={-2} y2={-1} fill="rgba(255,107,107,0.07)" ifOverflow="hidden" />
                      <ReferenceLine y={0}   stroke="rgba(255,255,255,0.2)"  strokeDasharray="4 4" />
                      <ReferenceLine y={2}   stroke="rgba(255,107,107,0.55)" strokeDasharray="3 3" strokeWidth={1.5} label={{ value: "+2σ entry", position: "right", fontSize: 10, fill: "rgba(255,107,107,0.85)" }} />
                      <ReferenceLine y={-2}  stroke="rgba(255,107,107,0.55)" strokeDasharray="3 3" strokeWidth={1.5} label={{ value: "-2σ entry", position: "right", fontSize: 10, fill: "rgba(255,107,107,0.85)" }} />
                      <ReferenceLine y={0.5} stroke="rgba(99,230,190,0.35)"  strokeDasharray="2 5" label={{ value: "exit", position: "right", fontSize: 9, fill: "rgba(99,230,190,0.65)" }} />
                      <ReferenceLine y={-0.5} stroke="rgba(99,230,190,0.35)" strokeDasharray="2 5" label={{ value: "exit", position: "right", fontSize: 9, fill: "rgba(99,230,190,0.65)" }} />
                    </>
                  )}
                  <Area type="monotone" dataKey={spreadViewMode === "z" ? "z" : "spread"} stroke="#63e6be" strokeWidth={1.5} fill="url(#spreadGrad)" dot={false} isAnimationActive={false} />
                  <Brush dataKey="date" height={20} stroke="#63e6be" fill="rgba(99,230,190,0.06)" travellerWidth={8} tickFormatter={formatDateLabel} />
                </ComposedChart>
              </ResponsiveContainer>
              <RegimeLegend />
            </>
          )}
        </section>
      )}

      {/* ── Regime Comparison View (spec §4 — pairs per regime) ───────────── */}
      {Object.keys(pairsByRegime).length > 0 && (
        <section className="card full-width">
          <div className="chart-header">
            <h3>Regime Comparison View</h3>
            <p className="chart-subtitle">Average half-life of cointegrated pairs discovered in each distinct market regime.</p>
          </div>
          <ResponsiveContainer width="100%" height={88}>
            <BarChart
              data={Object.entries(pairsByRegime)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([regime, prs]) => ({
                  label: REGIME_META[Number(regime)]?.label?.split(" ")[0] ?? `R${regime}`,
                  count: prs.length,
                  regime: Number(regime),
                  avgHL: prs.reduce((s, p) => s + (Number(p.half_life_days) || 0), 0) / Math.max(1, prs.length),
                }))}
              margin={{ top: 4, right: 16, bottom: 0, left: 16 }}
            >
              <CartesianGrid strokeDasharray="2 2" stroke="rgba(255,255,255,0.04)" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#98a6b3" }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 11, fill: "#98a6b3" }} tickLine={false} axisLine={false} width={28} />
              <Tooltip
                contentStyle={{ background: "rgba(8,22,36,0.95)", border: "1px solid rgba(99,230,190,0.18)", borderRadius: 10, fontSize: 12 }}
                formatter={(v, name, props) => [
                  <span key="rc">{Number(v).toFixed(1)}d avg HL &nbsp;<span style={{ color: "#98a6b3" }}>{(props.payload?.count ?? 0)} pairs</span></span>,
                  "Avg HL",
                ]}
              />
              <Bar dataKey="avgHL" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                {Object.entries(pairsByRegime)
                  .sort(([a], [b]) => Number(a) - Number(b))
                  .map(([regime], i) => (
                    <Cell key={`rc-${i}`} fill={REGIME_META[Number(regime)]?.color ?? "#888"} opacity={0.82} />
                  ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="regime-comparison-grid">
            {Object.entries(pairsByRegime)
              .sort(([a], [b]) => Number(a) - Number(b))
              .map(([regime, pairs]) => (
                <div key={regime} className="regime-comparison-col">
                  <div
                    className="regime-comparison-header"
                    style={{ borderColor: REGIME_META[Number(regime)]?.color ?? "#888" }}
                  >
                    <span
                      className="regime-dot-inline"
                      style={{ "--dot-color": REGIME_META[Number(regime)]?.color ?? "#888" }}
                    />
                    <strong>{REGIME_META[Number(regime)]?.label ?? `Regime ${regime}`}</strong>
                    <span className="stat-badge" style={{ marginLeft: "auto" }}>{pairs.length}</span>
                  </div>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr><th>Pair</th><th>p-value</th><th>Half-Life</th><th>Corr</th></tr>
                      </thead>
                      <tbody>
                        {pairs.slice(0, 10).map((p) => (
                          <tr
                            key={`${regime}-${p.pair_id}`}
                            className="clickable-row"
                            onClick={() => onSelectPair(p.pair_id)}
                            title="Click to open Pair Explorer"
                          >
                            <td>{p.pair_id}</td>
                            <td style={{ color: Number(p.coint_pvalue) < 0.01 ? "#51cf66" : "#ffd166" }}>
                              {Number(p.coint_pvalue).toExponential(2)}
                            </td>
                            <td>{Number(p.half_life_days ?? 0).toFixed(1)}d</td>
                            <td>{Number(p.corr ?? 0).toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
          </div>
        </section>
      )}

      {/* ── Network View (spec — optional, nodes=assets, edges=strong pairs) ── */}
      {networkData && (
        <section className="card full-width">
          <div className="chart-header">
            <div>
              <h3>Network View</h3>
              <p className="chart-subtitle">
                Assets as nodes, strong cointegrated relationships as edges. Edge weight = 1 − p-value.
              </p>
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}>
              Filter regime:
              <select
                value={networkRegime}
                onChange={(e) => onNetworkRegimeChange(e.target.value)}
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, color: "#e0eaf0", padding: "4px 8px" }}
              >
                <option value="">All</option>
                {Object.keys(pairsByRegime || {}).sort((a,b) => Number(a) - Number(b)).map((r) => (
                  <option key={r} value={r}>{`${REGIME_META[Number(r)]?.label ?? `Regime ${r}`} (${(pairsByRegime?.[r]?.length) ?? 0})`}</option>
                ))}
              </select>
            </label>
          </div>
          {networkData.nodes.length > 0 ? (
            <NetworkGraph nodes={networkData.nodes} edges={networkData.edges} onSelectPair={onSelectPair} />
          ) : (
            <p className="chart-subtitle" style={{ paddingTop: 8 }}>
              No network data available for the current discovery run{networkRegime ? " and selected regime filter" : ""}.
            </p>
          )}
        </section>
      )}

      {/* ── Footer ── */}
      <footer className="status-footer">
        <span>API: {health?.ok ? "✓ Online" : "✗ Offline"}</span>
        <span>Has result: {String(health?.hasResult ?? false)}</span>
        {health?.lastError && <span className="footer-error">Last error: {health.lastError}</span>}
      </footer>
    </main>
  );
}