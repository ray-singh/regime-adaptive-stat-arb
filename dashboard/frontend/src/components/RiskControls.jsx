import React from "react";
import { REGIME_META } from "../controls";

export default function RiskControls({ controls, setControls, setError }) {
  return (
    <div className="card risk-controls" style={{ marginTop: 12 }}>
      <h3 style={{ marginTop: 0 }}>Risk Manager — Regime Maps &amp; Presets</h3>
      <p className="chart-subtitle">Tune per-regime caps. Values are validated before running.</p>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 8 }}>
        <label style={{ display: "flex", gap: 8, alignItems: "center" }} title="Quick presets adjust all regime caps at once">
          Preset
          <select onChange={(e) => {
            const p = e.target.value;
            if (p === "aggressive") {
              setControls({ ...controls,
                regimeLeverageCaps: { 0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0 },
                regimeMaxOpenPairs: { 0: 15, 1: 12, 2: 8, 3: 3 },
                regimePairNotionalPct: { 0: 0.25, 1: 0.20, 2: 0.12, 3: 0.06 },
                regimeTickerNotionalPct: { 0: 0.30, 1: 0.25, 2: 0.18, 3: 0.10 },
              });
            } else if (p === "conservative") {
              setControls({ ...controls,
                regimeLeverageCaps: { 0: 3.0, 1: 2.5, 2: 1.8, 3: 1.0 },
                regimeMaxOpenPairs: { 0: 8, 1: 6, 2: 4, 3: 1 },
                regimePairNotionalPct: { 0: 0.18, 1: 0.14, 2: 0.08, 3: 0.04 },
                regimeTickerNotionalPct: { 0: 0.22, 1: 0.18, 2: 0.12, 3: 0.06 },
              });
            } else {
              setControls({ ...controls,
                regimeLeverageCaps: { 0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0 },
                regimeMaxOpenPairs: { 0: 10, 1: 8, 2: 5, 3: 2 },
                regimePairNotionalPct: { 0: 0.20, 1: 0.15, 2: 0.10, 3: 0.05 },
                regimeTickerNotionalPct: { 0: 0.25, 1: 0.20, 2: 0.15, 3: 0.08 },
              });
            }
          }} defaultValue="balanced">
            <option value="balanced">Balanced (default)</option>
            <option value="aggressive">Aggressive</option>
            <option value="conservative">Conservative</option>
          </select>
        </label>
      </div>

      <table className="risk-table">
        <thead>
          <tr>
            <th>Regime</th>
            <th title="Max gross leverage allowed in this regime">Leverage cap</th>
            <th title="Max concurrent open pairs in this regime">Max open pairs</th>
            <th title="Max notional per pair leg (fraction of equity)">Pair notional %</th>
            <th title="Max notional per ticker (fraction of equity)">Ticker notional %</th>
          </tr>
        </thead>
        <tbody>
          {[0,1,2,3].map((r) => (
            <tr key={`r-${r}`}>
              <td><strong>{REGIME_META[r].label}</strong></td>
              <td>
                <input type="number" step="0.1" min="0.5" max="10" value={controls.regimeLeverageCaps?.[r] ?? ""}
                  onChange={(e) => setControls({ ...controls, regimeLeverageCaps: { ...controls.regimeLeverageCaps, [r]: Number(e.target.value) } })} />
              </td>
              <td>
                <input type="number" step="1" min="0" max="50" value={controls.regimeMaxOpenPairs?.[r] ?? ""}
                  onChange={(e) => setControls({ ...controls, regimeMaxOpenPairs: { ...controls.regimeMaxOpenPairs, [r]: Number(e.target.value) } })} />
              </td>
              <td>
                <input type="number" step="0.01" min="0" max="1" value={controls.regimePairNotionalPct?.[r] ?? ""}
                  onChange={(e) => setControls({ ...controls, regimePairNotionalPct: { ...controls.regimePairNotionalPct, [r]: Number(e.target.value) } })} />
              </td>
              <td>
                <input type="number" step="0.01" min="0" max="1" value={controls.regimeTickerNotionalPct?.[r] ?? ""}
                  onChange={(e) => setControls({ ...controls, regimeTickerNotionalPct: { ...controls.regimeTickerNotionalPct, [r]: Number(e.target.value) } })} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 8, fontSize: 13, color: "#98a6b3" }}>
        Tip: Values are applied before running the backtest. Ensure fractions are entered as decimals (e.g. 0.15 = 15%).
      </div>
    </div>
  );
}


/** Signal threshold controls — always shown (independent of risk manager toggle) */
export function SignalThresholdControls({ controls, setControls }) {
  return (
    <div className="card risk-controls" style={{ marginTop: 12 }}>
      <h3 style={{ marginTop: 0 }}>Signal Thresholds — Regime-Adaptive</h3>
      <p className="chart-subtitle">
        Per-regime entry / exit z-score thresholds and position size multipliers used by the Kalman + ensemble signal model.
        Bull markets allow tighter entries; crisis regimes block new positions.
      </p>

      <table className="risk-table">
        <thead>
          <tr>
            <th>Regime</th>
            <th title="Z-score magnitude required to enter a trade (lower = more trades)">Entry Z</th>
            <th title="Z-score below which the position is closed (convergence target)">Exit Z</th>
            <th title="Fraction of baseline position size in this regime (1.0 = full, 0 = blocked)">Position scale</th>
          </tr>
        </thead>
        <tbody>
          {[0,1,2,3].map((r) => (
            <tr key={`sig-${r}`}>
              <td>
                <span className="regime-dot-inline" style={{ "--dot-color": REGIME_META[r].color }} />
                <strong>{REGIME_META[r].label}</strong>
              </td>
              <td>
                <input
                  type="number" step="0.1" min="0.5" max="6"
                  value={controls.regimeEntryZ?.[r] ?? ""}
                  onChange={(e) => setControls({
                    ...controls,
                    regimeEntryZ: { ...controls.regimeEntryZ, [r]: Number(e.target.value) },
                  })}
                />
              </td>
              <td>
                <input
                  type="number" step="0.05" min="0" max="3"
                  value={controls.regimeExitZ?.[r] ?? ""}
                  onChange={(e) => setControls({
                    ...controls,
                    regimeExitZ: { ...controls.regimeExitZ, [r]: Number(e.target.value) },
                  })}
                />
              </td>
              <td>
                <input
                  type="number" step="0.05" min="0" max="1"
                  value={controls.regimePositionScale?.[r] ?? ""}
                  onChange={(e) => setControls({
                    ...controls,
                    regimePositionScale: { ...controls.regimePositionScale, [r]: Number(e.target.value) },
                  })}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 8, fontSize: 13, color: "#98a6b3" }}>
        Entry Z must be &gt; Exit Z. Position scale of 0 = no new entries in that regime.
      </div>
    </div>
  );
}
