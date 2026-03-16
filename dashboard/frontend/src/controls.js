// Shared constants for the dashboard UI
export const REGIME_META = {
  0: { label: "Bull / Low-Vol",       color: "#51cf66", bg: "rgba(81,207,102,0.18)"  },
  1: { label: "Neutral / Mid-Vol",    color: "#ffd166", bg: "rgba(255,209,102,0.14)" },
  2: { label: "Bear / High-Vol",      color: "#ff8fa3", bg: "rgba(255,143,163,0.22)" },
  3: { label: "Crisis / Extreme-Vol", color: "#ff6b6b", bg: "rgba(255,107,107,0.30)" },
};

export const DEFAULT_CONTROLS = {
  initialCapital: 1000000,
  trainPct: 0.5,
  maxPairs: 10,
  reselectionInterval: 63,
  reselectionEnabled: true,
  useRisk: true,
  useMacroTickers: false,
  // Minimum number of bars required to run discovery in a regime
  minRegimeBars: 126,
  universe: "top200",
  entryZ: 2.0,
  exitZ: 0.5,
  stopZ: 3.5,
  nStates: 4,
  // Risk manager regime maps
  regimeLeverageCaps:     { 0: 4.0,  1: 3.0,  2: 2.0,  3: 1.0  },
  regimeMaxOpenPairs:     { 0: 10,   1: 8,    2: 5,    3: 2    },
  regimePairNotionalPct:  { 0: 0.2,  1: 0.15, 2: 0.10, 3: 0.05 },
  regimeTickerNotionalPct:{ 0: 0.25, 1: 0.20, 2: 0.15, 3: 0.08 },
  // Signal / strategy regime-adaptive thresholds
  regimeEntryZ:           { 0: 1.5,  1: 2.0,  2: 2.5,  3: 4.0  },
  regimeExitZ:            { 0: 0.3,  1: 0.5,  2: 0.8,  3: 1.0  },
  regimePositionScale:    { 0: 1.0,  1: 0.75, 2: 0.5,  3: 0.1  },
};

export const SCENARIO_PRESETS = [
  {
    id: "stress_test",
    icon: "🔥",
    label: "Stress Test",
    tagline: "Can the strategy survive a crisis?",
    description:
      "Raises the entry bar, tightens stop losses, caps the number of pairs at 5, and layers in VIX + gold macro signals so the regime model 'knows' about fear. A good answer to: what happens when markets break?",
    color: "#ff6b6b",
    controls: {
      entryZ: 2.5, exitZ: 0.8, stopZ: 3.0, nStates: 4, maxPairs: 5, useRisk: true, useMacroTickers: true,
      regimeEntryZ: { 0: 2.0, 1: 2.5, 2: 3.0, 3: 5.0 },
      regimeExitZ:  { 0: 0.5, 1: 0.8, 2: 1.0, 3: 1.5 },
      regimePositionScale: { 0: 0.8, 1: 0.6, 2: 0.3, 3: 0.0 },
    },
  },
  {
    id: "regime_replay",
    icon: "🔄",
    label: "Regime Shift Replay",
    tagline: "Watch the strategy adapt as markets rotate.",
    description:
      "Enables quarterly pair re-selection and VIX+gold enrichment so the strategy automatically rebuilds its pair universe each time the regime model detects a structural shift. Best for understanding the core adaptive mechanism.",
    color: "#ffd166",
    controls: {
      nStates: 4, useMacroTickers: true, reselectionEnabled: true, reselectionInterval: 63, maxPairs: 8, useRisk: true,
      regimeEntryZ: { 0: 1.5, 1: 2.0, 2: 2.5, 3: 4.0 },
      regimeExitZ:  { 0: 0.3, 1: 0.5, 2: 0.8, 3: 1.0 },
      regimePositionScale: { 0: 1.0, 1: 0.75, 2: 0.5, 3: 0.1 },
    },
  },
  {
    id: "bull_run",
    icon: "🚀",
    label: "Bull Market Run",
    tagline: "Maximum alpha in calm conditions.",
    description:
      "Aggressive entry threshold, more pairs, and a wide stop to extract the most spread convergence during quiet, low-volatility bull markets. Risk manager is off — full speed ahead.",
    color: "#51cf66",
    controls: {
      entryZ: 1.5, exitZ: 0.3, stopZ: 4.0, maxPairs: 15, useRisk: false, useMacroTickers: false, reselectionEnabled: false,
      regimeEntryZ: { 0: 1.2, 1: 1.5, 2: 2.0, 3: 3.5 },
      regimeExitZ:  { 0: 0.2, 1: 0.3, 2: 0.5, 3: 0.8 },
      regimePositionScale: { 0: 1.0, 1: 1.0, 2: 0.7, 3: 0.2 },
    },
  },
  {
    id: "conservative",
    icon: "🛡",
    label: "Conservative Income",
    tagline: "Capital preservation first.",
    description:
      "High entry bar, strict risk manager, only 4 pairs. Trade less, lose less. Optimised for resilience rather than raw returns — the question is: what is the cost of safety?",
    color: "#63e6be",
    controls: {
      entryZ: 2.5, exitZ: 1.0, stopZ: 3.0, maxPairs: 4, useRisk: true, useMacroTickers: false, reselectionEnabled: true,
      regimeEntryZ: { 0: 2.0, 1: 2.5, 2: 3.0, 3: 5.0 },
      regimeExitZ:  { 0: 0.5, 1: 0.8, 2: 1.2, 3: 2.0 },
      regimePositionScale: { 0: 0.7, 1: 0.5, 2: 0.3, 3: 0.0 },
    },
  },
];

export default {
  REGIME_META,
  DEFAULT_CONTROLS,
  SCENARIO_PRESETS,
};
