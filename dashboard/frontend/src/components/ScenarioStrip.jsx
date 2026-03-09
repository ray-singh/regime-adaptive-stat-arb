import React from "react";
import { SCENARIO_PRESETS } from "../controls";

export default function ScenarioStrip({ activeId, onSelect }) {
  return (
    <section className="scenario-strip card">
      <div className="scenario-strip-top">
        <div>
          <p className="eyebrow" style={{ marginBottom: 2 }}>Story Mode</p>
          <p className="scenario-strip-sub">Choose a pre-configured scenario to explore a particular market narrative, or tune the controls below manually.</p>
        </div>
      </div>
      <div className="scenario-cards">
        {SCENARIO_PRESETS.map((s) => (
          <button
            key={s.id}
            className={`scenario-card${activeId === s.id ? " active" : ""}`}
            style={{ "--sc-color": s.color }}
            onClick={() => onSelect(activeId === s.id ? null : s)}
          >
            <span className="scenario-icon">{s.icon}</span>
            <span className="scenario-name">{s.label}</span>
            <span className="scenario-tagline">{s.tagline}</span>
          </button>
        ))}
      </div>
      {activeId && (() => {
        const s = SCENARIO_PRESETS.find((p) => p.id === activeId);
        return s ? (
          <div className="scenario-description" style={{ borderLeftColor: s.color }}>
            <strong style={{ color: s.color }}>{s.icon} {s.label} — </strong>{s.description}
          </div>
        ) : null;
      })()}
    </section>
  );
}
