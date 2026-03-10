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
    </section>
  );
}
