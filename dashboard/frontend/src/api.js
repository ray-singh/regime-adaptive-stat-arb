const API_BASE = import.meta.env.DEV ? "http://localhost:5001" : "";

export async function getHealth() {
  const response = await fetch(`${API_BASE}/api/health`);
  return response.json();
}

export async function runBacktest(payload) {
  const response = await fetch(`${API_BASE}/api/run-backtest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload ?? {}),
  });
  return response.json();
}

export async function getSummary() {
  const response = await fetch(`${API_BASE}/api/summary`);
  return response.json();
}

export async function getEquity() {
  const response = await fetch(`${API_BASE}/api/equity`);
  return response.json();
}

export async function getTrades(limit = 200) {
  const response = await fetch(`${API_BASE}/api/trades?limit=${limit}`);
  return response.json();
}

export async function getPairs() {
  const response = await fetch(`${API_BASE}/api/pairs`);
  return response.json();
}

export async function getRegime() {
  const response = await fetch(`${API_BASE}/api/regime`);
  return response.json();
}

export async function getRegimePerformance() {
  const response = await fetch(`${API_BASE}/api/regime-performance`);
  return response.json();
}

export async function getDrawdown() {
  const response = await fetch(`${API_BASE}/api/drawdown`);
  return response.json();
}

export async function getRollingSharpe(window = 60) {
  const response = await fetch(`${API_BASE}/api/rolling-sharpe?window=${window}`);
  return response.json();
}
