#!/usr/bin/env python3
import time, json
import requests

BASE = "http://localhost:5001"
print("GET /api/health ->", requests.get(f"{BASE}/api/health").text)

payload = {
    "tickers": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
        "V", "UNH", "ADBE", "AMD", "AVGO", "CRM", "ORCL",
    ],
    "nStates": 4,
    "minRegimeBars": 5,
}
print("POST /api/discover ->", requests.post(f"{BASE}/api/discover", json=payload).text)

for i in range(60):
    s = requests.get(f"{BASE}/api/discovery/status").json()
    print(f"poll {i+1}:", json.dumps(s))
    if s.get("hasResult"):
        print("Discovery finished with result")
        break
    if not s.get("running"):
        print("Discovery stopped without result")
        break
    time.sleep(2)

print("\n--- /api/pairs/by-regime (first 4000 chars) ---")
print(requests.get(f"{BASE}/api/pairs/by-regime").text[:4000])
print("\n--- /api/pairs/ranked (first 4000 chars) ---")
print(requests.get(f"{BASE}/api/pairs/ranked").text[:4000])
