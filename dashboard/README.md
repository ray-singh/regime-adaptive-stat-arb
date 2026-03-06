# Dashboard

React + Flask dashboard for the regime-adaptive stat-arb platform.

## Architecture

- `backend/app.py`: Flask API that runs backtests and serves metrics/trades/equity.
- `frontend/`: React + Vite UI with interactive controls and charts.

## API Endpoints

- `GET /api/health`
- `POST /api/run-backtest`
- `GET /api/summary`
- `GET /api/equity`
- `GET /api/trades?limit=200`
- `GET /api/pairs`

## Run Backend

From project root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python dashboard/backend/app.py
```

Backend starts on `http://localhost:5001`.

## Run Frontend

In a second terminal:

```bash
cd dashboard/frontend
npm install
npm run dev
```

Frontend starts on `http://localhost:5173` and proxies `/api` to Flask.

## Customization Controls in UI

- Initial capital
- Train split %
- Max pairs
- Pair re-selection interval
- Entry / Exit / Stop z-thresholds
- Risk manager on/off
- Pair re-selection on/off

Control presets are persisted in `localStorage`.
