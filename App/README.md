# AML  App

This folder contains the React dashboard and FastAPI backend for the money-mule detection project. The active frontend is `Frontend/src/AmlDashboard_v2.jsx`; the live sample action communicates with the backend route documented below.

## Requirements

- Node.js and npm
- Python 3.10 or later
- The project prediction file at `../data/graphs/ensemble_predictions.csv`

Install frontend dependencies once:

```powershell
cd D:\money-mule-detection\App\Frontend
npm install
```

Create and install a Python environment once:

```powershell
cd D:\money-mule-detection\App\Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
```

## Run Locally

Start the backend in one terminal:

```powershell
cd D:\money-mule-detection\App\Backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --app-dir . --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
cd D:\money-mule-detection\App\Frontend
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000/v1"
npm run dev -- --host 127.0.0.1
```

Open the Vite URL printed in the frontend terminal, normally `http://127.0.0.1:5173/`.

The V2 dashboard uses Tailwind CSS 3.4 and its theme is in `Frontend/src/aml-theme-v2.css`. `App.jsx` imports that theme and mounts `AmlDashboard_v2.jsx`.

## Live Sample Data Flow

The **Run live sample** button calls:

```text
GET http://127.0.0.1:8000/v1/analysis/live-random-account
```

The route is implemented in `Backend/api/v1/live_detection.py`. It reads `data/graphs/ensemble_predictions.csv` from the repository root, keeps rows with a valid account ID, test label, and all three finite model test probabilities, then randomly selects one row per request.

The backend recomputes the ensemble score from the stored probabilities using the validation-selected weights:

| Model | Weight |
|---|---:|
| GraphSAGE | 0.78 |
| GATv2 | 0.19 |
| GIN | 0.03 |

The predicted label uses a fixed threshold of `0.50`. The response includes the account ID and node index, ensemble score, threshold prediction, saved test label, individual model probabilities, source description, and scoring timestamp. The frontend displays those fields below Signals. Each request can return a different test account.

This is a fresh random sample from saved held-out test predictions. It does **not** load new transactions, run model training, or perform fresh inference on new data. The test label is dataset ground truth for evaluation, not a live operational finding.

The backend also exposes `GET /health` for a basic process health check. Live detection reads the saved CSV directly; no database or model-training process is required for this endpoint.

## Verify

Build the frontend:

```powershell
cd D:\money-mule-detection\App\Frontend
npm run build
```

Check backend health and request a sample:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/v1/analysis/live-random-account
```

If the sample endpoint returns `503`, confirm the prediction CSV exists at `D:\money-mule-detection\data\graphs\ensemble_predictions.csv` and contains held-out rows with all three test probability columns populated.
