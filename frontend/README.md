# AML Sentinel — Money Mule Detection Frontend

Premium React/Vite frontend for a graph-based money mule detection major project.

## Run
1. Install Node.js 20+
2. Open this folder in VS Code
3. Run:
   npm install
   npm run dev
4. Open the localhost URL shown by Vite.

## Pages
- Command Center
- Network Explorer (Cytoscape graph)
- Suspicious Accounts
- AI Risk Analysis (GCN/GAT/GraphSAGE ensemble)
- Explainable AI
- Priority Intelligence
- Alerts & Cases

## Backend integration
Mock data lives in `src/data/mock.ts`.
Set `VITE_API_URL=http://localhost:8000/api` in `.env` and connect calls through `src/services/api.ts`.

Recommended FastAPI contract:
- GET /api/dashboard
- GET /api/accounts
- GET /api/accounts/{id}
- GET /api/network/{id}
- GET /api/predictions/{id}
- GET /api/explain/{id}
- GET /api/rankings

Important: UI numbers and account IDs are demonstration mock data, not repository results.
