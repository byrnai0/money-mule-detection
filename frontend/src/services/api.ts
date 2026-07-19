import axios from "axios";
export const api=axios.create({baseURL:import.meta.env.VITE_API_URL||"http://localhost:8000/api"});
// Replace mock data with these endpoints when FastAPI is ready:
// GET /dashboard, /accounts, /accounts/:id, /network/:id, /predictions/:id, /explain/:id, /rankings
