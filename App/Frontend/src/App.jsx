
// import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
//  import SentinelLayout from './layouts/SentinelLayout';
//  import Index from './Index';

//  import CommandCenter from './pages/CommandCenter';
//  import NetworkExplorer from './pages/NetworkExplorer';
//  import SuspiciousAccounts from './pages/SuspiciousAccounts';
//  import RiskAnalysis from './pages/RiskAnalysis';
//  import AlertsCases from './pages/AlertsCases';
//  import Investigations from './pages/Investigations';
//  import Reports from './pages/Reports';

//  function ProtectedRoute({ children }) {
//    const token = localStorage.getItem('token');

//    if (!token) {
//      return <Navigate to="/" replace />;
//    }

//    return children;
//  }

//  function AppRoutes() {
//    return (
//      <Routes>
//        {/* Login/Register route */}
//        <Route path="/" element={<Index />} />

//        <Route
//          element={
//            <ProtectedRoute>
//              <SentinelLayout />
//            </ProtectedRoute>
//          }
//        >
//          <Route path="/dashboard" element={<CommandCenter />} />
//          <Route path="/network" element={<NetworkExplorer />} />
//          <Route path="/suspicious" element={<SuspiciousAccounts />} />
//          <Route path="/analysis" element={<RiskAnalysis />} />
//          <Route path="/explainability" element={<Navigate to="/analysis" replace />} />
//          <Route path="/priority" element={<Navigate to="/dashboard" replace />} />
//          <Route path="/alerts" element={<AlertsCases />} />
//          <Route path="/investigations" element={<Investigations />} />
//          <Route path="/investigations/:id" element={<Investigations />} />
//          <Route path="/reports" element={<Reports />} />
//        </Route>

//        <Route path="*" element={<Navigate to="/dashboard" replace />} />
//      </Routes>
//    );
//  }

//  export default function App() {
//    return (
//      <BrowserRouter>
//        <AppRoutes />
//      </BrowserRouter>
//    );
//  }

import AmlDashboard from './AmlDashboard_v2.jsx';
import './aml-theme-v2.css';

export default function App() {
  return <AmlDashboard />;
}
