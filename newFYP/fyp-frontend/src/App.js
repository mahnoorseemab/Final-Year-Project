import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";

import LandingPage from "./pages/LandingPage";
import Register from "./pages/Register";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Transaction from "./pages/Transaction";
import AllTransactions from "./pages/AllTransactions";
import FraudDetails from "./pages/FraudDetails";
import ViewReport from "./pages/ViewReport";
import Recommendation from "./pages/Recommendation";
import ProtectedRoute from "./components/ProtectedRoute";

const App = () => {
  return (
    <Router>
      <Routes>

        {/* Public Routes */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />

        {/* Admin + Doctor + Staff — Dashboard */}
        <Route path="/dashboard" element={
          <ProtectedRoute allowedRoles={["admin", "doctor", "staff"]}>
            <Dashboard />
          </ProtectedRoute>
        } />

        {/* Staff only — Transaction Form */}
        <Route path="/transaction" element={
          <ProtectedRoute allowedRoles={["staff"]}>
            <Transaction />
          </ProtectedRoute>
        } />

        {/* Admin + Doctor + Staff — All Transactions */}
        <Route path="/all-transactions" element={
          <ProtectedRoute allowedRoles={["admin", "doctor", "staff"]}>
            <AllTransactions />
          </ProtectedRoute>
        } />

        {/* Admin + Doctor + Staff — Fraud Details */}
        <Route path="/fraud-details/:id" element={
          <ProtectedRoute allowedRoles={["admin", "doctor", "staff"]}>
            <FraudDetails />
          </ProtectedRoute>
        } />

        {/* Admin + Doctor + Staff — View Report */}
        <Route path="/view-report/:id" element={
          <ProtectedRoute allowedRoles={["admin", "doctor", "staff"]}>
            <ViewReport />
          </ProtectedRoute>
        } />

        {/* Patient only — Recommendation */}
        <Route path="/recommendation" element={
          <ProtectedRoute allowedRoles={["patient"]}>
            <Recommendation />
          </ProtectedRoute>
        } />

        {/* Unknown route */}
        <Route path="*" element={<Navigate to="/" replace />} />

      </Routes>
    </Router>
  );
};

export default App;