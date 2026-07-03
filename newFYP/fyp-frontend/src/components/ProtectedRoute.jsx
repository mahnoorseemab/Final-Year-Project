import React from "react";
import { Navigate } from "react-router-dom";

const ProtectedRoute = ({ children, allowedRoles }) => {
    const role = localStorage.getItem("role");
    
    if (!role) {
        return <Navigate to="/login" replace />;
    }
    
    if (allowedRoles && !allowedRoles.includes(role)) {
        if (role === "patient") return <Navigate to="/recommendation" replace />;
        return <Navigate to="/dashboard" replace />;
    }
    
    return children;
};

export default ProtectedRoute;