import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import "../styles/Navbar.css";

const Navbar = () => {
  const navigate = useNavigate();
  const role = localStorage.getItem("role");
  const user = localStorage.getItem("user");

  const handleLogout = () => {
    localStorage.removeItem("role");
    localStorage.removeItem("user");
    localStorage.removeItem("email");
    localStorage.removeItem("password");
    navigate("/login");
  };

  return (
    <aside className="navbar">
      <div className="nav-logo">
        <div className="nav-logo-icon">🛡️</div>
        <span>Medi<span className="nav-logo-ai">Guard AI</span></span>
      </div>

      <nav className="nav-links">
        <div className="nav-section-label">Main Menu</div>

        <NavLink to="/dashboard" className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
          <span className="nav-icon">🏠</span> Dashboard
        </NavLink>

        <NavLink to="/all-transactions" className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
          <span className="nav-icon">📄</span> Transactions
        </NavLink>

        {role === "staff" && (
          <NavLink to="/transaction" className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
            <span className="nav-icon">➕</span> New Transaction
          </NavLink>
        )}
      </nav>

      <div className="nav-bottom">
        <div className="nav-user">
          <div className="nav-avatar">{user ? user.charAt(0).toUpperCase() : "U"}</div>
          <div className="nav-user-info">
            <span className="nav-user-name">{user || "User"}</span>
            <span className="nav-user-role">{role || "Guest"}</span>
          </div>
        </div>
        <button className="nav-logout-btn" onClick={handleLogout}>
          🚪 Logout
        </button>
      </div>
    </aside>
  );
};

export default Navbar;