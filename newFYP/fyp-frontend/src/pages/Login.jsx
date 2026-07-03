import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import "../styles/Login.css";

const Login = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch("http://localhost:8000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password
        })
      });

      const data = await response.json();
      setLoading(false);

      if (!response.ok) {
        setError(data.detail || "Invalid email or password.");
        return;
      }

      // Save user info
      localStorage.setItem('user_id', data.user_id);
      localStorage.setItem('email', data.email);
      localStorage.setItem('role', data.role);
      localStorage.setItem('user', data.full_name);

      // Redirect based on role
      if (data.role === "admin" || data.role === "doctor" || data.role === "auditor") {
        navigate("/dashboard");
      } else if (data.role === "patient") {
        navigate("/recommendation");
      } else {
        navigate("/dashboard");
      }

    } catch (err) {
      setLoading(false);
      setError("Server se connect nahi ho saka! Backend chala hua hai?");
      console.error(err);
    }
  };

  return (
    <div className="login-wrapper">

      {/* BG Shapes */}
      <div className="login-shapes">
        <div className="ls ls-1" />
        <div className="ls ls-2" />
        <div className="ls ls-3" />
        <div className="ls ls-4" />
      </div>

      {/* Split Card */}
      <motion.div
        className="login-box"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
      >
        {/* Left Panel */}
        <div className="login-left">
          <div className="login-logo">
            <div className="login-logo-icon">🛡️</div>
            <span>Medi<span className="login-ai">Guard AI</span></span>
          </div>
          <h2>Welcome Back!</h2>
          <p>Sign in to access your fraud monitoring dashboard and keep your healthcare system secure.</p>

          <div className="login-info-cards">
            <div className="login-info-card">
              <span>🔍</span>
              <div>
                <strong>Real-Time Detection</strong>
                <p>Monitor transactions live</p>
              </div>
            </div>
            <div className="login-info-card">
              <span>🛡️</span>
              <div>
                <strong>Fraud Prevention</strong>
                <p>AI-powered risk scoring</p>
              </div>
            </div>
          </div>

          <p className="login-back">
            <span onClick={() => navigate("/")}>← Back to Home</span>
          </p>
        </div>

        {/* Right Panel */}
        <div className="login-right">
          <div className="login-form-header">
            <h3>Sign In</h3>
            <p>Enter your credentials to continue</p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="login-field">
              <label>Email Address</label>
              <input
                type="email"
                name="email"
                placeholder="Enter your email"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>

            <div className="login-field">
              <label>Password</label>
              <input
                type="password"
                name="password"
                placeholder="Enter your password"
                value={formData.password}
                onChange={handleChange}
                required
              />
            </div>

            {error && (
              <motion.div
                className="login-error"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                ⚠ {error}
              </motion.div>
            )}

            <motion.button
              type="submit"
              className="login-btn"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              disabled={loading}
            >
              {loading ? <span className="login-spinner">Signing in...</span> : "Sign In →"}
            </motion.button>
          </form>

          <p className="login-register">
            Don't have an account?{" "}
            <span onClick={() => navigate("/register")}>Register here</span>
          </p>
        </div>

      </motion.div>
    </div>
  );
};

export default Login;