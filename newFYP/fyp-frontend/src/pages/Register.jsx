import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import "../styles/Register.css";

const Register = () => {
  const navigate = useNavigate();

  const [step, setStep] = useState("form");
  const [formData, setFormData] = useState({
    fullName: "",
    email: "",
    password: "",
    confirmPassword: "",
    role: "",
    pmdcNumber: ""
  });
  const [otp, setOtp] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const isDoctor = formData.role === "doctor";

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError("");
  };

  // Step 1: Submit form → send OTP
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match!");
      return;
    }

    if (isDoctor && !formData.pmdcNumber.trim()) {
      setError("PMDC Number is required for doctors.");
      return;
    }

    setLoading(true);
    try {
      const body = {
        full_name: formData.fullName,
        email:     formData.email,
        password:  formData.password,
        role:      formData.role,
      };

      if (isDoctor) {
        body.pmdc_number = formData.pmdcNumber;
      }

      const response = await fetch("http://localhost:8000/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });

      const data = await response.json();
      setLoading(false);

      if (!response.ok) {
        setError(data.detail || "Registration failed.");
        return;
      }

      setStep("otp");

    } catch (err) {
      setLoading(false);
      setError("Server se connect nahi ho saka! Backend chala hua hai?");
      console.error(err);
    }
  };

  // Step 2: Submit OTP → create account
  const handleOtpSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (otp.length !== 6) {
      setError("Please enter the 6-digit OTP.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/verify-otp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: formData.email,
          otp:   otp
        })
      });

      const data = await response.json();
      setLoading(false);

      if (!response.ok) {
        setError(data.detail || "OTP verification failed.");
        return;
      }

      localStorage.setItem('user_id', data.user_id);
      localStorage.setItem('email',   data.email);
      localStorage.setItem('role',    data.role);
      localStorage.setItem('user',    data.full_name);

      alert("Account created! Welcome " + data.full_name);

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
    <div className="reg-wrapper">

      {/* BG Shapes */}
      <div className="reg-shapes">
        <div className="rs rs-1" />
        <div className="rs rs-2" />
        <div className="rs rs-3" />
        <div className="rs rs-4" />
      </div>

      <motion.div
        className="reg-box"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
      >
        {/* Left Panel */}
        <div className="reg-left">
          <div className="reg-logo">
            <div className="reg-logo-icon">🛡️</div>
            <span>Medi<span className="reg-ai">Guard AI</span></span>
          </div>

          <h2>{step === "form" ? "Join MediGuard AI" : "Verify Your Email"}</h2>
          <p>
            {step === "form"
              ? "Create your account to start monitoring healthcare transactions and detecting fraud in real-time."
              : `We sent a 6-digit verification code to ${formData.email}. Please check your inbox.`}
          </p>

          <div className="reg-info-cards">
            <div className="reg-info-card">
              <span>🔍</span>
              <div>
                <strong>Anomaly Detection</strong>
                <p>Catch fraud before it happens</p>
              </div>
            </div>
            <div className="reg-info-card">
              <span>📊</span>
              <div>
                <strong>Risk Scoring</strong>
                <p>Every transaction analyzed</p>
              </div>
            </div>
            <div className="reg-info-card">
              <span>🛡️</span>
              <div>
                <strong>Secure Platform</strong>
                <p>Role-based access control</p>
              </div>
            </div>
          </div>

          <p className="reg-back">
            <span onClick={() => step === "otp" ? setStep("form") : navigate("/")}>
              {step === "otp" ? "← Back to Form" : "← Back to Home"}
            </span>
          </p>
        </div>

        {/* Right Panel */}
        <div className="reg-right">

          {/* STEP 1: Registration Form */}
          {step === "form" && (
            <>
              <div className="reg-form-header">
                <h3>Create Account</h3>
                <p>Fill in your details to get started</p>
              </div>

              <form onSubmit={handleSubmit}>
                <div className="reg-field">
                  <label>Full Name</label>
                  <input
                    type="text"
                    name="fullName"
                    placeholder="Enter your full name"
                    onChange={handleChange}
                    required
                  />
                </div>

                <div className="reg-field">
                  <label>Email Address</label>
                  <input
                    type="email"
                    name="email"
                    placeholder="Enter your email"
                    onChange={handleChange}
                    required
                  />
                </div>

                <div className="reg-row">
                  <div className="reg-field">
                    <label>Password</label>
                    <input
                      type="password"
                      name="password"
                      placeholder="Create password"
                      onChange={handleChange}
                      required
                    />
                  </div>
                  <div className="reg-field">
                    <label>Confirm Password</label>
                    <input
                      type="password"
                      name="confirmPassword"
                      placeholder="Confirm password"
                      onChange={handleChange}
                      required
                    />
                  </div>
                </div>

                <div className="reg-field">
                  <label>Select Role</label>
                  <select name="role" onChange={handleChange} required defaultValue="">
                    <option value="" disabled>Choose your role</option>
                    <option value="admin">Admin</option>
                    <option value="doctor">Doctor</option>
                    <option value="staff">Billing Staff</option>
                    <option value="patient">Patient</option>
                  </select>
                </div>

                {/* PMDC Number — sirf Doctor select karne par show hoga */}
                {isDoctor && (
                  <motion.div
                    className="reg-field"
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <label>
                      PMDC Registration Number
                      <span className="reg-pmdc-badge">Doctors Only</span>
                    </label>
                    <input
                      type="text"
                      name="pmdcNumber"
                      placeholder="e.g. PMDC-12345"
                      onChange={handleChange}
                      value={formData.pmdcNumber}
                      required
                    />
                  </motion.div>
                )}

                {error && (
                  <motion.div
                    className="login-error"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    ⚠ {error}
                  </motion.div>
                )}

                <motion.button
                  type="submit"
                  className="reg-btn"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  disabled={loading}
                >
                  {loading ? "Sending OTP..." : "Continue →"}
                </motion.button>
              </form>

              <p className="reg-login">
                Already have an account?{" "}
                <span onClick={() => navigate("/login")}>Login here</span>
              </p>
            </>
          )}

          {/* STEP 2: OTP Verification */}
          {step === "otp" && (
            <>
              <div className="reg-form-header">
                <h3>Enter OTP</h3>
                <p>Check your email <strong>{formData.email}</strong> for the 6-digit code</p>
              </div>

              <form onSubmit={handleOtpSubmit}>
                <div className="reg-field">
                  <label>Verification Code</label>
                  <input
                    type="text"
                    placeholder="Enter 6-digit OTP"
                    maxLength={6}
                    value={otp}
                    onChange={(e) => {
                      setOtp(e.target.value);
                      setError("");
                    }}
                    required
                    style={{ letterSpacing: "0.3em", fontSize: "1.2rem", textAlign: "center" }}
                  />
                </div>

                {error && (
                  <motion.div
                    className="login-error"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    ⚠ {error}
                  </motion.div>
                )}

                <motion.button
                  type="submit"
                  className="reg-btn"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  disabled={loading}
                >
                  {loading ? "Verifying..." : "Verify & Create Account →"}
                </motion.button>
              </form>

              <p className="reg-login">
                Didn't receive the code?{" "}
                <span onClick={() => setStep("form")}>Go back and try again</span>
              </p>
            </>
          )}

        </div>
      </motion.div>
    </div>
  );
};

export default Register;