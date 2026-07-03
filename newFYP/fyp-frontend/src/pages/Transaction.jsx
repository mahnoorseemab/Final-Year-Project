import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import Navbar from "../components/Navbar";
import "../styles/Transaction.css";

const SPECIALITY_OPTIONS = [
  { label: "Anesthesiology", value: "Anesthesiologist" },
  { label: "Blood Bank", value: "Blood Bank Staff" },
  { label: "Cardiology", value: "Cardiologist" },
  { label: "Critical Care", value: "Critical Care" },
  { label: "Dermatology", value: "Dermatologist" },
  { label: "ENT", value: "ENT Specialist" },
  { label: "Emergency", value: "Emergency" },
  { label: "Endocrinology", value: "Endocrinologist" },
  { label: "Gastroenterology", value: "Gastroenterogist" },
  { label: "General Surgery", value: "General Surgeon" },
  { label: "Infectious Diseases", value: "Infectious Diseases" },
  { label: "Medical Specialist", value: "Medical Specialist" },
  { label: "Neurology", value: "Neurologist" },
  { label: "Nephrology", value: "Nephrologist" },
  { label: "Neurosurgery", value: "Neurosurgery" },
  { label: "Nutritionist", value: "Nutritionist" },
  { label: "Gynecology", value: "OB/Gyne" },
  { label: "Ophthalmology", value: "Opthalmologist" },
  { label: "Orthopedics", value: "Orthopedic" },
  { label: "Pain Management", value: "Pain Management" },
  { label: "Pediatric Cardiology", value: "Pediatric Cardiologist" },
  { label: "Pediatric Surgery", value: "Pediatric Surgeon" },
  { label: "Pediatrics", value: "Pediatrician" },
  { label: "Physical Rehab", value: "Physical Med & Rehabilitation" },
  { label: "Psychiatry", value: "Psychiatrist" },
  { label: "Pulmonology", value: "Pulmonologist" },
  { label: "Rheumatology", value: "Rheumatologist" },
  { label: "Cardiac Surgery", value: "Surgery - Cardiac" },
  { label: "Plastic Surgery", value: "Surgery - Plastic" },
  { label: "Urology", value: "Urologist" },
];

// ── Reusable Searchable Dropdown ─────────────────────────────
const SearchableDropdown = ({ options, value, onChange, placeholder, disabled }) => {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef(null);

  const filtered = options.filter(opt =>
    opt.toLowerCase().includes(search.toLowerCase())
  );

  useEffect(() => {
    const handleClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) {
        setOpen(false);
        setSearch("");
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const handleSelect = (opt) => {
    onChange(opt);
    setOpen(false);
    setSearch("");
  };

  return (
    <div ref={ref} style={{ position: "relative", width: "100%" }}>
      <div
        onClick={() => { if (!disabled) setOpen(o => !o); }}
        style={{
          width: "100%",
          padding: "10px 12px",
          borderRadius: "8px",
          border: "1px solid #e2e8f0",
          fontSize: "14px",
          background: disabled ? "#f8fafc" : "white",
          color: value ? "#1e293b" : "#94a3b8",
          cursor: disabled ? "not-allowed" : "pointer",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          boxSizing: "border-box",
          fontFamily: "Inter, sans-serif",
        }}
      >
        <span>{value || placeholder}</span>
        <span style={{ color: "#94a3b8", fontSize: "12px" }}>▼</span>
      </div>

      {open && (
        <div style={{
          position: "absolute",
          top: "calc(100% + 4px)",
          left: 0,
          right: 0,
          background: "white",
          border: "1px solid #e2e8f0",
          borderRadius: "10px",
          boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
          zIndex: 999,
          maxHeight: "220px",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}>
          {/* Search input */}
          <div style={{ padding: "8px", borderBottom: "1px solid #f1f5f9" }}>
            <input
              autoFocus
              type="text"
              placeholder="Type to search..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{
                width: "100%",
                padding: "7px 10px",
                borderRadius: "6px",
                border: "1px solid #e2e8f0",
                fontSize: "13px",
                outline: "none",
                fontFamily: "Inter, sans-serif",
                color: "#1e293b",
                boxSizing: "border-box",
              }}
            />
          </div>

          {/* Options list */}
          <div style={{ overflowY: "auto", maxHeight: "160px" }}>
            {filtered.length > 0 ? filtered.map((opt, i) => (
              <div
                key={i}
                onMouseDown={() => handleSelect(opt)}
                style={{
                  padding: "9px 14px",
                  fontSize: "14px",
                  color: opt === value ? "#0ea5e9" : "#1e293b",
                  background: opt === value ? "#f0f9ff" : "white",
                  cursor: "pointer",
                  fontFamily: "Inter, sans-serif",
                  transition: "background 0.15s",
                }}
                onMouseEnter={e => e.currentTarget.style.background = "#f8fafc"}
                onMouseLeave={e => e.currentTarget.style.background = opt === value ? "#f0f9ff" : "white"}
              >
                {opt}
              </div>
            )) : (
              <div style={{ padding: "10px 14px", color: "#94a3b8", fontSize: "13px" }}>
                No match — your custom value will be used
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// ── Speciality Searchable Dropdown (value+label) ─────────────
const SpecialityDropdown = ({ value, onChange, placeholder }) => {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef(null);

  const filtered = SPECIALITY_OPTIONS.filter(opt =>
    opt.label.toLowerCase().includes(search.toLowerCase()) ||
    opt.value.toLowerCase().includes(search.toLowerCase())
  );

  useEffect(() => {
    const handleClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) {
        setOpen(false);
        setSearch("");
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const selectedLabel = SPECIALITY_OPTIONS.find(o => o.value === value)?.label || value;

  const handleSelect = (opt) => {
    onChange(opt.value);
    setOpen(false);
    setSearch("");
  };

  return (
    <div ref={ref} style={{ position: "relative", width: "100%" }}>
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%",
          padding: "10px 12px",
          borderRadius: "8px",
          border: "1px solid #e2e8f0",
          fontSize: "14px",
          background: "white",
          color: value ? "#1e293b" : "#94a3b8",
          cursor: "pointer",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          boxSizing: "border-box",
          fontFamily: "Inter, sans-serif",
        }}
      >
        <span>{value ? selectedLabel : placeholder}</span>
        <span style={{ color: "#94a3b8", fontSize: "12px" }}>▼</span>
      </div>

      {open && (
        <div style={{
          position: "absolute",
          top: "calc(100% + 4px)",
          left: 0,
          right: 0,
          background: "white",
          border: "1px solid #e2e8f0",
          borderRadius: "10px",
          boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
          zIndex: 999,
          maxHeight: "220px",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}>
          <div style={{ padding: "8px", borderBottom: "1px solid #f1f5f9" }}>
            <input
              autoFocus
              type="text"
              placeholder="Type to search..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{
                width: "100%",
                padding: "7px 10px",
                borderRadius: "6px",
                border: "1px solid #e2e8f0",
                fontSize: "13px",
                outline: "none",
                fontFamily: "Inter, sans-serif",
                color: "#1e293b",
                boxSizing: "border-box",
              }}
            />
          </div>
          <div style={{ overflowY: "auto", maxHeight: "160px" }}>
            {filtered.length > 0 ? filtered.map((opt, i) => (
              <div
                key={i}
                onMouseDown={() => handleSelect(opt)}
                style={{
                  padding: "9px 14px",
                  fontSize: "14px",
                  color: opt.value === value ? "#0ea5e9" : "#1e293b",
                  background: opt.value === value ? "#f0f9ff" : "white",
                  cursor: "pointer",
                  fontFamily: "Inter, sans-serif",
                }}
                onMouseEnter={e => e.currentTarget.style.background = "#f8fafc"}
                onMouseLeave={e => e.currentTarget.style.background = opt.value === value ? "#f0f9ff" : "white"}
              >
                <span style={{ fontWeight: 500 }}>{opt.value}</span>
                <span style={{ color: "#94a3b8", fontSize: "12px", marginLeft: "6px" }}>
                  {opt.label}
                </span>
              </div>
            )) : (
              <div style={{ padding: "10px 14px", color: "#94a3b8", fontSize: "13px" }}>
                No match found
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const Transaction = () => {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    patient_id: "",
    doctor_id: "",
    speciality: "",
    diagnosis: "",
    service_description: "",
  });

  const [submitted, setSubmitted] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [services, setServices] = useState([]);
  const [servicesLoading, setServicesLoading] = useState(false);

  const fetchServices = async (specialityValue) => {
    if (!specialityValue) return;
    setServicesLoading(true);
    setServices([]);
    try {
      const res = await fetch(
        `http://localhost:8000/services-by-specialty?specialty=${encodeURIComponent(specialityValue)}`
      );
      const data = await res.json();
      setServices(data.services || []);
    } catch (err) {
      console.error("Services fetch failed:", err);
    } finally {
      setServicesLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSpecialityChange = (value) => {
    setFormData(prev => ({ ...prev, speciality: value, service_description: "" }));
    fetchServices(value);
  };

  const handleServiceChange = (value) => {
    setFormData(prev => ({ ...prev, service_description: value }));
  };

  const getStatusFromRisk = (risk) => {
    if (risk === "HIGH RISK") return "Fraud";
    if (risk === "MEDIUM RISK") return "Suspicious";
    return "Legit";
  };

  const getRiskColor = (risk) => {
    if (risk === "HIGH RISK" || risk === "Fraud") return "#ef4444";
    if (risk === "MEDIUM RISK" || risk === "Suspicious") return "#d97706";
    return "#0d9488";
  };

  const getStatusIcon = (risk) => {
    if (risk === "HIGH RISK" || risk === "Fraud") return "🚨";
    if (risk === "MEDIUM RISK" || risk === "Suspicious") return "⚠️";
    return "✅";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/transaction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          PATIENT_ID: parseInt(formData.patient_id),
          DOCTOR_ID: formData.doctor_id,
          SPECIALITY_NAME: formData.speciality,
          DIAGNOSIS: formData.diagnosis,
          SERVICE_DESCRIPTION: formData.service_description,
        })
      });

      const data = await response.json();
      setLoading(false);

      if (!response.ok) {
        alert("Error: " + data.detail);
        return;
      }

      const reportKey = `report_${data.transaction_id}`;
      localStorage.setItem(reportKey, JSON.stringify(data));

      const selectedOption = SPECIALITY_OPTIONS.find(
        (opt) => opt.value === formData.speciality
      );
      const specialityLabel = selectedOption ? selectedOption.label : formData.speciality;

      const newTransaction = {
        id: "T" + data.transaction_id,
        patient_id: formData.patient_id,
        doctor_id: formData.doctor_id,
        speciality: specialityLabel,
        diagnosis: formData.diagnosis,
        service_description: formData.service_description,
        service_id: data.transaction_details?.service_id || "",
        status: data.overall_risk,
        risk_score: data.ttsgan?.score?.toFixed(2) || "0.00",
        timestamp: new Date().toLocaleString(),
      };

      const existing = JSON.parse(localStorage.getItem("transactions") || "[]");
      existing.unshift(newTransaction);
      localStorage.setItem("transactions", JSON.stringify(existing));

      setResult(newTransaction);
      setSubmitted(true);
      setTimeout(() => navigate("/dashboard"), 5000);

    } catch (err) {
      setLoading(false);
      alert("Server se connect nahi ho saka! Backend chala hua hai?");
      console.error(err);
    }
  };

  return (
    <div className="trans-app">
      <Navbar />
      <div className="trans-main">

        <svg className="trans-bg-shapes" viewBox="0 0 560 480" xmlns="http://www.w3.org/2000/svg">
          <circle cx="480" cy="70" r="170" fill="#0ea5e9" />
          <circle cx="330" cy="390" r="120" fill="#0d9488" />
          <rect x="90" y="40" width="110" height="110" rx="28" fill="#0ea5e9" transform="rotate(20 145 95)" />
          <rect x="400" y="280" width="85" height="85" rx="18" fill="#0d9488" transform="rotate(45 442 322)" />
          <circle cx="70" cy="410" r="65" fill="#0ea5e9" />
        </svg>

        <div className="trans-topbar">
          <div>
            <h1>New <span className="trans-topbar-highlight">Transaction</span></h1>
            <p>Submit a new claim for fraud analysis</p>
          </div>
        </div>

        <div className="trans-container">
          <motion.div
            className="trans-card"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            {loading && (
              <div className="trans-loading-overlay">
                <div className="trans-loading-box">
                  <div className="trans-spinner"></div>
                  <p className="trans-loading-text">Analyzing Transaction...</p>
                  <p className="trans-loading-sub">AI fraud detection in progress</p>
                </div>
              </div>
            )}

            {submitted && result ? (
              <motion.div
                className="trans-success"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <div className="trans-success-icon">{getStatusIcon(result.status)}</div>
                <h3>Transaction Submitted!</h3>
                <p className="trans-success-sub">AI analysis complete — here are your results</p>

                <div className="trans-result-banner" style={{
                  background: `${getRiskColor(result.status)}12`,
                  border: `1px solid ${getRiskColor(result.status)}33`,
                }}>
                  <div className="trans-result-left">
                    <span>{getStatusIcon(result.status)}</span>
                    <div>
                      <p className="trans-result-meta">Detection Result</p>
                      <p className="trans-result-val" style={{ color: getRiskColor(result.status) }}>
                        {getStatusFromRisk(result.status)}
                      </p>
                    </div>
                  </div>
                  <div className="trans-result-right">
                    <p className="trans-result-meta">Risk Score</p>
                    <p className="trans-result-score" style={{ color: getRiskColor(result.status) }}>
                      {result.risk_score}
                    </p>
                  </div>
                </div>

                <div className="trans-result-grid">
                  {[
                    { label: "Transaction ID", value: result.id },
                    { label: "Patient ID", value: result.patient_id },
                    { label: "Doctor ID", value: result.doctor_id },
                    { label: "Speciality", value: result.speciality },
                    { label: "Service ID", value: result.service_id },
                    { label: "Diagnosis", value: result.diagnosis },
                  ].map((item, i) => (
                    <div key={i} className="trans-result-item">
                      <span className="trans-result-label">{item.label}</span>
                      <span className="trans-result-value">{item.value}</span>
                    </div>
                  ))}
                </div>

                <p className="trans-redirect">Redirecting to dashboard in 5 seconds...</p>
                <button className="trans-go-btn" onClick={() => navigate("/dashboard")}>
                  Go to Dashboard →
                </button>
              </motion.div>
            ) : (
              <>
                <div className="trans-card-header">
                  <h2>Claim Details</h2>
                  <p>Fill in the information below to submit for analysis</p>
                </div>

                <form onSubmit={handleSubmit}>
                  <div className="trans-row">
                    <div className="trans-field">
                      <label>Patient ID</label>
                      <input type="text" name="patient_id" placeholder="e.g. P001" onChange={handleChange} required />
                    </div>
                    <div className="trans-field">
                      <label>Doctor ID</label>
                      <input type="text" name="doctor_id" placeholder="e.g. D001" onChange={handleChange} required />
                    </div>
                  </div>

                  <div className="trans-field">
                    <label>Speciality</label>
                    <SpecialityDropdown
                      value={formData.speciality}
                      onChange={handleSpecialityChange}
                      placeholder="Type or select speciality..."
                    />
                  </div>

                  <div className="trans-field">
                    <label>Diagnosis</label>
                    <input type="text" name="diagnosis" placeholder="e.g. Acute Myocardial Infarction" onChange={handleChange} required />
                  </div>

                  <div className="trans-field">
                    <label>Service Description</label>
                    <SearchableDropdown
                      options={services}
                      value={formData.service_description}
                      onChange={handleServiceChange}
                      placeholder={
                        servicesLoading
                          ? "Loading services..."
                          : formData.speciality
                            ? "Type or select service..."
                            : "Select speciality first..."
                      }
                      disabled={!formData.speciality || servicesLoading}
                    />
                  </div>

                  <motion.button
                    type="submit"
                    className="trans-btn"
                    whileHover={!loading ? { scale: 1.02 } : {}}
                    whileTap={!loading ? { scale: 0.98 } : {}}
                    disabled={loading}
                  >
                    Submit Transaction →
                  </motion.button>
                </form>
              </>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Transaction;