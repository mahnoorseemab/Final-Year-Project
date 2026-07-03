import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import Navbar from "../components/Navbar";
import "../styles/FraudDetails.css";

const FraudDetails = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [transaction, setTransaction] = useState(null);
  const [loading, setLoading] = useState(true);

  const numericId = id?.replace("T", "");

  useEffect(() => {
    const fetchTransaction = async () => {
      try {
        const res  = await fetch(`http://localhost:8000/transaction/${numericId}`);
        if (!res.ok) { setTransaction(null); setLoading(false); return; }
        const data = await res.json();
        setTransaction(data);
      } catch (err) {
        console.error(err);
        setTransaction(null);
      } finally {
        setLoading(false);
      }
    };
    fetchTransaction();
  }, [numericId]);

  const getStatusFromRisk = (risk) => {
    if (risk === "HIGH RISK")   return "Fraud";
    if (risk === "MEDIUM RISK") return "Suspicious";
    return "Legit";
  };

  const getStatusClass = (risk) => {
    if (risk === "HIGH RISK")   return "fd-fraud";
    if (risk === "MEDIUM RISK") return "fd-suspicious";
    return "fd-legit";
  };

  const getRiskColor = (risk) => {
    if (risk === "HIGH RISK")   return "#ef4444";
    if (risk === "MEDIUM RISK") return "#d97706";
    return "#0d9488";
  };

  const getRiskDescription = (risk) => {
    if (risk === "HIGH RISK")   return "⚠️ High fraud probability detected. Immediate review recommended.";
    if (risk === "MEDIUM RISK") return "🔶 Moderate risk detected. Further investigation advised.";
    return "✅ Low risk. Transaction appears legitimate.";
  };

  const getRiskPercent = (risk) => {
    if (risk === "HIGH RISK")   return 90;
    if (risk === "MEDIUM RISK") return 55;
    return 15;
  };

  if (loading) {
    return (
      <div className="fd-app">
        <Navbar />
        <div className="fd-main">
          <div className="fd-not-found">
            <span>⏳</span>
            <p>Loading transaction...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!transaction) {
    return (
      <div className="fd-app">
        <Navbar />
        <div className="fd-main">
          <div className="fd-not-found">
            <span>🔍</span>
            <p>Transaction not found.</p>
            <button className="fd-back-btn" onClick={() => navigate(-1)}>Go Back</button>
          </div>
        </div>
      </div>
    );
  }

  const risk = transaction.overall_risk;

  return (
    <div className="fd-app">
      <Navbar />
      <div className="fd-main">

        <svg className="fd-bg-shapes" viewBox="0 0 560 480" xmlns="http://www.w3.org/2000/svg">
          <circle cx="480" cy="70"  r="170" fill="#0ea5e9" />
          <circle cx="330" cy="390" r="120" fill="#0d9488" />
          <rect x="90"  y="40"  width="110" height="110" rx="28" fill="#0ea5e9" transform="rotate(20 145 95)" />
          <rect x="400" y="280" width="85"  height="85"  rx="18" fill="#0d9488" transform="rotate(45 442 322)" />
          <circle cx="70" cy="410" r="65" fill="#0ea5e9" />
        </svg>

        <div className="fd-topbar">
          <div>
            <h1>Transaction <span className="fd-highlight">Details</span></h1>
            <p>ID: <span className="fd-id-val">T{transaction.id}</span></p>
          </div>
          <span className={`fd-status-badge ${getStatusClass(risk)}`}>
            {getStatusFromRisk(risk)}
          </span>
        </div>

        <div className="fd-container">
          <div className="fd-grid">

            {/* Transaction Info */}
            <motion.div className="fd-card"
              initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}>
              <div className="fd-card-header">
                <span>📋</span>
                <h2>Transaction Information</h2>
              </div>
              <div className="fd-info-grid">
                {[
                  { label: "Patient ID",  value: transaction.patient_id },
                  { label: "Doctor ID",   value: transaction.doctor_id },
                  { label: "Speciality",  value: transaction.speciality_name },
                  { label: "Service ID",  value: transaction.service_id },
                  { label: "Diagnosis",   value: transaction.diagnosis },
                  { label: "Date & Time", value: transaction.created_at },
                ].map((item, i) => (
                  <div key={i} className="fd-info-item">
                    <span className="fd-info-label">{item.label}</span>
                    <span className="fd-info-value">{item.value || "-"}</span>
                  </div>
                ))}
              </div>
              <div className="fd-desc-section">
                <span className="fd-info-label">Service Description</span>
                <p className="fd-desc">{transaction.service_description || "-"}</p>
              </div>
            </motion.div>

            {/* Risk Analysis */}
            <motion.div className="fd-card"
              initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}>
              <div className="fd-card-header">
                <span>🎯</span>
                <h2>Risk Analysis</h2>
              </div>

              <div className="fd-risk-display">
                <div className="fd-risk-circle" style={{ borderColor: getRiskColor(risk) }}>
                  <span className="fd-risk-num" style={{ color: getRiskColor(risk) }}>
                    {getRiskPercent(risk)}%
                  </span>
                  <span className="fd-risk-sublabel">Risk Score</span>
                </div>
                <div className="fd-risk-info">
                  <p className="fd-risk-desc">{getRiskDescription(risk)}</p>
                  {transaction.ttsgan_score !== null && (
                    <div style={{ marginTop: "0.8rem", fontSize: "0.82rem", color: "#64748b" }}>
                      <p>TTS-WGAN Score: <strong style={{ color: getRiskColor(risk) }}>{transaction.ttsgan_score?.toFixed(4)}</strong></p>
                      <p>DCT-GAN Score: <strong style={{ color: getRiskColor(risk) }}>{transaction.dctgan_score?.toFixed(4)}</strong></p>
                    </div>
                  )}
                </div>
              </div>

              <div className="fd-risk-bar-section">
                <div className="fd-risk-bar-label">
                  <span>Risk Level</span>
                  <span style={{ color: getRiskColor(risk) }}>{getStatusFromRisk(risk)}</span>
                </div>
                <div className="fd-risk-bar-bg">
                  <motion.div
                    className="fd-risk-bar-fill"
                    style={{ background: getRiskColor(risk) }}
                    initial={{ width: 0 }}
                    animate={{ width: `${getRiskPercent(risk)}%` }}
                    transition={{ duration: 1, delay: 0.5 }}
                  />
                </div>
              </div>

              <div className="fd-final-status">
                <span className="fd-info-label">Final Status</span>
                <span className={`fd-status-badge ${getStatusClass(risk)}`}>
                  {getStatusFromRisk(risk)}
                </span>
              </div>
            </motion.div>
          </div>

          <motion.div className="fd-actions"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.4 }}>
            <button className="fd-back-btn" onClick={() => navigate(-1)}>← Back</button>
            <button className="fd-report-btn" onClick={() => navigate(`/view-report/T${transaction.id}`)}>
              View Report →
            </button>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default FraudDetails;