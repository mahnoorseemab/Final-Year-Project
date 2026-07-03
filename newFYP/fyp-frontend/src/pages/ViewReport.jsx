import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { jsPDF } from "jspdf";
import Navbar from "../components/Navbar";
import "../styles/ViewReport.css";

const ViewReport = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [transaction, setTransaction] = useState(null);
  const [loading, setLoading] = useState(true);

  const numericId = id?.replace("T", "");

  useEffect(() => {
    const fetchTransaction = async () => {
      try {
        // Pehle localStorage mein check karo — fresh report hai wahan
        const cached = localStorage.getItem(`report_${numericId}`);
        if (cached) {
          const data = JSON.parse(cached);
          setTransaction(data);
          setLoading(false);
          return;
        }

        // Agar localStorage mein nahi to backend se fetch karo
        const res = await fetch(`http://localhost:8000/transaction/${numericId}`);
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
    if (risk === "HIGH RISK") return "Fraud";
    if (risk === "MEDIUM RISK") return "Suspicious";
    return "Legit";
  };

  const getRiskColor = (risk) => {
    if (risk === "HIGH RISK") return "#ef4444";
    if (risk === "MEDIUM RISK") return "#d97706";
    return "#0d9488";
  };

  const getStatusClass = (risk) => {
    if (risk === "HIGH RISK") return "vr-fraud";
    if (risk === "MEDIUM RISK") return "vr-suspicious";
    return "vr-legit";
  };

  const getRiskPercent = (risk) => {
    if (risk === "HIGH RISK") return 90;
    if (risk === "MEDIUM RISK") return 55;
    return 15;
  };

  const downloadPDF = () => {
    if (!transaction) return;
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const risk = transaction.overall_risk;
    const explanation = transaction?.agent1?.report || transaction?.report || "Report not available.";

    // ── HEADER ──────────────────────────────────────────
    doc.setFillColor(2, 8, 24);
    doc.rect(0, 0, pageWidth, 45, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(22);
    doc.setFont("helvetica", "bold");
    doc.text("MediGuard AI", 20, 20);
    doc.setFontSize(11);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(0, 212, 255);
    doc.text("Healthcare Fraud Detection Report", 20, 32);
    doc.setTextColor(180, 190, 210);
    doc.setFontSize(9);
    doc.text(`Generated: ${new Date().toLocaleString()}`, pageWidth - 15, 32, { align: "right" });

    // ── TRANSACTION ID BAR ───────────────────────────────
    doc.setFillColor(26, 107, 255);
    doc.rect(0, 45, pageWidth, 12, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(10);
    doc.setFont("helvetica", "bold");
    doc.text(`Transaction ID: T${transaction.transaction_id || transaction.id}`, pageWidth / 2, 53, { align: "center" });

    let y = 68;

    // ── TRANSACTION DETAILS SECTION ──────────────────────
    doc.setFillColor(240, 249, 255);
    doc.roundedRect(10, y - 6, pageWidth - 20, 8, 2, 2, "F");
    doc.setTextColor(2, 132, 199);
    doc.setFontSize(10);
    doc.setFont("helvetica", "bold");
    doc.text("TRANSACTION DETAILS", 15, y);
    y += 8;

    // divider line
    doc.setDrawColor(186, 230, 253);
    doc.line(10, y, pageWidth - 10, y);
    y += 8;

    const txDetails = transaction.transaction_details || {};
    const details = [
      ["Patient ID", transaction.patient_id],
      ["Doctor ID", transaction.doctor_id],
      ["Speciality", txDetails.speciality_name || transaction.speciality_name],
      ["Service ID", txDetails.service_id || transaction.service_id],
      ["Diagnosis", txDetails.diagnosis || transaction.diagnosis || "-"],
      ["Date", transaction.created_at || "-"],
    ];

    details.forEach(([label, value], i) => {
      const isLeft = i % 2 === 0;
      const col = isLeft ? 15 : pageWidth / 2 + 5;
      const row = y + Math.floor(i / 2) * 13;
      doc.setFont("helvetica", "bold");
      doc.setFontSize(8.5);
      doc.setTextColor(100, 116, 139);
      doc.text(`${label}:`, col, row);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(15, 23, 42);
      doc.text(String(value || "-"), col + 28, row);
    });

    y += Math.ceil(details.length / 2) * 13 + 6;

    // Service Description
    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(100, 116, 139);
    doc.text("Service Description:", 15, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(15, 23, 42);
    const svcDesc = txDetails.service_description || transaction.service_description || "-";
    const svcLines = doc.splitTextToSize(svcDesc, pageWidth - 60);
    doc.text(svcLines, 55, y);
    y += svcLines.length * 5 + 10;

    // ── OVERALL RISK ─────────────────────────────────────
    const riskRGB = risk === "HIGH RISK" ? [220, 38, 38] : risk === "MEDIUM RISK" ? [217, 119, 6] : [13, 148, 136];
    doc.setFillColor(...riskRGB.map(c => Math.round(c * 0.15 + 240)));
    doc.roundedRect(10, y - 6, pageWidth - 20, 14, 2, 2, "F");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.setTextColor(100, 116, 139);
    doc.text("Overall Risk:", 15, y + 2);
    doc.setTextColor(...riskRGB);
    doc.setFontSize(11);
    doc.text(getStatusFromRisk(risk).toUpperCase(), 55, y + 2);
    y += 20;

    // ── INVESTIGATION REPORT SECTION ─────────────────────
    doc.setFillColor(240, 249, 255);
    doc.roundedRect(10, y - 6, pageWidth - 20, 8, 2, 2, "F");
    doc.setTextColor(2, 132, 199);
    doc.setFontSize(10);
    doc.setFont("helvetica", "bold");
    doc.text("INVESTIGATION REPORT", 15, y);
    y += 8;
    doc.setDrawColor(186, 230, 253);
    doc.line(10, y, pageWidth - 10, y);
    y += 8;

    // Parse report lines — headings bold, rest normal
    const reportLines = explanation.split("\n");
    const headingKeywords = [
      "VERDICT", "EXECUTIVE SUMMARY", "TRANSACTION ANALYSIS",
      "BILLING PATTERN ANALYSIS", "CLINICAL SIGNALS",
      "INVESTIGATION FINDINGS", "POLICY VIOLATIONS",
      "OVERBILLING ASSESSMENT", "DOCTOR ACCOUNTABILITY",
      "RECOMMENDATION", "ACTION ITEMS", "NOTES", "DISCLAIMER",
      "IMPORTANT DISCLAIMER"
    ];

    const margin = 15;
    const maxWidth = pageWidth - margin * 2;

    reportLines.forEach((line) => {
      const trimmed = line.trim();
      if (!trimmed) { y += 3; return; }

      // Check if new page needed
      if (y > pageHeight - 20) {
        doc.addPage();
        y = 20;
      }

      const isHeading = headingKeywords.some(kw => trimmed.toUpperCase().startsWith(kw));
      const isDivider = trimmed.startsWith("===");

      if (isDivider) {
        doc.setDrawColor(226, 232, 240);
        doc.line(margin, y, pageWidth - margin, y);
        y += 4;
        return;
      }

      if (isHeading) {
        // Heading background
        doc.setFillColor(248, 250, 252);
        doc.rect(margin - 2, y - 4, maxWidth + 4, 8, "F");
        doc.setFont("helvetica", "bold");
        doc.setFontSize(9.5);
        doc.setTextColor(14, 165, 233);
        const wrapped = doc.splitTextToSize(trimmed, maxWidth);
        doc.text(wrapped, margin, y);
        y += wrapped.length * 6 + 3;
      } else {
        doc.setFont("helvetica", "normal");
        doc.setFontSize(8.5);
        doc.setTextColor(30, 41, 59);
        const wrapped = doc.splitTextToSize(trimmed, maxWidth);
        doc.text(wrapped, margin, y);
        y += wrapped.length * 5 + 2;
      }
    });

    doc.save(`MediGuard_Report_T${transaction.transaction_id || transaction.id}.pdf`);
  };

  if (loading) {
    return (
      <div className="vr-app">
        <Navbar />
        <div className="vr-main">
          <div className="vr-not-found"><span>⏳</span><p>Loading report...</p></div>
        </div>
      </div>
    );
  }

  if (!transaction) {
    return (
      <div className="vr-app">
        <Navbar />
        <div className="vr-main">
          <div className="vr-not-found">
            <span>🔍</span>
            <p>Transaction not found.</p>
            <button className="vr-back-btn" onClick={() => navigate(-1)}>Go Back</button>
          </div>
        </div>
      </div>
    );
  }

  const risk = transaction.overall_risk;
  const txDetails = transaction.transaction_details || {};
  const explanation = transaction?.agent1?.report ||
    transaction?.report ||
    "Report not available for this transaction.";

  return (
    <div className="vr-app">
      <Navbar />
      <div className="vr-main">

        <svg className="vr-bg-shapes" viewBox="0 0 560 480" xmlns="http://www.w3.org/2000/svg">
          <circle cx="480" cy="70" r="170" fill="#0ea5e9" />
          <circle cx="330" cy="390" r="120" fill="#0d9488" />
          <rect x="90" y="40" width="110" height="110" rx="28" fill="#0ea5e9" transform="rotate(20 145 95)" />
          <rect x="400" y="280" width="85" height="85" rx="18" fill="#0d9488" transform="rotate(45 442 322)" />
          <circle cx="70" cy="410" r="65" fill="#0ea5e9" />
        </svg>

        <div className="vr-topbar">
          <div>
            <h1>Fraud <span className="vr-highlight">Report</span></h1>
            <p>Transaction ID: <span className="vr-id-val">T{transaction.transaction_id || transaction.id}</span></p>
            <p className="vr-date">Generated: {new Date().toLocaleString()}</p>
          </div>
          <span className={`vr-status-badge ${getStatusClass(risk)}`}>
            {getStatusFromRisk(risk)}
          </span>
        </div>

        <div className="vr-container">

          <motion.div className="vr-card"
            initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}>
            <div className="vr-card-header"><span>📋</span><h2>Transaction Information</h2></div>
            <div className="vr-info-grid">
              {[
                { label: "Patient ID", value: transaction.patient_id },
                { label: "Doctor ID", value: transaction.doctor_id },
                { label: "Speciality", value: txDetails.speciality_name || transaction.speciality_name },
                { label: "Service ID", value: txDetails.service_id || transaction.service_id },
                { label: "Diagnosis", value: txDetails.diagnosis || transaction.diagnosis },
                { label: "Date & Time", value: transaction.created_at },
              ].map((item, i) => (
                <div key={i} className="vr-info-item">
                  <span className="vr-info-label">{item.label}</span>
                  <span className="vr-info-value">{item.value || "-"}</span>
                </div>
              ))}
            </div>
            <div className="vr-desc-wrap">
              <span className="vr-info-label">Service Description</span>
              <p className="vr-desc">{txDetails.service_description || transaction.service_description || "-"}</p>
            </div>
          </motion.div>

          <motion.div className="vr-card"
            initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}>
            <div className="vr-card-header"><span>🎯</span><h2>Risk Analysis</h2></div>
            <div className="vr-risk-row">
              <div className="vr-risk-circle" style={{ borderColor: getRiskColor(risk) }}>
                <span className="vr-risk-num" style={{ color: getRiskColor(risk) }}>
                  {getRiskPercent(risk)}%
                </span>
                <span className="vr-risk-sublabel">Risk Score</span>
              </div>
              <div className="vr-risk-bar-section">
                <div className="vr-risk-bar-label">
                  <span>Fraud Probability</span>
                  <span style={{ color: getRiskColor(risk) }}>{getRiskPercent(risk)}%</span>
                </div>
                <div className="vr-risk-bar-bg">
                  <motion.div
                    className="vr-risk-bar-fill"
                    style={{ background: getRiskColor(risk) }}
                    initial={{ width: 0 }}
                    animate={{ width: `${getRiskPercent(risk)}%` }}
                    transition={{ duration: 1, delay: 0.5 }}
                  />
                </div>
                <div className="vr-final-status">
                  <span className="vr-info-label">Final Status</span>
                  <span className={`vr-status-badge ${getStatusClass(risk)}`}>
                    {getStatusFromRisk(risk)}
                  </span>
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div className="vr-card vr-ai-card"
            initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}>
            <div className="vr-card-header">
              <span>🧠</span>
              <h2>AI Analysis & RAG Explanation</h2>
              <span className="vr-ai-badge">Powered by LLaMA 3.3 70B</span>
            </div>
            <div className="vr-explanation">
              {explanation.split("\n\n").map((para, i) => (
                <p key={i} className="vr-para">{para}</p>
              ))}
            </div>
          </motion.div>

          <motion.div className="vr-actions"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.4 }}>
            <button className="vr-back-btn" onClick={() => navigate(-1)}>← Back</button>
            <button className="vr-download-btn" onClick={downloadPDF}>⬇ Download PDF Report</button>
          </motion.div>

        </div>
      </div>
    </div>
  );
};

export default ViewReport;