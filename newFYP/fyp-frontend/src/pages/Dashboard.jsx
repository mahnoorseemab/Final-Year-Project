import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import Navbar from "../components/Navbar";
import "../styles/Dashboard.css";
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from "recharts";

const Dashboard = () => {
  const navigate = useNavigate();
  const role = localStorage.getItem("role");
  const user = localStorage.getItem("user");

  const [transactions, setTransactions] = useState([]);
  const [fraudDist, setFraudDist] = useState([]);
  const [doctorScores, setDoctorScores] = useState([]);
  const [counts, setCounts] = useState({ transactions: 0, fraud: 0, suspicious: 0, doctors: 0 });

  // Fetch recent transactions from backend
  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        const res = await fetch("http://localhost:8000/recent-transactions");
        const data = await res.json();
        setTransactions(data.transactions || []);

        const fraud = (data.transactions || []).filter(t => t.overall_risk === "HIGH RISK").length;
        const suspicious = (data.transactions || []).filter(t => t.overall_risk === "MEDIUM RISK").length;
        const doctors = [...new Set((data.transactions || []).map(t => t.doctor_id))].length;

        const targets = {
          transactions: data.total || 0,
          fraud, suspicious, doctors
        };

        const steps = 60;
        let step = 0;
        const timer = setInterval(() => {
          step++;
          setCounts({
            transactions: Math.min(Math.round((targets.transactions / steps) * step), targets.transactions),
            fraud: Math.min(Math.round((targets.fraud / steps) * step), targets.fraud),
            suspicious: Math.min(Math.round((targets.suspicious / steps) * step), targets.suspicious),
            doctors: Math.min(Math.round((targets.doctors / steps) * step), targets.doctors),
          });
          if (step >= steps) clearInterval(timer);
        }, 2000 / steps);

      } catch (err) {
        console.error("Transactions fetch error:", err);
      }
    };
    fetchTransactions();
  }, []);

  // Fetch fraud distribution for pie chart
  useEffect(() => {
    const fetchFraudDist = async () => {
      try {
        const res = await fetch("http://localhost:8000/fraud-distribution");
        const data = await res.json();
        const dist = data.distribution || {};
        setFraudDist([
          { name: "High Risk", value: dist["HIGH RISK"] || 0, color: "#ef4444" },
          { name: "Medium Risk", value: dist["MEDIUM RISK"] || 0, color: "#d97706" },
          { name: "Normal", value: dist["NORMAL"] || 0, color: "#0d9488" },
        ]);
      } catch (err) {
        console.error("Fraud dist error:", err);
      }
    };
    fetchFraudDist();
  }, []);

  // Fetch doctor scores for bar chart
  useEffect(() => {
    const fetchDoctorScores = async () => {
      try {
        const res = await fetch("http://localhost:8000/doctor-scores");
        const data = await res.json();
        setDoctorScores((data.scores || []).slice(-8).reverse());
      } catch (err) {
        console.error("Doctor scores error:", err);
      }
    };
    fetchDoctorScores();
  }, []);

  const getStatusFromRisk = (risk) => {
    if (risk === "HIGH RISK") return "Fraud";
    if (risk === "MEDIUM RISK") return "Suspicious";
    return "Legit";
  };

  const getStatusClass = (risk) => {
    if (risk === "HIGH RISK") return "status-fraud";
    if (risk === "MEDIUM RISK") return "status-suspicious";
    return "status-legit";
  };

  const getRiskColor = (risk) => {
    if (risk === "HIGH RISK") return "#ef4444";
    if (risk === "MEDIUM RISK") return "#d97706";
    return "#0d9488";
  };

  const getBarColor = (score) => {
    if (score <= 30) return "#ef4444";
    if (score <= 60) return "#d97706";
    return "#0d9488";
  };

  const stats = [
    { icon: "📋", label: "Total Transactions", value: counts.transactions, color: "s-blue", trend: "All time" },
    { icon: "🚨", label: "Fraud Detected", value: counts.fraud, color: "s-red", trend: "Requires attention" },
    { icon: "⚠️", label: "Suspicious Cases", value: counts.suspicious, color: "s-amber", trend: "Under review" },
    { icon: "👨‍⚕️", label: "Doctors Monitored", value: counts.doctors, color: "s-teal", trend: "Active in system" },
  ];

  return (
    <div className="dash-app">
      <Navbar />
      <div className="dash-main">

        <svg className="dash-bg-shapes" viewBox="0 0 560 480" xmlns="http://www.w3.org/2000/svg">
          <circle cx="480" cy="70" r="170" fill="#0ea5e9" />
          <circle cx="330" cy="390" r="120" fill="#0d9488" />
          <rect x="90" y="40" width="110" height="110" rx="28" fill="#0ea5e9" transform="rotate(20 145 95)" />
          <rect x="400" y="280" width="85" height="85" rx="18" fill="#0d9488" transform="rotate(45 442 322)" />
          <circle cx="70" cy="410" r="65" fill="#0ea5e9" />
        </svg>

        {/* Topbar */}
        <div className="dash-topbar">
          <div>
            <p className="dash-topbar-date">
              {new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}
            </p>
            <h1>Welcome back, <span className="dash-topbar-name">{user || "User"}</span> 👋</h1>
            <p className="dash-topbar-sub">Here's your fraud monitoring overview</p>
          </div>
          <div className="dash-role-badge">
            <span className="dash-role-dot" />
            {role}
          </div>
        </div>

        <div className="dash-container">

          {/* Stats */}
          <div className="dash-stats">
            {stats.map((s, i) => (
              <motion.div
                key={i}
                className={`dash-stat-card ${s.color}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: i * 0.08 }}
                whileHover={{ y: -3, transition: { duration: 0.2 } }}
              >
                <div className="dash-stat-top">
                  <div className={`dash-stat-icon ${s.color}`}>{s.icon}</div>
                  <span className="dash-stat-trend">{s.trend}</span>
                </div>
                <div className={`dash-stat-value ${s.color}`}>{s.value}</div>
                <div className="dash-stat-label">{s.label}</div>
              </motion.div>
            ))}
          </div>

          {/* Charts Row */}
          <div className="dash-charts-row">

            {/* Pie Chart */}
            <motion.div className="dash-card dash-chart-card"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}>
              <div className="dash-card-header">
                <div>
                  <h3>Risk Distribution</h3>
                  <p>Fraud vs Suspicious vs Normal</p>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={fraudDist}
                    cx="50%" cy="50%"
                    innerRadius={60} outerRadius={90}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {fraudDist.map((entry, index) => (
                      <Cell key={index} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [value, "Transactions"]} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Bar Chart */}
            <motion.div className="dash-card dash-chart-card"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}>
              <div className="dash-card-header">
                <div>
                  <h3>Doctor Trust Scores</h3>
                  <p>Top flagged doctors (0-100)</p>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={doctorScores} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="doctor_id" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(value) => [value.toFixed(1), "Trust Score"]} />
                  <Bar dataKey="current_score" radius={[4, 4, 0, 0]}>
                    {doctorScores.map((entry, index) => (
                      <Cell key={index} fill={getBarColor(entry.current_score)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </motion.div>
          </div>

          {/* Recent Transactions */}
          <motion.div className="dash-card"
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}>
            <div className="dash-card-header">
              <div>
                <h3>Recent Transactions</h3>
                <p>Latest 5 submitted claims</p>
              </div>
              <button className="dash-view-all" onClick={() => navigate("/all-transactions")}>View All →</button>
            </div>
            {transactions.length === 0 ? (
              <div className="dash-empty">No transactions yet.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead>
                    <tr>
                      <th>Trans ID</th><th>Patient ID</th><th>Doctor ID</th>
                      <th>Speciality</th><th>Diagnosis</th><th>Service ID</th>
                      <th>Status</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {transactions.slice(0, 5).map((t, i) => (
                      <tr key={i}>
                        <td className="dash-id">T{t.id}</td>
                        <td>{t.patient_id}</td>
                        <td>{t.doctor_id}</td>
                        <td>{t.speciality_name}</td>
                        <td>{t.diagnosis}</td>
                        <td>{t.service_id}</td>
                        <td>
                          <span className={`dash-status ${getStatusClass(t.overall_risk)}`}>
                            {getStatusFromRisk(t.overall_risk)}
                          </span>
                        </td>
                        <td>
                          <button className="dash-details-btn" onClick={() => navigate(`/fraud-details/T${t.id}`)}>
                            Details
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </motion.div>

        </div>
      </div>
    </div>
  );
};

export default Dashboard;