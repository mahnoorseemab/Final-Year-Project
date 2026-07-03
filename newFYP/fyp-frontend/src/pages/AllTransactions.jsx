import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import Navbar from "../components/Navbar";
import "../styles/AllTransactions.css";

const AllTransactions = () => {
  const navigate = useNavigate();
  const [transactions, setTransactions] = useState([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        const res = await fetch("http://localhost:8000/all-transactions");
        const data = await res.json();
        setTransactions(data.transactions || []);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchTransactions();
  }, []);

  const getStatusFromRisk = (risk) => {
    if (risk === "HIGH RISK")   return "Fraud";
    if (risk === "MEDIUM RISK") return "Suspicious";
    return "Legit";
  };

  const getStatusClass = (risk) => {
    if (risk === "HIGH RISK")   return "at-fraud";
    if (risk === "MEDIUM RISK") return "at-suspicious";
    return "at-legit";
  };

  const getRiskColor = (risk) => {
    if (risk === "HIGH RISK")   return "#ef4444";
    if (risk === "MEDIUM RISK") return "#d97706";
    return "#0d9488";
  };

  const filtered = transactions.filter(t =>
    String(t.patient_id)?.toLowerCase().includes(search.toLowerCase()) ||
    t.doctor_id?.toLowerCase().includes(search.toLowerCase()) ||
    t.speciality_name?.toLowerCase().includes(search.toLowerCase()) ||
    t.diagnosis?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="at-app">
      <Navbar />
      <div className="at-main">

        <svg className="at-bg-shapes" viewBox="0 0 560 480" xmlns="http://www.w3.org/2000/svg">
          <circle cx="480" cy="70"  r="170" fill="#0ea5e9" />
          <circle cx="330" cy="390" r="120" fill="#0d9488" />
          <rect x="90"  y="40"  width="110" height="110" rx="28" fill="#0ea5e9" transform="rotate(20 145 95)" />
          <rect x="400" y="280" width="85"  height="85"  rx="18" fill="#0d9488" transform="rotate(45 442 322)" />
          <circle cx="70" cy="410" r="65" fill="#0ea5e9" />
        </svg>

        <div className="at-topbar">
          <div>
            <h1>All <span className="at-topbar-highlight">Transactions</span></h1>
            <p>Complete transaction history with fraud analysis</p>
          </div>
          <div className="at-count-badge">{transactions.length} Total</div>
        </div>

        <div className="at-container">

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <input
              type="text"
              className="at-search"
              placeholder="🔍  Search by Patient ID, Doctor ID, Speciality, Diagnosis..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </motion.div>

          <motion.div
            className="at-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            {loading ? (
              <div className="at-empty">Loading transactions...</div>
            ) : filtered.length === 0 ? (
              <div className="at-empty">
                {search ? "No transactions found for your search." : "No transactions yet."}
              </div>
            ) : (
              <div className="at-table-wrap">
                <table className="at-table">
                  <thead>
                    <tr>
                      <th>Trans ID</th>
                      <th>Patient ID</th>
                      <th>Doctor ID</th>
                      <th>Speciality</th>
                      <th>Diagnosis</th>
                      <th>Service ID</th>
                      <th>Status</th>
                      <th>Date</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((t, i) => (
                      <motion.tr
                        key={i}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3, delay: i * 0.03 }}
                      >
                        <td className="at-id">T{t.id}</td>
                        <td>{t.patient_id}</td>
                        <td>{t.doctor_id}</td>
                        <td>{t.speciality_name}</td>
                        <td>{t.diagnosis}</td>
                        <td>{t.service_id}</td>
                        <td>
                          <span className={`at-status ${getStatusClass(t.overall_risk)}`}>
                            {getStatusFromRisk(t.overall_risk)}
                          </span>
                        </td>
                        <td className="at-date">{t.created_at}</td>
                        <td>
                          <button className="at-btn" onClick={() => navigate(`/fraud-details/T${t.id}`)}>
                            Details
                          </button>
                        </td>
                      </motion.tr>
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

export default AllTransactions;