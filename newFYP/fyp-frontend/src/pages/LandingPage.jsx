import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import doctorImg from "../assets/doctor_no_bg.png"
import "../styles/LandingPage.css";

const LandingPage = () => {
  const navigate = useNavigate();
  const [counts, setCounts] = useState({ accuracy: 0, transactions: 0, doctors: 0 });

  useEffect(() => {
    const targets = { accuracy: 97, transactions: 4252, doctors: 150 };
    const steps = 60;
    let step = 0;
    const timer = setInterval(() => {
      step++;
      setCounts({
        accuracy: Math.min(Math.round((targets.accuracy / steps) * step), targets.accuracy),
        transactions: Math.min(Math.round((targets.transactions / steps) * step), targets.transactions),
        doctors: Math.min(Math.round((targets.doctors / steps) * step), targets.doctors),
      });
      if (step >= steps) clearInterval(timer);
    }, 2000 / steps);
    return () => clearInterval(timer);
  }, []);

  const aboutCards = [
    { icon: "🔍", title: "Anomaly Detection", desc: "Our system automatically detects unusual and suspicious patterns in healthcare billing and transactions before they become costly problems." },
    { icon: "🛡️", title: "Fraud Prevention", desc: "Real-time monitoring flags high-risk claims instantly, helping hospital management take action before fraudulent transactions are processed." },
    { icon: "📊", title: "Risk Scoring", desc: "Every transaction is assigned a clear risk score — Low, Suspicious, or Fraud — so your team always knows where to focus attention." },
    { icon: "👁️", title: "Human Review", desc: "Final decisions always remain with your qualified reviewers. Our system supports your team, not replaces them." },
    { icon: "📋", title: "Detailed Reports", desc: "Get clear, easy-to-read fraud analysis reports with evidence and recommendations for every flagged transaction." },
    { icon: "⚡", title: "Real-Time Monitoring", desc: "Transactions are analyzed instantly as they are submitted, giving your management team live visibility into the system." },
  ];

  return (
    <div className="landing">

      {/* ── NAVBAR ── */}
      <nav className="land-nav">
        <motion.div className="land-logo" initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6 }}>
          <div className="land-logo-icon">🛡️</div>
          <span>Medi<span className="logo-highlight">Guard AI</span></span>
        </motion.div>

        <motion.ul className="land-links" initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.2 }}>
          <li><a href="#home">Home</a></li>
          <li><a href="#about">About Us</a></li>
          <li><a href="#contact">Contact</a></li>
        </motion.ul>

        <motion.div className="land-btns" initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6, delay: 0.3 }}>
          <button className="btn-login" onClick={() => navigate("/login")}>Login</button>
          <button className="btn-register" onClick={() => navigate("/register")}>Register</button>
        </motion.div>
      </nav>

      {/* ── HERO ── */}
      <section className="land-hero" id="home">
        <div className="hero-shapes">
          <div className="shape shape-1" />
          <div className="shape shape-2" />
          <div className="shape shape-3" />
          <div className="shape shape-4" />
          <div className="shape shape-5" />
        </div>

        <div className="hero-content">
          <div className="hero-text">
            <motion.div className="hero-badge" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
              ✦ AI-Powered Healthcare Fraud Detection
            </motion.div>

            <motion.h1 initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.2 }}>
              Protecting Healthcare
              <span className="hero-highlight"> Integrity</span>
              <br />with Intelligent AI
            </motion.h1>

            <motion.p className="hero-sub" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.4 }}>
              MediGuard AI helps hospital management detect fraudulent and
              suspicious transactions in real-time — keeping your healthcare
              system safe, transparent, and trustworthy.
            </motion.p>

            <motion.div className="hero-actions" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.6 }}>
              <button className="btn-primary" onClick={() => navigate("/register")}>
                Get Started <span>→</span>
              </button>
              <button className="btn-secondary" onClick={() => navigate("/login")}>
                Login to Dashboard
              </button>
            </motion.div>

            <motion.div className="hero-stats" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.8 }}>
              <div className="stat">
                <span className="stat-num">{counts.accuracy}%</span>
                <span className="stat-label">Detection Accuracy</span>
              </div>
              <div className="stat-divider" />
              <div className="stat">
                <span className="stat-num">{counts.transactions.toLocaleString()}+</span>
                <span className="stat-label">Transactions Analyzed</span>
              </div>
              <div className="stat-divider" />
              <div className="stat">
                <span className="stat-num">{counts.doctors}+</span>
                <span className="stat-label">Doctors Monitored</span>
              </div>
            </motion.div>
          </div>

          {/* Doctor Image */}
          <motion.div className="hero-image" initial={{ opacity: 0, x: 40 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.8, delay: 0.4 }}>
            <div className="hero-img-wrapper">
              <img src={doctorImg} alt="Doctor" className="doctor-img" />
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── ABOUT ── */}
      <section className="land-about" id="about">
        <motion.div className="section-header" initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.7 }}>
          <span className="section-tag">About Us</span>
          <h2>Smart Fraud Detection for <span className="hero-highlight">Hospital Management</span></h2>
          <p>
            MediGuard AI is built specifically for hospital management teams.
            Our system monitors transactions, detects anomalies, and flags
            suspicious activity automatically — so your team can focus on
            what matters most: patient care.
          </p>
        </motion.div>

        <div className="about-grid">
          {aboutCards.map((card, i) => (
            <motion.div
              key={i}
              className="about-card"
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              whileHover={{ y: -6, transition: { duration: 0.2 } }}
            >
              <div className="about-card-icon">{card.icon}</div>
              <h3>{card.title}</h3>
              <p>{card.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── CONTACT ── */}
      <section className="land-contact" id="contact">
        <motion.div className="section-header" initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.7 }}>
          <span className="section-tag">Contact Us</span>
          <h2>Get In <span className="hero-highlight">Touch</span></h2>
        </motion.div>

        <div className="contact-grid">
          {[
            { icon: "📧", title: "Email", text: "mediguard15@gmail.com", delay: 0.1 },
            { icon: "🏥", title: "Institution", text: "Final Year Project — Healthcare Fraud Detection", delay: 0.2 },
            { icon: "🔗", title: "GitHub", text: "github.com/mahnoorseemab", delay: 0.3 },
          ].map((item, i) => (
            <motion.div
              key={i}
              className="contact-card"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: item.delay }}
            >
              <div className="contact-icon">{item.icon}</div>
              <h4>{item.title}</h4>
              <p>{item.text}</p>
            </motion.div>
          ))}
        </div>

        <motion.div className="land-footer" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ duration: 0.7 }}>
          <p>© 2026 MediGuard AI. All Rights Reserved. | Healthcare Fraud Detection System</p>
        </motion.div>
      </section>

    </div>
  );
};

export default LandingPage;