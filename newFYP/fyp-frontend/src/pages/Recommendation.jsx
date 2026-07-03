import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import "../styles/Recommendation.css";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

const hospitalCoords = {
  "AFIC": [33.5964, 73.0447],
  "CMH": [33.5853, 73.0369],
  "MH": [33.5831, 73.0428],
  "Shifa International Hospital": [33.7215, 73.0635],
};

function MapCenter({ center, zoom }) {
  const map = useMap();
  useEffect(() => { map.setView(center, zoom); }, [center, zoom, map]);
  return null;
}

const Recommendation = () => {
  const [formData, setFormData] = useState({ speciality: "", service: "" });
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [location, setLocation] = useState(null);
  const [locationName, setLocationName] = useState("");
  const [mapCenter, setMapCenter] = useState([33.6400, 73.0480]);
  const [mapZoom, setMapZoom] = useState(12);
  const [error, setError] = useState("");
  const [specialities, setSpecialities] = useState([]);

  useEffect(() => {
    const fetchSpecialities = async () => {
      try {
        const res = await fetch("http://localhost:8000/recommendation-doctors");
        const data = await res.json();
        const doctors = data.doctors || [];
        const depts = [...new Set(doctors.map(d => d.Department || d.department).filter(Boolean))].sort();
        setSpecialities(depts);
      } catch (err) {
        console.error("Specialities fetch error:", err);
      }
    };
    fetchSpecialities();
  }, []);

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const { latitude, longitude } = pos.coords;
          setLocation([latitude, longitude]);
          setLocationName("Your Location");
        },
        () => {
          setLocation([33.6007, 73.0679]);
          setLocationName("Rawalpindi (Default)");
        }
      );
    }
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSearched(false);
    setError("");

    try {
      const patientId = localStorage.getItem("user_id") || "E1";

      const res = await fetch("http://localhost:8000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId,
          required_specialty: formData.speciality,
          max_fee: null,
          top_n: 5,
        })
      });

      const data = await res.json();

      if (!res.ok) {
        setError("Error: " + data.detail);
        setLoading(false);
        return;
      }

      const doctors = data.recommended_doctors || [];

      if (doctors.length === 0) {
        setRecommendations([]);
        setSearched(true);
        setLoading(false);
        return;
      }

      const ranked = doctors.map((doc, i) => {
        const hospitalName = doc.Hospital || doc.hospital || "-";
        const baseCoords = hospitalCoords[hospitalName];
        const offset = i * 0.0004;
        const coords = baseCoords
          ? [baseCoords[0] + offset, baseCoords[1] + offset]
          : [33.6400 + (Math.random() - 0.5) * 0.05,
          73.0480 + (Math.random() - 0.5) * 0.05];

        return {
          rank: i + 1,
          name: doc.Doctor_Name || doc.doctor_name || "-",
          hospital: hospitalName,
          speciality: doc.Department || doc.department || formData.speciality,
          service: formData.service,
          score: doc.hybrid_score
            ? (doc.hybrid_score * 100).toFixed(1)
            : (90 - i * 5).toFixed(1),
          fee: doc.Avg_Fee || doc.avg_fee || "-",
          rating: doc.Avg_Rating || doc.avg_rating || "-",
          coords,
        };
      });

      // Map center — agar Shifa wale doctors hain toh zoom out karo
      const hasShifa = ranked.some(d => d.hospital === "Shifa International Hospital");
      const hasRawalpindi = ranked.some(d =>
        ["AFIC", "CMH", "MH"].includes(d.hospital)
      );

      if (hasShifa && hasRawalpindi) {
        setMapCenter([33.6500, 73.0550]);
        setMapZoom(11);
      } else if (hasShifa) {
        setMapCenter([33.7215, 73.0635]);
        setMapZoom(14);
      } else {
        setMapCenter([33.5900, 73.0430]);
        setMapZoom(14);
      }

      setRecommendations(ranked);
      setSearched(true);

    } catch (err) {
      setError("Can't connect to server! Backend is running?");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getMedalColor = (rank) => {
    if (rank === 1) return "#f59e0b";
    if (rank === 2) return "#94a3b8";
    if (rank === 3) return "#d97706";
    return "#0ea5e9";
  };

  const getMedalEmoji = (rank) => {
    if (rank === 1) return "🥇";
    if (rank === 2) return "🥈";
    if (rank === 3) return "🥉";
    return `#${rank}`;
  };

  return (
    <div className="rec-wrapper">
      <div className="rec-shapes">
        <div className="rec-s rec-s1" />
        <div className="rec-s rec-s2" />
        <div className="rec-s rec-s3" />
        <div className="rec-s rec-s4" />
      </div>

      <div className="rec-container">

        {/* Header */}
        <motion.div className="rec-header"
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}>
          <div className="rec-logo">
            <div className="rec-logo-icon">🛡️</div>
            <span>Medi<span className="rec-ai">Guard AI</span></span>
          </div>
          <h1>Doctor <span className="rec-highlight">Recommendation</span></h1>
          <p>Enter your speciality — our AI will recommend the best nearby doctors</p>
          {locationName && (
            <div className="rec-location-badge">📍 {locationName}</div>
          )}
        </motion.div>

        {/* Form */}
        <motion.div className="rec-form-card"
          initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}>
          <form onSubmit={handleSubmit}>
            <div className="rec-form-row">
              <div className="rec-field">
                <label>Speciality</label>
                <input
                  type="text"
                  name="speciality"
                  list="speciality-list"
                  placeholder="Type or select speciality..."
                  value={formData.speciality}
                  onChange={handleChange}
                  required
                />
                <datalist id="speciality-list">
                  {specialities.map(s => <option key={s} value={s} />)}
                </datalist>
              </div>
              <div className="rec-field">
                <label>Service / Diagnosis</label>
                <input
                  type="text"
                  name="service"
                  placeholder="e.g. Heart Surgery, ECG..."
                  value={formData.service}
                  onChange={handleChange}
                />
              </div>
              <motion.button
                type="submit"
                className="rec-btn"
                whileHover={!loading ? { scale: 1.02 } : {}}
                whileTap={!loading ? { scale: 0.98 } : {}}
                disabled={loading}
              >
                {loading ? "Searching..." : "Find Doctors →"}
              </motion.button>
            </div>
          </form>
          {error && <p className="rec-error">{error}</p>}
        </motion.div>

        {/* Loading */}
        <AnimatePresence>
          {loading && (
            <motion.div className="rec-loading"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="rec-spinner" />
              <p>AI Agent analyzing best doctors nearby...</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence>
          {searched && !loading && (
            <motion.div className="rec-results-split"
              initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }} transition={{ duration: 0.6 }}>

              {/* Left - Doctor List */}
              <div className="rec-left-panel">
                <div className="rec-results-header">
                  <h2>Top Recommended Doctors</h2>
                  <span className="rec-tag">{formData.speciality}</span>
                </div>

                {recommendations.length === 0 ? (
                  <div className="rec-empty">
                    No doctors found for "{formData.speciality}". Try another speciality.
                  </div>
                ) : (
                  <div className="rec-cards">
                    {recommendations.map((doc, i) => (
                      <motion.div
                        key={i}
                        className={`rec-card ${doc.rank === 1 ? "rec-card-top" : ""}`}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.4, delay: i * 0.1 }}
                        whileHover={{ y: -3, transition: { duration: 0.2 } }}
                      >
                        <div className="rec-rank" style={{ color: getMedalColor(doc.rank) }}>
                          <span className="rec-medal">{getMedalEmoji(doc.rank)}</span>
                          <span className="rec-rank-label">Rank {doc.rank}</span>
                        </div>

                        <div className="rec-doc-info">
                          <div className="rec-avatar" style={{
                            borderColor: getMedalColor(doc.rank),
                            background: `${getMedalColor(doc.rank)}18`
                          }}>
                            <span style={{ color: getMedalColor(doc.rank) }}>
                              {doc.name.split(" ").map(w => w[0]).join("").slice(0, 2)}
                            </span>
                          </div>
                          <div>
                            <h3 className="rec-doc-name">{doc.name}</h3>
                            <p className="rec-doc-spec">🏥 {doc.hospital}</p>
                            <p className="rec-doc-spec">🩺 {doc.speciality}</p>
                          </div>
                        </div>

                        <div className="rec-doc-meta">
                          {doc.fee !== "-" && <span>💰 Rs. {doc.fee}</span>}
                          {doc.rating !== "-" && <span>⭐ {typeof doc.rating === 'number' ? doc.rating.toFixed(1) : doc.rating}/5</span>}
                        </div>

                        <div className="rec-score-wrap">
                          <span className="rec-score-label">AI Score</span>
                          <span className="rec-score" style={{ color: getMedalColor(doc.rank) }}>
                            {doc.score}%
                          </span>
                        </div>

                        <div className="rec-bar-bg">
                          <motion.div
                            className="rec-bar-fill"
                            style={{ background: getMedalColor(doc.rank) }}
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.min(doc.score, 100)}%` }}
                            transition={{ duration: 0.8, delay: i * 0.1 }}
                          />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}
              </div>

              {/* Right - Map */}
              <div className="rec-right-panel">
                <div className="rec-map-header">
                  <h2>📍 Nearby Hospitals</h2>
                  <p>Showing all recommended doctors on map</p>
                </div>
                <div className="rec-map-wrap">
                  <MapContainer
                    center={mapCenter}
                    zoom={mapZoom}
                    style={{ height: "100%", width: "100%", borderRadius: "12px" }}
                  >
                    <TileLayer
                      url="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
                      attribution='&copy; Google Maps'
                    />
                    <MapCenter center={mapCenter} zoom={mapZoom} />

                    {/* User location marker */}
                    {location && (
                      <Marker position={location}
                        icon={L.divIcon({
                          className: "",
                          html: `<div style="background:#0ea5e9;width:16px;height:16px;border-radius:50%;border:3px solid white;box-shadow:0 2px 8px rgba(0,0,0,0.3)"></div>`
                        })}>
                        <Popup>📍 Your Location</Popup>
                      </Marker>
                    )}

                    {/* All doctor markers */}
                    {recommendations.map((doc, i) => (
                      <Marker
                        key={i}
                        position={doc.coords}
                        icon={L.divIcon({
                          className: "",
                          html: `<div style="background:${getMedalColor(doc.rank)};color:white;width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:13px;border:2px solid white;box-shadow:0 2px 8px rgba(0,0,0,0.35)">${doc.rank}</div>`
                        })}
                      >
                        <Popup>
                          <strong>{doc.name}</strong><br />
                          🏥 {doc.hospital}<br />
                          🩺 {doc.speciality}<br />
                          ⭐ {doc.rating}/5<br />
                          💰 Rs. {doc.fee}
                        </Popup>
                      </Marker>
                    ))}
                  </MapContainer>
                </div>
              </div>

            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </div>
  );
};

export default Recommendation;