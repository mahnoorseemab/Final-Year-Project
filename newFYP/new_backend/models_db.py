# ============================================================
# MODELS_DB.PY — SQLAlchemy Table Definitions
# Defines exact structure of MySQL tables
# Must match tables created in MySQL Workbench!
#
# CHANGE: DoctorScore now has review_stars column.
#   Populated once by seed_review_stars.py from Reviews_new.csv.
#   Read by scoring_node to compute display_score.
# ============================================================

from sqlalchemy import (
    Column, Integer, String,
    Float, Text, DateTime,
    ForeignKey, Boolean
)
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


# ─────────────────────────────────────────────
# TABLE 1: transactions
# ─────────────────────────────────────────────
class Transaction(Base):
    __tablename__ = "transactions"

    id                  = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id          = Column(Integer,     nullable=True)
    doctor_id           = Column(String(50),  nullable=False, index=True)
    speciality_name     = Column(String(100), nullable=True)
    diagnosis           = Column(String(200), nullable=True)
    service_description = Column(String(200), nullable=True)
    service_id          = Column(String(50),  nullable=True)
    created_at          = Column(DateTime, default=datetime.utcnow)

    fraud_result = relationship(
        "FraudResult",
        back_populates = "transaction",
        uselist        = False
    )

    def __repr__(self):
        return (
            f"Transaction("
            f"id={self.id}, "
            f"doctor={self.doctor_id}, "
            f"patient={self.patient_id}, "
            f"specialty={self.speciality_name})"
        )


# ─────────────────────────────────────────────
# TABLE 2: fraud_results
# ─────────────────────────────────────────────
class FraudResult(Base):
    __tablename__ = "fraud_results"

    id             = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    patient_id     = Column(Integer,    nullable=True)
    doctor_id      = Column(String(50), nullable=True, index=True)
    ttsgan_score   = Column(Float,      nullable=True)
    dctgan_score   = Column(Float,      nullable=True)
    overall_risk   = Column(String(50), nullable=True)
    detected_at    = Column(DateTime,   default=datetime.utcnow)

    transaction = relationship(
        "Transaction",
        back_populates = "fraud_result"
    )

    def __repr__(self):
        return (
            f"FraudResult("
            f"id={self.id}, "
            f"doctor={self.doctor_id}, "
            f"ttsgan={self.ttsgan_score}, "
            f"dctgan={self.dctgan_score}, "
            f"risk={self.overall_risk})"
        )


# ─────────────────────────────────────────────
# TABLE 3: doctor_scores
#
# CHANGE: Added review_stars column (Float, default 5.0).
#
#   review_stars: average star rating from Reviews_new.csv
#     Range : 1.0 – 5.0
#     Source: seeded once by seed_review_stars.py at startup
#     Usage : scoring_node reads this to compute display_score
#             display_score = current_score × (review_stars / 5)
#
#   display_score is NOT stored here — it is computed live in
#   scoring_node each time, so it always reflects the latest
#   DTS and the stored review_stars together.
# ─────────────────────────────────────────────
class DoctorScore(Base):
    __tablename__ = "doctor_scores"

    id              = Column(Integer,    primary_key=True, index=True)
    doctor_id       = Column(String(50), nullable=False, unique=True, index=True)
    current_score   = Column(Float,      default=100.0)
    total_penalties = Column(Float,      default=0.0)
    fraud_count     = Column(Integer,    default=0)
    review_stars    = Column(Float,      default=5.0)
    # review_stars default is 5.0 (no penalty from missing data).
    # seed_review_stars.py overwrites this with real values from
    # Reviews_new.csv for all 118 known FYP doctors.
    last_updated    = Column(DateTime,   default=datetime.utcnow)

    def __repr__(self):
        return (
            f"DoctorScore("
            f"doctor={self.doctor_id}, "
            f"score={self.current_score}, "
            f"stars={self.review_stars})"
        )


# ─────────────────────────────────────────────
# TABLE 4: users
# ─────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id              = Column(Integer,    primary_key=True, index=True, autoincrement=True)
    full_name       = Column(String(100), nullable=False)
    email           = Column(String(150), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    role            = Column(String(50),  nullable=False, default="viewer")
    pmdc_number     = Column(String(50),  nullable=True)
    is_active       = Column(Boolean,     default=True)
    created_at      = Column(DateTime,    default=datetime.utcnow)
    updated_at      = Column(DateTime,    default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"User(id={self.id}, email={self.email}, role={self.role})"




# ─────────────────────────────────────────────
# TABLE 5: pmdc_numbers
# ─────────────────────────────────────────────
class PmdcNumber(Base):
    __tablename__ = "pmdc_numbers"

    doctor_id   = Column(String(20), primary_key=True)
    pmdc_number = Column(String(50), nullable=False)

    def __repr__(self):
        return f"PmdcNumber(doctor={self.doctor_id}, pmdc={self.pmdc_number})"
    

# ─────────────────────────────────────────────
# CREATE TABLES
# Run this file directly to create tables.
# Only creates if they don't exist.
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from database import engine
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully!")
    print("   ✅ transactions   table ready!")
    print("   ✅ fraud_results  table ready!")
    print("   ✅ doctor_scores  table ready!  (includes review_stars column)")
    print("   ✅ users          table ready!")
    print()
    print("Next step: run seed_review_stars.py to populate review_stars from Reviews_new.csv")
    print("   ✅ pmdc_numbers  table ready!")