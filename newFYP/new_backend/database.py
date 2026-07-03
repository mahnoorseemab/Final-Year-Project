# ============================================================
# DATABASE.PY — MySQL Connection
# Uses SQLAlchemy to connect to MySQL
# ============================================================

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# ─────────────────────────────────────────────
# CHANGE THESE TO YOUR MYSQL DETAILS
# ─────────────────────────────────────────────
MYSQL_USER     = "root"
MYSQL_PASSWORD = "1234"   # ← change this!
MYSQL_HOST     = "localhost"
MYSQL_PORT     = "3306"
MYSQL_DATABASE = "fraud_detection"

# ─────────────────────────────────────────────
# DATABASE URL
# ─────────────────────────────────────────────
DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)

# ─────────────────────────────────────────────
# CREATE ENGINE
# Engine = main connection to MySQL
# pool_pre_ping = auto reconnect if connection drops
# ─────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping = True,
    pool_recycle  = 3600
)

# ─────────────────────────────────────────────
# SESSION
# Session = one conversation with database
# Each request gets its own session
# ─────────────────────────────────────────────
SessionLocal = sessionmaker(
    autocommit = False,
    autoflush  = False,
    bind       = engine
)

# ─────────────────────────────────────────────
# BASE
# All table models will inherit from this
# ─────────────────────────────────────────────
Base = declarative_base()

# ─────────────────────────────────────────────
# GET DB SESSION
# Used in FastAPI endpoints
# Opens session → use → close automatically
# ─────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─────────────────────────────────────────────
# TEST CONNECTION
# Run this file directly to test connection
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        with engine.connect() as conn:
            print("✅ MySQL connection successful!")
            print(f"   Database : {MYSQL_DATABASE}")
            print(f"   Host     : {MYSQL_HOST}")
            print(f"   Port     : {MYSQL_PORT}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")