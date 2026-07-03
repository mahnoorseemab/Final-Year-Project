# ============================================================
# CRUD.PY — Database Operations
#
# CHANGES IN THIS VERSION:
#
#   1. handle_cold_start_10: removed dead 'db' parameter
#      It was never used inside the function — callers no
#      longer need to pass db to it.
#
#   2. FOUR NEW QUERY FUNCTIONS for multi-perspective inference:
#
#      Training built sequences from 4 groupings:
#        - DOCTOR_ID     (3628 sequences)
#        - SPECIALITY_ID (4042 sequences)
#        - SERVICE_ID    (2672 sequences)
#        - DIAGNOSIS_ID  (3561 sequences)
#
#      Old backend only fetched doctor sequences → only 26% of
#      what the model was trained on. Now we fetch all 4 so
#      inference matches training exactly.
#
#      New functions:
#        get_last_10_by_speciality(db, speciality_name)
#        get_last_10_by_service(db, service_id)
#        get_last_10_by_diagnosis(db, diagnosis)
#        get_last_10_transactions(db, doctor_id)  ← unchanged
#
#   3. UNCHANGED: all auth, save, score, fraud history functions
# ============================================================
from sqlalchemy.orm import Session
from sqlalchemy import desc
from models_db import Transaction, FraudResult, DoctorScore, User
from datetime import datetime
from passlib.context import CryptContext

# ─────────────────────────────────────────────
# PASSWORD HASHING SETUP
# ─────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ─────────────────────────────────────────────
# AUTH — Hash Password
# ─────────────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# ─────────────────────────────────────────────
# AUTH — Verify Password
# ─────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ─────────────────────────────────────────────
# AUTH — Register New User
# Returns (user, None) on success
# Returns (None, error_message) on failure
# ─────────────────────────────────────────────
def create_user(
    db          : Session,
    full_name   : str,
    email       : str,
    password    : str,
    role        : str,
    pmdc_number: str = None
):
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return None, "Email already registered"
 
    if role == "doctor":
        if not pmdc_number:
            return None, "PMDC number is required for doctors"
        from models_db import PmdcNumber
        valid = db.query(PmdcNumber).filter(
            PmdcNumber.pmdc_number == pmdc_number
        ).first()
        if not valid:
            return None, "Invalid PMDC number. Not found in registry."
    
    user = User(
        full_name       = full_name,
        email           = email,
        hashed_password = hash_password(password),
        role            = role,
       pmdc_number     = pmdc_number,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    print(f"✅ New user registered: {email} (role={role})")
    return user, None


# ─────────────────────────────────────────────
# AUTH — Login / Authenticate User
# Returns (user, None) on success
# Returns (None, error_message) on failure
# ─────────────────────────────────────────────
def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None, "Email not found"

    if not verify_password(password, user.hashed_password):
        return None, "Incorrect password"

    if not user.is_active:
        return None, "Account is deactivated. Contact your administrator."

    print(f"✅ User logged in: {email} (role={user.role})")
    return user, None


# ─────────────────────────────────────────────
# CREATE — Save New Transaction
# ─────────────────────────────────────────────
def save_transaction(db: Session, transaction_data: dict):
    new_transaction = Transaction(
        patient_id          = transaction_data.get('PATIENT_ID'),
        doctor_id           = transaction_data.get('DOCTOR_ID'),
        speciality_name     = transaction_data.get('SPECIALITY_NAME'),
        diagnosis           = transaction_data.get('DIAGNOSIS'),
        service_description = transaction_data.get('SERVICE_DESCRIPTION'),
        service_id          = transaction_data.get('SERVICE_ID'),
        created_at          = datetime.utcnow()
    )

    db.add(new_transaction)
    db.commit()
    db.refresh(new_transaction)

    print(f"✅ Transaction saved! ID: {new_transaction.id}")
    return new_transaction


# ─────────────────────────────────────────────
# READ — Fetch Last 10 Transactions by DOCTOR_ID
#
# Used for GROUPING 1: Doctor billing behaviour
# Sequence = 10 consecutive bills by the same doctor
# Fraud signal: doctor suddenly bills unrelated specialties
# ─────────────────────────────────────────────
def get_last_10_transactions(db: Session, doctor_id: str):
    transactions = (
        db.query(Transaction)
        .filter(Transaction.doctor_id == doctor_id)
        .order_by(desc(Transaction.id))
        .limit(10)
        .all()
    )
    print(f"📋 [DOCTOR grouping]    Found {len(transactions)} transactions for doctor={doctor_id}")
    return transactions


# ─────────────────────────────────────────────
# READ — Fetch Last 10 Transactions by SPECIALITY
# Used for GROUPING 2: Specialty billing patterns
# Sequence = 10 consecutive bills under the same specialty
# Fraud signal: wrong services appearing inside a specialty
# e.g. Cardiology suddenly billing Pediatric services
# ─────────────────────────────────────────────
def get_last_10_by_speciality(db: Session, speciality_name: str):
    transactions = (
        db.query(Transaction)
        .filter(Transaction.speciality_name == speciality_name)
        .order_by(desc(Transaction.id))
        .limit(10)
        .all()
    )
    print(f"📋 [SPECIALITY grouping] Found {len(transactions)} transactions for speciality={speciality_name}")
    return transactions


# ─────────────────────────────────────────────
# READ — Fetch Last 10 Transactions by SERVICE_ID
#
# Used for GROUPING 3: Service usage patterns
# Sequence = 10 consecutive bills for the same service
# Fraud signal: expensive service billed by unrelated doctors
# or service appearing in wrong specialty context
# ─────────────────────────────────────────────
def get_last_10_by_service(db: Session, service_id: str):
    transactions = (
        db.query(Transaction)
        .filter(Transaction.service_id == service_id)
        .order_by(desc(Transaction.id))
        .limit(10)
        .all()
    )
    print(f"📋 [SERVICE grouping]   Found {len(transactions)} transactions for service={service_id}")
    return transactions

# ─────────────────────────────────────────────
# READ — Fetch Last 10 Transactions by DIAGNOSI
# Used for GROUPING 4: Diagnosis-service patterns
# Sequence = 10 consecutive bills with the same diagnosis
# Fraud signal: wrong services billed for a diagnosis
# e.g. Diabetes diagnosis billed with orthopedic surgery
# ─────────────────────────────────────────────
def get_last_10_by_diagnosis(db: Session, diagnosis: str):
    transactions = (
        db.query(Transaction)
        .filter(Transaction.diagnosis == diagnosis)
        .order_by(desc(Transaction.id))
        .limit(10)
        .all()
    )
    print(f"📋 [DIAGNOSIS grouping] Found {len(transactions)} transactions for diagnosis={diagnosis}")
    return transactions


# ─────────────────────────────────────────────
# READ — Fetch Last 3 Transactions by Doctor
# Kept for backward compatibility only
# ─────────────────────────────────────────────
def get_last_3_transactions(db: Session, doctor_id: str):
    transactions = (
        db.query(Transaction)
        .filter(Transaction.doctor_id == doctor_id)
        .order_by(desc(Transaction.id))
        .limit(3)
        .all()
    )
    print(f"📋 Found {len(transactions)} transactions for {doctor_id}")
    return transactions


# ─────────────────────────────────────────────
# READ — Get Transaction Count by Doctor
# ─────────────────────────────────────────────
def get_doctor_transaction_count(db: Session, doctor_id: str):
    count = (
        db.query(Transaction)
        .filter(Transaction.doctor_id == doctor_id)
        .count()
    )
    return count


# ─────────────────────────────────────────────
# CREATE — Save Fraud Detection Result
# ─────────────────────────────────────────────
def save_fraud_result(db: Session, result_data: dict):
    new_result = FraudResult(
        transaction_id = result_data.get('transaction_id'),
        patient_id     = result_data.get('patient_id'),
        doctor_id      = result_data.get('doctor_id'),
        ttsgan_score   = result_data.get('ttsgan_score'),
        dctgan_score   = result_data.get('dctgan_score'),
        overall_risk   = result_data.get('overall_risk'),
        detected_at    = datetime.utcnow()
    )

    db.add(new_result)
    db.commit()
    db.refresh(new_result)

    print(f"✅ Fraud result saved! Risk: {new_result.overall_risk}")
    return new_result


# ─────────────────────────────────────────────
# READ — Get All Fraud Results by Doctor
# ─────────────────────────────────────────────
def get_doctor_fraud_history(db: Session, doctor_id: str):
    results = (
        db.query(FraudResult)
        .filter(FraudResult.doctor_id == doctor_id)
        .order_by(desc(FraudResult.detected_at))
        .all()
    )
    return results


# ─────────────────────────────────────────────
# READ — Get All Doctor Scores
# Used by Agent 3 to pass scores to Agent 2
# ─────────────────────────────────────────────
def get_all_doctor_scores(db: Session) -> list:
    rows = (
        db.query(DoctorScore)
        .order_by(desc(DoctorScore.current_score))
        .all()
    )
    result = []
    for row in rows:
        result.append({
            'doctor_id'    : row.doctor_id,
            'current_score': row.current_score,
            'fraud_count'  : row.fraud_count,
            'review_stars' : float(row.review_stars) if row.review_stars is not None else 5.0,
            'last_updated' : str(row.last_updated) if row.last_updated else None,
        })
    print(f"📋 Fetched {len(result)} doctor scores from DB")
    return result


# ─────────────────────────────────────────────
# READ — Get Single Doctor Score by ID
# ─────────────────────────────────────────────
def get_doctor_score(db: Session, doctor_id: str) -> dict:
    row = (
        db.query(DoctorScore)
        .filter(DoctorScore.doctor_id == doctor_id)
        .first()
    )
    if row:
        return {
            'doctor_id'    : row.doctor_id,
            'current_score': row.current_score,
            'fraud_count'  : row.fraud_count,
            'review_stars' : float(row.review_stars) if row.review_stars is not None else 5.0,
            'last_updated' : str(row.last_updated) if row.last_updated else None,
        }
    return None


# ─────────────────────────────────────────────
# HELPER — Build Sequence from ORM Transactions
# Converts DB Transaction objects → list of dicts
# with ID fields (for encoding in inference)
# ─────────────────────────────────────────────
def build_sequence_from_transactions(transactions: list):
    sequence = []
    for t in transactions:
        sequence.append({
            'SPECIALITY_NAME'    : t.speciality_name     or 'Unknown',
            'DIAGNOSIS'          : t.diagnosis            or 'Unknown',
            'SERVICE_DESCRIPTION': t.service_description  or 'Unknown',
            'SERVICE_ID'         : t.service_id           or 'Unknown',
            'DOCTOR_ID'          : t.doctor_id            or 'Unknown',
        })
    return sequence


# ─────────────────────────────────────────────
# HELPER — Cold Start for SEQ_LEN=10
#
# CHANGE: removed dead 'db' parameter
# It was never used inside this function.
# Old signature: handle_cold_start_10(db, doctor_id, transactions)
# New signature: handle_cold_start_10(doctor_id, transactions)
#
# If doctor/specialty/service/diagnosis has fewer than 10
# records in DB, we cycle what we have to fill up to 10.
# is_cold_start=True is returned so caller knows padding happened.
# ─────────────────────────────────────────────
def handle_cold_start_10(doctor_id: str, transactions: list):
    count = len(transactions)

    if count >= 10:
        return transactions[:10], False

    print(f"⚠️  Cold start: '{doctor_id}' has only {count} transactions")
    print(f"   Filling {10 - count} slots with cycling...")

    if count == 0:
        print(f"   Brand new entry — only current transaction available")
        return transactions, True

    filled = list(transactions)
    while len(filled) < 10:
        filled.append(transactions[len(filled) % count])

    print(f"   Filled to 10 transactions using cycling!")
    return filled[:10], True


# ─────────────────────────────────────────────
# HELPER — Cold Start for SEQ_LEN=3
# Kept for backward compatibility — do not remove
# ─────────────────────────────────────────────
def handle_cold_start(
    db          : Session,
    doctor_id   : str,
    transactions: list
):
    count = len(transactions)

    if count >= 3:
        return transactions[:3], False

    print(f"⚠️  Cold start: {doctor_id} has only {count} transactions")

    if count == 0:
        return transactions, True

    filled = list(transactions)
    while len(filled) < 3:
        filled.append(transactions[len(filled) % count])

    return filled[:3], True


# ─────────────────────────────────────────────
# OTP — In-Memory Store
# ─────────────────────────────────────────────
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

_otp_store = {}  # { email: { "otp": "123456", "user_data": {...} } }

GMAIL_SENDER   = "mediguard15@gmail.com"
GMAIL_APP_PASS = "bnem nhzs yxia swzw"


# ─────────────────────────────────────────────
# OTP — Generate 6-digit OTP
# ─────────────────────────────────────────────
def generate_otp() -> str:
    return str(random.randint(100000, 999999))


# ─────────────────────────────────────────────
# OTP — Send OTP via Gmail
# Returns True on success, False on failure
# ─────────────────────────────────────────────
def send_otp_email(to_email: str, otp: str) -> bool:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "MediGuard AI — Your OTP Code"
        msg["From"]    = GMAIL_SENDER
        msg["To"]      = to_email

        html = f"""
        <html><body style="font-family: Arial, sans-serif; background:#f4f4f4; padding:30px;">
          <div style="max-width:480px; margin:auto; background:white; border-radius:10px; padding:30px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
            <h2 style="color:#0a6e6e;">MediGuard AI</h2>
            <p>Your One-Time Password (OTP) for registration is:</p>
            <h1 style="letter-spacing:8px; color:#0a6e6e; text-align:center;">{otp}</h1>
            <p style="color:#888; font-size:13px;">This OTP is valid for <strong>10 minutes</strong>. Do not share it with anyone.</p>
          </div>
        </body></html>
        """

        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_SENDER, GMAIL_APP_PASS)
            server.sendmail(GMAIL_SENDER, to_email, msg.as_string())

        print(f"✅ OTP sent to {to_email}")
        return True

    except Exception as e:
        print(f"❌ Failed to send OTP to {to_email}: {e}")
        return False


# ─────────────────────────────────────────────
# OTP — Store OTP + user data temporarily
# ─────────────────────────────────────────────
def store_otp(email: str, otp: str, user_data: dict):
    _otp_store[email] = {
        "otp"      : otp,
        "user_data": user_data,
    }
    print(f"📦 OTP stored for {email}")


# ─────────────────────────────────────────────
# OTP — Verify OTP and return user data
# Returns (user_data, None) on success
# Returns (None, error_message) on failure
# ─────────────────────────────────────────────
def verify_otp(email: str, otp: str):
    entry = _otp_store.get(email)

    if not entry:
        return None, "OTP expired or not found. Please register again."

    if entry["otp"] != otp:
        return None, "Incorrect OTP. Please try again."

    user_data = entry["user_data"]
    del _otp_store[email]  # OTP use ho gaya, delete karo

    print(f"✅ OTP verified for {email}")
    return user_data, None