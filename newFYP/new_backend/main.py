# ============================================================
# main.py — FastAPI Backend
# VERSION 13.0.0 — Clean API Separation
#
# WHAT CHANGED FROM v12:
#
# PROBLEM IN v12:
#   /analyze required BOTH TransactionInput AND PatientInput
#   together in one request body (AnalyzeRequest).
#   This made no sense — a doctor submitting a billing transaction
#   has nothing to do with a patient asking for a recommendation.
#   They are two completely separate workflows by different users.
#
# FIX IN v13 — Two separate endpoints, each with its own form:
#
#   POST /transaction
#     Who uses it: Doctor, Admin, Staff, Auditor
#     What they fill: TransactionInput (billing details only)
#     What runs: Agent 3 → Agent 1 only (fraud detection)
#     Returns: fraud verdict, GAN scores, investigation report,
#              doctor trust score update
#
#   POST /recommend
#     Who uses it: Patient only
#     What they fill: PatientInput (specialty, budget, top_n)
#     What runs: Agent 3 → Agent 2 only (doctor recommendation)
#     Returns: ranked doctor list, RL top pick, fraud warning
#
#   POST /detect-fraud → REMOVED (was redundant with /transaction)
#   POST /analyze      → REMOVED (was wrong: forced both forms together)
#
# TRAINING CHANGE (notebook v2):
#   resolve_transaction() already passes SPECIALITY_NAME and
#   DIAGNOSIS through the resolved dict for cross feature
#   computation in the new inference pipeline. No change needed.
#
# UNCHANGED:
#   POST /register, POST /login, POST /feedback
#   GET  /, /health, /specialties, /services, /diagnoses
#   GET  /doctor-scores, /recommendation-doctors
# ============================================================

from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
import uvicorn
import os

from database  import engine, get_db
from models_db import Base
from crud      import save_transaction, save_fraud_result, get_all_doctor_scores

Base.metadata.create_all(bind=engine)

REVIEWS_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Reviews_new.csv"
)

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "FYPDATA.csv"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 55)
    print("Healthcare Fraud Detection System v13.0.0")
    print("=" * 55)

    try:
        from id_mappings import get_maps
        get_maps(CSV_PATH)
        print("✅ ID mapping tables loaded!")
    except Exception as e:
        print(f"❌ ID mapping failed: {e}")

    try:
        from ttsgan_loader import load_ttsgan
        load_ttsgan()
        print("✅ TTS-WGAN loaded!")
    except Exception as e:
        print(f"❌ TTS-WGAN failed: {e}")

    try:
        from dctgan_loader import load_dctgan
        load_dctgan()
        print("✅ DCT-GAN loaded!")
    except Exception as e:
        print(f"❌ DCT-GAN failed: {e}")

    try:
        from agent1.agent1_graph import agent1_graph
        print("✅ Agent 1 ready!")
    except Exception as e:
        print(f"❌ Agent 1 failed: {e}")

    try:
        from agent2.agent2_graph import initialize_agent2_cache
        initialize_agent2_cache(REVIEWS_CSV_PATH)
        print("✅ Agent 2 (Recommendation) ready!")
    except Exception as e:
        print(f"❌ Agent 2 failed: {e}")

    try:
       from agent3.agent3_graph import agent3_transaction_graph, agent3_recommend_graph
       print("✅ Agent 3 (Orchestration) ready!")
    except Exception as e:
        print(f"❌ Agent 3 failed: {e}")

    print("=" * 55)
    print("API: http://localhost:8000")
    print("=" * 55)
    yield

app = FastAPI(
    title    = "Healthcare Fraud Detection API",
    version  = "13.0.0",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    full_name    : str
    email        : str
    password     : str
    role         : str   # "admin" | "doctor" | "auditor" | "viewer"
    pmdc_number : Optional[str] = None


class LoginRequest(BaseModel):
    email    : str
    password : str


class TransactionInput(BaseModel):
    """
    Billing transaction submitted by Doctor / Admin / Staff / Auditor.
    Human-readable names only — IDs resolved internally.
    DOCTOR_ID stays as 'doc_01' format.
    DIAGNOSIS is optional — None means no diagnosis recorded.
    """
    PATIENT_ID          : int
    DOCTOR_ID           : str
    SPECIALITY_NAME     : str
    SERVICE_DESCRIPTION : str
    DIAGNOSIS           : Optional[str] = None


class PatientInput(BaseModel):
    """
    Patient's doctor recommendation request.
    Completely independent of billing transactions.
    Patient fills this form — no transaction details needed.
    """
    patient_id         : str
    required_specialty : Optional[str] = ""
    max_fee            : Optional[int] = None
    top_n              : Optional[int] = 5


class FeedbackRequest(BaseModel):
    doctor_name   : str
    actual_rating : float  # 1.0 to 5.0


# ══════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register step 1 — validate and send OTP."""
    from crud import create_user, generate_otp, send_otp_email, store_otp

    if len(request.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    valid_roles = {"admin", "doctor", "auditor", "viewer", "patient", "staff"}
    if request.role not in valid_roles:
        raise HTTPException(400, f"Role must be one of: {sorted(valid_roles)}")

    if "@" not in request.email:
        raise HTTPException(400, "Please enter a valid email address")

    # Check if email already registered
    from crud import pwd_context
    from models_db import User
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(400, "Email already registered")

    # Generate and send OTP
    otp = generate_otp()
    sent = send_otp_email(request.email, otp)
    if not sent:
        raise HTTPException(500, "Failed to send OTP email. Please check your email address.")

    # Store OTP + user data temporarily
    store_otp(request.email, otp, {
        "full_name"   : request.full_name,
        "email"       : request.email,
        "password"    : request.password,
        "role"        : request.role,
        "pmdc_number": request.pmdc_number or "",
    })

    return {
        "status" : "otp_sent",
        "message": f"OTP sent to {request.email}. Please verify to complete registration.",
        "email"  : request.email,
    }

@app.post("/verify-otp")
def verify_otp_endpoint(
    payload: dict,
    db: Session = Depends(get_db)
):
    """Register step 2 — verify OTP and create account."""
    from crud import verify_otp, create_user

    email = payload.get("email", "").strip()
    otp   = payload.get("otp", "").strip()

    if not email or not otp:
        raise HTTPException(400, "Email and OTP are required")

    user_data, error = verify_otp(email, otp)
    if error:
        raise HTTPException(400, error)

    # Now create the actual user account
    user, err = create_user(
        db           = db,
        full_name    = user_data["full_name"],
        email        = user_data["email"],
        password     = user_data["password"],
        role         = user_data["role"],
        pmdc_number = user_data["pmdc_number"],
    )

    if err:
        raise HTTPException(400, err)

    return {
        "status"   : "success",
        "message"  : "Account created successfully! You can now log in.",
        "user_id"  : user.id,
        "full_name": user.full_name,
        "email"    : user.email,
        "role"     : user.role,
    }

@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate a user and return their profile."""
    from crud import authenticate_user

    if not request.email or not request.password:
        raise HTTPException(400, "Email and password are required")

    user, error = authenticate_user(db, request.email, request.password)
    if error:
        raise HTTPException(401, error)

    return {
        "status"   : "success",
        "message"  : f"Welcome back, {user.full_name}!",
        "user_id"  : user.id,
        "full_name": user.full_name,
        "email"    : user.email,
        "role"     : user.role,
    }


# ══════════════════════════════════════════════════════════════
# INFO / LOOKUP ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "system" : "Healthcare Fraud Detection & Doctor Recommendation",
        "version": "13.0.0",
        "endpoints": {
            "POST /transaction": "Doctor/Admin/Staff — submit billing for fraud detection → Agent 1",
            "POST /recommend"  : "Patient — get doctor recommendations → Agent 2",
            "POST /feedback"   : "Patient — rate a recommended doctor (RL update)",
            "POST /register"   : "Create new user account",
            "POST /login"      : "Authenticate user",
        },
        "lookup_endpoints": {
            "GET /specialties"            : "All specialty names for dropdown",
            "GET /services"               : "All service descriptions for dropdown",
            "GET /diagnoses"              : "All diagnosis names for dropdown",
            "GET /doctor-scores"          : "All trust scores (admin dashboard)",
            "GET /recommendation-doctors" : "All doctors in recommendation pool",
        }
    }


@app.get("/health")
def health():
    import ttsgan_loader, dctgan_loader
    from agent2.agent2_graph import _agent2_ready
    return {
        "status": "online",
        "ttsgan": "loaded" if ttsgan_loader.ttsgan_model else "not loaded",
        "dctgan": "loaded" if dctgan_loader.dctgan_model else "not loaded",
        "agent2": "ready"  if _agent2_ready else "not loaded",
    }


@app.get("/doctor-scores")
def get_doctor_scores(db: Session = Depends(get_db)):
    """All fraud trust scores — for admin dashboard."""
    scores = get_all_doctor_scores(db)
    return {"total_doctors": len(scores), "scores": scores}


@app.get("/specialties")
def get_specialties():
    """All specialty names — for transaction form dropdown."""
    from id_mappings import get_all_specialties
    return {"specialties": get_all_specialties(CSV_PATH)}

@app.get("/services")
def get_services():
    """All service descriptions — for transaction form dropdown."""
    from id_mappings import get_all_services
    return {"services": get_all_services(CSV_PATH)}

@app.get("/services-by-specialty")
def get_services_by_specialty(specialty: str):
    """Services filtered by specialty — for transaction form."""
    from id_mappings import get_services_by_specialty
    return {"services": get_services_by_specialty(specialty)}

@app.get("/diagnoses")
def get_diagnoses():
    """All diagnosis names — for transaction form dropdown."""
    from id_mappings import get_all_diagnoses
    return {"diagnoses": get_all_diagnoses(CSV_PATH)}


@app.get("/recommendation-doctors")
def get_recommendation_doctors():
    """All doctors in recommendation pool — info endpoint."""
    from agent2.agent2_graph import get_all_recommendation_doctors
    doctors = get_all_recommendation_doctors()
    return {"total": len(doctors), "doctors": doctors}


# ── Helper: convert names → IDs for GAN ──────────────────────

def resolve_transaction(tx: TransactionInput) -> dict:
    """
    Converts TransactionInput (human names) into a flat dict
    with resolved IDs (for GAN) + original names (for reports/DB/
    cross feature computation in new inference pipeline).
    """
    from id_mappings import names_to_ids

    resolved = names_to_ids(
        doctor_id       = tx.DOCTOR_ID,
        speciality_name = tx.SPECIALITY_NAME,
        service_desc    = tx.SERVICE_DESCRIPTION,
        diagnosis       = tx.DIAGNOSIS,
        patient_id      = tx.PATIENT_ID,
        csv_path        = CSV_PATH,
    )

    return {
        # IDs — used by GAN inference
        'DOCTOR_ID'    : resolved['DOCTOR_ID'],
        'SPECIALITY_ID': resolved['SPECIALITY_ID'],
        'SERVICE_ID'   : resolved['SERVICE_ID'],
        'DIAGNOSIS_ID' : resolved['DIAGNOSIS_ID'],
        'PATIENT_ID'   : resolved['PATIENT_ID'],

        # Names — for cross features + DB + reports
        'SPECIALITY_NAME'    : tx.SPECIALITY_NAME,
        'SERVICE_DESCRIPTION': tx.SERVICE_DESCRIPTION,
        'DIAGNOSIS'          : resolved['_names']['diagnosis'],

        '_mapping_warnings'  : resolved['_warnings'],
    }


# ══════════════════════════════════════════════════════════════
# RL FEEDBACK
# ══════════════════════════════════════════════════════════════

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """
    Patient submits rating after visiting a recommended doctor.
    RL agent updates Q-value for better future recommendations.
    """
    if not (1.0 <= request.actual_rating <= 5.0):
        raise HTTPException(400, "Rating must be 1.0 to 5.0")
    try:
        from agent2.agent2_graph import update_rl_feedback
        update_rl_feedback(request.doctor_name, request.actual_rating)
        return {
            "status" : "success",
            "doctor" : request.doctor_name,
            "rating" : request.actual_rating,
            "message": "RL agent updated. Next recommendations will reflect this feedback."
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════
# ENDPOINT 1: /transaction — Agent 3 → Agent 1 only
# Who uses it: Doctor, Admin, Staff, Auditor
# ══════════════════════════════════════════════════════════════

@app.post("/transaction")
def submit_transaction(transaction: TransactionInput, db: Session = Depends(get_db)):
    """
    Submit a billing transaction for fraud detection.

    WHO USES THIS: Doctor, Admin, Staff, Auditor
    WHAT RUNS:     Agent 3 → Agent 1 (fraud detection only)
                   No recommendation runs here.

    Body: TransactionInput
      PATIENT_ID, DOCTOR_ID, SPECIALITY_NAME,
      SERVICE_DESCRIPTION, DIAGNOSIS (optional)

    Returns:
      overall_risk     → HIGH RISK / MEDIUM RISK / NORMAL
      is_fraud         → true / false
      ttsgan / dctgan  → individual GAN scores and verdicts
      voting           → how the two models voted
      agent1.report    → full investigation report
      doctor_score     → updated trust score for this doctor
      + fraud investigation details if is_fraud = true
    """
    try:
        tx_data = resolve_transaction(transaction)

        print(f"\n{'='*55}")
        print(f"[/transaction] Doctor:{tx_data['DOCTOR_ID']}")
        print(f"  Specialty : {tx_data['SPECIALITY_NAME']} (ID={tx_data['SPECIALITY_ID']})")
        print(f"  Service   : {tx_data['SERVICE_DESCRIPTION']} (ID={tx_data['SERVICE_ID']})")
        print(f"  Diagnosis : {tx_data['DIAGNOSIS']} (ID={tx_data['DIAGNOSIS_ID']})")
        print(f"{'='*55}")

        # Save transaction to DB
        saved = save_transaction(db, {
            'PATIENT_ID'         : tx_data['PATIENT_ID'],
            'DOCTOR_ID'          : tx_data['DOCTOR_ID'],
            'SPECIALITY_NAME'    : tx_data['SPECIALITY_NAME'],
            'DIAGNOSIS'          : tx_data['DIAGNOSIS'],
            'SERVICE_DESCRIPTION': tx_data['SERVICE_DESCRIPTION'],
            'SERVICE_ID'         : tx_data['SERVICE_ID'],
        })

        # Run Agent 3 in transaction-only mode → Agent 1 only
        from agent3.agent3_graph import run_agent3_transaction
        result = run_agent3_transaction(transaction=tx_data, db=db)

        overall_risk = result.get('overall_risk', 'UNKNOWN')
        is_fraud     = result.get('is_fraud', False)
        tts          = result.get('ttsgan_result', {})
        dct          = result.get('dctgan_result', {})
        votes        = result.get('risk_votes', {})

        save_fraud_result(db, {
            "transaction_id": saved.id,
            "patient_id"    : transaction.PATIENT_ID,
            "doctor_id"     : transaction.DOCTOR_ID,
            "ttsgan_score"  : tts.get('score'),
            "dctgan_score"  : dct.get('score'),
            "overall_risk"  : overall_risk
        })

        resp = {
            "transaction_id": saved.id,
            "patient_id"    : transaction.PATIENT_ID,
            "doctor_id"     : transaction.DOCTOR_ID,

            "transaction_details": {
                "speciality_name"    : tx_data['SPECIALITY_NAME'],
                "service_description": tx_data['SERVICE_DESCRIPTION'],
                "diagnosis"          : tx_data['DIAGNOSIS'],
                "speciality_id"      : tx_data['SPECIALITY_ID'],
                "service_id"         : tx_data['SERVICE_ID'],
                "diagnosis_id"       : tx_data['DIAGNOSIS_ID'],
            },

            "overall_risk" : overall_risk,
            "is_fraud"     : is_fraud,
            "ttsgan"       : tts,
            "dctgan"       : dct,

            "voting": {
                "tts_vote"         : votes.get('tts_vote'),
                "dct_vote"         : votes.get('dct_vote'),
                "high_risk_votes"  : votes.get('high_risk', 0),
                "medium_risk_votes": votes.get('medium_risk', 0),
                "normal_votes"     : votes.get('normal', 0),
            },

            "doctor_score": {
                "doctor_id"    : transaction.DOCTOR_ID,
                "updated_score": result.get('updated_doctor_score', 100.0),
                "score_status" : result.get('score_breakdown', {}).get('score_status', 'UNKNOWN'),
            },

            "agent1": {
                "report": result.get('report')
            },

            "mapping_warnings": tx_data.get('_mapping_warnings', []),
        }

        if is_fraud:
            resp["agent1"].update({
                "responsible_party" : result.get('responsible_party'),
                "primary_fraud_type": result.get('primary_fraud_type'),
                "fraud_types"       : result.get('fraud_types'),
                "confidence"        : result.get('confidence'),
                "reasons"           : result.get('reasons'),
                "overbilling_flag"  : result.get('overbilling_flag'),
                "overbilling_note"  : result.get('overbilling_note'),
                "doctor_score_detail": {
                    "penalty"      : result.get('doctor_score_penalty'),
                    "updated_score": result.get('updated_doctor_score'),
                    "breakdown"    : result.get('score_breakdown'),
                },
            })

        return resp

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════
# ENDPOINT 2: /recommend — Agent 3 → Agent 2 only
# Who uses it: Patient only
# ══════════════════════════════════════════════════════════════

@app.post("/recommend")
def get_recommendation(patient_input: PatientInput, db: Session = Depends(get_db)):
    """
    Get doctor recommendations for a patient.

    WHO USES THIS: Patient only
    WHAT RUNS:     Agent 3 → Agent 2 (recommendation only)
                   No fraud detection runs here.

    Body: PatientInput
      patient_id          → patient identifier (e.g. 'E1')
      required_specialty  → specialty needed (optional filter)
      max_fee             → maximum fee (optional filter)
      top_n               → number of doctors to return (default 5)

    Returns:
      recommended_doctors → ranked list with hybrid scores
      rl_top_pick         → RL agent's single best recommendation
      recommendation_note → explanation of ranking
      fraud_warning       → warns if a recommended doctor has low trust score
    """
    try:
        patient_data = patient_input.dict()

        print(f"\n{'='*55}")
        print(f"[/recommend] Patient:{patient_data['patient_id']}")
        print(f"  Specialty : {patient_data.get('required_specialty', 'any')}")
        print(f"  Max fee   : {patient_data.get('max_fee', 'no limit')}")
        print(f"  Top N     : {patient_data.get('top_n', 5)}")
        print(f"{'='*55}")

        # Run Agent 3 in recommendation-only mode → Agent 2 only
        from agent3.agent3_graph import run_agent3_recommend
        result = run_agent3_recommend(patient_input=patient_data, db=db)

        return {
            "patient_id"         : patient_data.get('patient_id'),
            "required_specialty" : patient_data.get('required_specialty', ''),
            "max_fee"            : patient_data.get('max_fee'),
            "recommended_doctors": result.get('recommended_doctors', []),
            "rl_top_pick"        : result.get('rl_top_pick', ''),
            "recommendation_note": result.get('recommendation_note', ''),
            "fraud_warning"      : result.get('fraud_warning', ''),
            "all_doctor_scores"  : result.get('all_doctor_scores', []),
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Recommendation error: {str(e)}")

# ══════════════════════════════════════════════════════════════
# DASHBOARD ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/recent-transactions")
def get_recent_transactions(db: Session = Depends(get_db)):
    """Latest 5 transactions with fraud results for dashboard."""
    from sqlalchemy import desc
    from models_db import Transaction, FraudResult

    rows = (
        db.query(Transaction)
        .order_by(desc(Transaction.id))
        .limit(5)
        .all()
    )

    total = db.query(Transaction).count()

    result = []
    for t in rows:
        fraud = t.fraud_result
        result.append({
            "id"                 : t.id,
            "patient_id"         : t.patient_id,
            "doctor_id"          : t.doctor_id,
            "speciality_name"    : t.speciality_name,
            "diagnosis"          : t.diagnosis,
            "service_description": t.service_description,
            "service_id"         : t.service_id,
            "overall_risk"       : fraud.overall_risk if fraud else "UNKNOWN",
            "created_at"         : str(t.created_at) if t.created_at else None,
        })

    return {"total": total, "transactions": result}

@app.get("/all-transactions")
def get_all_transactions(db: Session = Depends(get_db)):
    """All transactions for AllTransactions page."""
    from sqlalchemy import desc
    from models_db import Transaction

    rows = (
        db.query(Transaction)
        .order_by(desc(Transaction.id))
        .all()
    )

    result = []
    for t in rows:
        fraud = t.fraud_result
        result.append({
            "id"                 : t.id,
            "patient_id"         : t.patient_id,
            "doctor_id"          : t.doctor_id,
            "speciality_name"    : t.speciality_name,
            "diagnosis"          : t.diagnosis,
            "service_description": t.service_description,
            "service_id"         : t.service_id,
            "overall_risk"       : fraud.overall_risk if fraud else "UNKNOWN",
            "created_at"         : str(t.created_at) if t.created_at else None,
        })

    return {"total": len(result), "transactions": result}

@app.get("/transaction/{transaction_id}")
def get_transaction_detail(transaction_id: int, db: Session = Depends(get_db)):
    """Single transaction detail for FraudDetails page."""
    from models_db import Transaction

    t = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not t:
        raise HTTPException(404, "Transaction not found")

    fraud = t.fraud_result

    return {
        "id"                 : t.id,
        "patient_id"         : t.patient_id,
        "doctor_id"          : t.doctor_id,
        "speciality_name"    : t.speciality_name,
        "diagnosis"          : t.diagnosis,
        "service_description": t.service_description,
        "service_id"         : t.service_id,
        "overall_risk"       : fraud.overall_risk if fraud else "UNKNOWN",
        "ttsgan_score"       : fraud.ttsgan_score if fraud else None,
        "dctgan_score"       : fraud.dctgan_score if fraud else None,
        "created_at"         : str(t.created_at) if t.created_at else None,
    }



@app.get("/fraud-distribution")
def get_fraud_distribution(db: Session = Depends(get_db)):
    """Risk distribution for pie chart."""
    from models_db import FraudResult

    rows = db.query(FraudResult).all()

    distribution = {"HIGH RISK": 0, "MEDIUM RISK": 0, "NORMAL": 0}
    for row in rows:
        risk = row.overall_risk or "NORMAL"
        if risk in distribution:
            distribution[risk] += 1
        else:
            distribution["NORMAL"] += 1

    return {"distribution": distribution}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)