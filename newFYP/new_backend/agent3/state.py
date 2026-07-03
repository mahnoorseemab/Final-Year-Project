# ============================================================
# agent3/state.py — UPDATED v2
# THE SHARED WHITEBOARD FOR AGENT 3
#
# WHAT CHANGED FROM v1:
#
# Because Agent 3 now has TWO separate graphs (one for transaction,
# one for recommendation), each graph only uses HALF the fields.
# The transaction graph never sets patient_input fields.
# The recommendation graph never sets agent1_result fields.
#
# All fields are now Optional so LangGraph doesn't raise
# TypedDict validation errors for whichever path was skipped.
#
# TRANSACTION PATH uses:
#   transaction, db → agent1_result, is_fraud, overall_risk,
#   doctor_id, updated_doctor_score, doctor_status
#
# RECOMMENDATION PATH uses:
#   patient_input, db → recommended_doctors, rl_top_pick,
#   recommendation_note, fraud_warning, all_doctor_scores
# ============================================================

from typing import TypedDict, Optional, List, Any


class Agent3State(TypedDict, total=False):
    """
    total=False means ALL fields are optional by default.
    Each graph only fills in the fields it uses.
    """

    # ── INPUTS ────────────────────────────────────────────────

    transaction: dict
    # Billing transaction dict — used by transaction path only
    # Contains: DOCTOR_ID, SPECIALITY_ID, SERVICE_ID,
    #           DIAGNOSIS_ID, PATIENT_ID, SPECIALITY_NAME,
    #           SERVICE_DESCRIPTION, DIAGNOSIS

    patient_input: dict
    # Patient recommendation request — used by recommend path only
    # Contains: patient_id, required_specialty, max_fee, top_n

    db: Any
    # SQLAlchemy session — passed through for DB access

    # ── TRANSACTION PATH OUTPUT (from transaction_node) ───────

    agent1_result: dict
    # Full output dict from run_agent1()
    # Contains all Agent 1 fields: fraud verdict, scores,
    # investigation result, RAG report, doctor score breakdown

    is_fraud: bool
    # Pulled from agent1_result for easy access

    overall_risk: str
    # "HIGH RISK" / "MEDIUM RISK" / "NORMAL"

    doctor_id: str
    # The doctor ID from the transaction

    updated_doctor_score: float
    # Doctor's score after Agent 1 penalty (fraud)
    # or existing score from DB (normal transaction)

    doctor_status: str
    # "TRUSTED" / "WATCH LIST" / "FLAGGED" / "SUSPENDED"

    # ── RECOMMENDATION PATH OUTPUT (from recommendation_node) ─

    recommended_doctors: list
    # Ranked list of doctor recommendations from Agent 2
    # Each item: { doctor_id, score, specialty, rating, ... }

    rl_top_pick: str
    # Single best doctor chosen by RL agent

    recommendation_note: str
    # Explanation of the recommendation ranking

    fraud_warning: str
    # Warning if a recommended doctor has low trust score

    recommendation_ready: bool
    # True = recommendation_node ran successfully

    all_doctor_scores: list
    # All doctor trust scores from DB (for admin dashboard)
    # Each item: { doctor_id, current_score, fraud_count, last_updated }