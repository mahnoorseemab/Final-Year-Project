# ============================================================
# agent3/nodes/transaction_node.py
# NODE 1 OF AGENT 3 — TRANSACTION HANDLER
#
# WHAT THIS NODE DOES:
#   This is the bridge between Agent 3 and Agent 1.
#   It takes the transaction from Agent 3's state,
#   calls run_agent1() fully (fraud detection + investigation
#   + scoring + RAG report — everything Agent 1 does),
#   then extracts the key results and writes them to
#   Agent 3's state so the next node (recommendation_node)
#   and the final response can use them.
#
# INPUT  (reads from Agent 3 state):
#   state['transaction'] → the billing transaction dict
#   state['db']          → SQLAlchemy session
#
# OUTPUT (writes to Agent 3 state):
#   agent1_result        → full Agent 1 output dict
#   is_fraud             → bool
#   overall_risk         → str
#   doctor_id            → str
#   updated_doctor_score → float
#   doctor_status        → str
# ============================================================

import sys
import os

# Make sure root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def transaction_node(state: dict) -> dict:
    """
    LangGraph Node 1 of Agent 3.

    Calls Agent 1 fully and extracts results into Agent 3 state.

    Args:
        state: current Agent3State

    Returns:
        dict: fields to update in Agent 3 state
    """

    print("\n" + "=" * 55)
    print("AGENT 3 — NODE 1: TRANSACTION NODE")
    print("=" * 55)

    transaction = state['transaction']
    db          = state['db']

    doctor_id = transaction.get('DOCTOR_ID', 'Unknown')
    print(f"Doctor     : {doctor_id}")
    print(f"Specialty  : {transaction.get('SPECIALITY_NAME', 'Unknown')}")
    print(f"Service    : {transaction.get('SERVICE_DESCRIPTION', 'Unknown')}")
    print(f"Diagnosis  : {transaction.get('DIAGNOSIS', 'Unknown')}")

    # ── Call Agent 1 fully ────────────────────────────────────
    # This triggers the full Agent 1 pipeline:
    #   model_node → investigation_node → scoring_node → rag_node
    # Everything Agent 1 does is done here.
    print("\nCalling Agent 1 (full pipeline)...")
    from agent1.agent1_graph import run_agent1
    agent1_result = run_agent1(transaction=transaction, db=db)

    print("Agent 1 complete!")
    print(f"  Overall Risk : {agent1_result.get('overall_risk', 'N/A')}")
    print(f"  Is Fraud     : {agent1_result.get('is_fraud', False)}")

    # ── Extract key values for Agent 3 state ─────────────────
    is_fraud     = agent1_result.get('is_fraud', False)
    overall_risk = agent1_result.get('overall_risk', 'NORMAL')

    # Get doctor score from Agent 1 result
    # If fraud → Agent 1 scoring_node updated it
    # If normal → read existing score from DB
    updated_doctor_score = agent1_result.get('updated_doctor_score', None)

    if updated_doctor_score is None:
        # Normal transaction — Agent 1 skipped scoring node
        # Read existing score from DB directly
        updated_doctor_score = _get_existing_doctor_score(db, doctor_id)

    # Derive doctor status from score
    doctor_status = _score_to_status(updated_doctor_score)

    print(f"\nDoctor Score : {updated_doctor_score}")
    print(f"Doctor Status: {doctor_status}")

    return {
        'agent1_result'       : agent1_result,
        'is_fraud'            : is_fraud,
        'overall_risk'        : overall_risk,
        'doctor_id'           : doctor_id,
        'updated_doctor_score': updated_doctor_score,
        'doctor_status'       : doctor_status,
    }


def _get_existing_doctor_score(db, doctor_id: str) -> float:
    """
    Reads existing doctor score from DB for normal transactions.
    If doctor not found → returns default 100.0 (no fraud history).
    """
    try:
        from models_db import DoctorScore
        row = db.query(DoctorScore).filter(
            DoctorScore.doctor_id == doctor_id
        ).first()
        if row:
            print(f"   Existing score from DB: {row.current_score}")
            return row.current_score
        else:
            print(f"   No score in DB yet — defaulting to 100.0")
            return 100.0
    except Exception as e:
        print(f"   ⚠️  Could not fetch doctor score: {e}")
        return 100.0


def _score_to_status(score: float) -> str:
    """
    Converts a numeric score to a human-readable status label.
    Matches the same thresholds used in scoring_node.py.
    """
    if score is None:
        return "UNKNOWN"
    if score >= 80:
        return "TRUSTED"
    elif score >= 60:
        return "WATCH LIST"
    elif score >= 40:
        return "FLAGGED"
    else:
        return "SUSPENDED"