# ============================================================
# agent3/nodes/recommendation_node.py — NODE 2 OF AGENT 3
#
# WHAT CHANGED FROM v1:
#
# OLD: Always assumed transaction_node ran before it, so it
#      expected doctor_id, doctor_status etc. to be in state.
#
# NEW: Works in TWO modes:
#
#   MODE 1 — Standalone (POST /recommend path):
#     transaction_node did NOT run before this.
#     doctor_id = '' (empty default from run_agent3_recommend)
#     fraud_context is empty / neutral.
#     Agent 2 runs purely on patient_input — no fraud context.
#
#   MODE 2 — After transaction_node (old /analyze path, deprecated):
#     doctor_id, doctor_status, is_fraud etc. are set by Node 1.
#     fraud_context is passed to Agent 2 for awareness.
#
# Both modes call run_agent2() the same way — Agent 2 handles
# the fraud_context internally (uses it or ignores it).
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def recommendation_node(state: dict) -> dict:

    print("\n" + "=" * 55)
    print("AGENT 3 — NODE 2: RECOMMENDATION NODE (Agent 2)")
    print("=" * 55)

    patient_input        = state.get('patient_input', {})
    # These may be empty strings/defaults in standalone mode
    doctor_id            = state.get('doctor_id', '')
    doctor_status        = state.get('doctor_status', 'UNKNOWN')
    updated_doctor_score = state.get('updated_doctor_score', 100.0)
    is_fraud             = state.get('is_fraud', False)
    overall_risk         = state.get('overall_risk', 'NORMAL')
    db                   = state.get('db')

    print(f"  Patient    : {patient_input.get('patient_id', 'Unknown')}")
    print(f"  Specialty  : {patient_input.get('required_specialty', 'Any')}")
    print(f"  Max fee    : {patient_input.get('max_fee', 'No limit')}")
    print(f"  Mode       : {'Standalone' if not doctor_id else 'Post-transaction'}")

    # fraud_context — empty/neutral in standalone mode
    fraud_context = {
        'doctor_id'    : doctor_id,
        'doctor_score' : updated_doctor_score,
        'doctor_status': doctor_status,
        'is_fraud'     : is_fraud,
        'overall_risk' : overall_risk,
    }

    try:
        from agent2.agent2_graph import run_agent2
        agent2_result = run_agent2(
            patient_input = patient_input,
            fraud_context = fraud_context,
            db            = db,
        )

        recommended_doctors = agent2_result.get('recommended_doctors', [])
        recommendation_note = agent2_result.get('recommendation_note', '')
        fraud_warning       = agent2_result.get('fraud_warning', '')
        rl_top_pick         = agent2_result.get('rl_top_pick', '')

        print(f"  ✅ Agent 2 done — {len(recommended_doctors)} recommendations")
        if rl_top_pick:
            print(f"  RL top pick : {rl_top_pick}")

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ⚠️  Agent 2 failed: {e}")
        recommended_doctors = []
        recommendation_note = f"Recommendation unavailable: {str(e)}"
        fraud_warning       = _fallback_warning(doctor_id, doctor_status, updated_doctor_score)
        rl_top_pick         = ''

    # Fetch all trust scores for dashboard/response
    all_scores = _fetch_all_scores(db)

    return {
        'all_doctor_scores'   : all_scores,
        'recommended_doctors' : recommended_doctors,
        'recommendation_note' : recommendation_note,
        'recommendation_ready': True,
        'fraud_warning'       : fraud_warning,
        'rl_top_pick'         : rl_top_pick,
    }


def _fetch_all_scores(db) -> list:
    """Fetch all doctor trust scores from DB for dashboard."""
    if not db:
        return []
    try:
        from models_db import DoctorScore
        from sqlalchemy import desc
        rows = db.query(DoctorScore).order_by(desc(DoctorScore.current_score)).all()
        return [{
            'doctor_id'    : r.doctor_id,
            'current_score': r.current_score,
            'fraud_count'  : r.fraud_count,
            'last_updated' : str(r.last_updated) if r.last_updated else None,
        } for r in rows]
    except Exception as e:
        print(f"  ⚠️  Score fetch failed: {e}")
        return []


def _fallback_warning(doctor_id: str, status: str, score: float) -> str:
    """Warning string when Agent 2 fails and doctor has low trust score."""
    if not doctor_id:
        return ''
    if status == 'SUSPENDED':
        return f"🚨 Doctor {doctor_id} is SUSPENDED (Score: {score}/100). Do not recommend."
    if status == 'FLAGGED':
        return f"⚠️  Doctor {doctor_id} is FLAGGED (Score: {score}/100). Recommend with caution."
    return ''