# ============================================================
# agent3/agent3_graph.py — UPDATED v2
# AGENT 3 — ORCHESTRATION AGENT
#
# WHAT CHANGED FROM v1:
#
# OLD DESIGN (wrong):
#   One single graph: transaction → recommendation → END
#   Always ran BOTH nodes for every request.
#   run_agent3() took both transaction + patient_input together.
#   This meant a doctor submitting a transaction always triggered
#   the recommendation node too — which made no sense.
#
# NEW DESIGN (correct):
#   TWO separate graphs — one per user workflow:
#
#   GRAPH 1: agent3_transaction_graph
#     Flow  : transaction_node → END
#     Called: run_agent3_transaction(transaction, db)
#     Who   : Doctor / Admin / Staff / Auditor
#     Does  : Calls Agent 1 only (fraud detection)
#     Import: from agent3.agent3_graph import run_agent3_transaction
#
#   GRAPH 2: agent3_recommend_graph
#     Flow  : recommendation_node → END
#     Called: run_agent3_recommend(patient_input, db)
#     Who   : Patient only
#     Does  : Calls Agent 2 only (doctor recommendation)
#     Import: from agent3.agent3_graph import run_agent3_recommend
#
# WHY TWO GRAPHS INSTEAD OF ONE WITH CONDITIONAL:
#   transaction_node needs: transaction dict + db
#   recommendation_node needs: patient_input dict + db
#   They share NO inputs and NO outputs with each other.
#   A single graph with conditional edges would still initialize
#   ALL state fields, causing TypedDict key errors for whichever
#   path was skipped. Two small graphs are cleaner and safer.
#
# UNCHANGED:
#   ✅ transaction_node.py  — no changes needed
#   ✅ recommendation_node.py — no changes needed
#   ✅ Agent3State — used by both graphs, all fields optional
# ============================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END

from agent3.state import Agent3State
from agent3.nodes.transaction_node    import transaction_node
from agent3.nodes.recommendation_node import recommendation_node


# ══════════════════════════════════════════════════════════════
# GRAPH 1 — Transaction only (Agent 1 path)
# ══════════════════════════════════════════════════════════════

def build_agent3_transaction_graph():
    graph = StateGraph(Agent3State)
    graph.add_node("transaction", transaction_node)
    graph.set_entry_point("transaction")
    graph.add_edge("transaction", END)
    compiled = graph.compile()
    print("✅ Agent 3 Transaction graph compiled!")
    print("   Flow: transaction_node → END")
    print("   Calls: Agent 1 only (fraud detection)")
    return compiled


# ══════════════════════════════════════════════════════════════
# GRAPH 2 — Recommendation only (Agent 2 path)
# ══════════════════════════════════════════════════════════════

def build_agent3_recommend_graph():
    graph = StateGraph(Agent3State)
    graph.add_node("recommendation", recommendation_node)
    graph.set_entry_point("recommendation")
    graph.add_edge("recommendation", END)
    compiled = graph.compile()
    print("✅ Agent 3 Recommendation graph compiled!")
    print("   Flow: recommendation_node → END")
    print("   Calls: Agent 2 only (doctor recommendation)")
    return compiled


# ── Build both graphs at import time ─────────────────────────
agent3_transaction_graph = build_agent3_transaction_graph()
agent3_recommend_graph   = build_agent3_recommend_graph()


# ══════════════════════════════════════════════════════════════
# PUBLIC ENTRY FUNCTIONS
# Called by main.py endpoints
# ══════════════════════════════════════════════════════════════

def run_agent3_transaction(transaction: dict, db) -> dict:
    """
    Entry point for POST /transaction endpoint.

    Called by Doctor / Admin / Staff / Auditor.
    Runs Agent 1 only — no recommendation.

    Args:
        transaction : resolved transaction dict (IDs + names)
        db          : SQLAlchemy session

    Returns:
        Agent 1 full state dict — fraud verdict, scores,
        investigation, report, doctor trust score update.
    """
    print("\n" + "=" * 55)
    print("AGENT 3 → TRANSACTION PATH (Agent 1 only)")
    print("=" * 55)
    print(f"  Doctor    : {transaction.get('DOCTOR_ID', 'Unknown')}")
    print(f"  Specialty : {transaction.get('SPECIALITY_NAME', 'Unknown')}")
    print(f"  Service   : {transaction.get('SERVICE_DESCRIPTION', 'Unknown')}")
    print(f"  Diagnosis : {transaction.get('DIAGNOSIS', 'Unknown')}")

    initial_state = {
        'transaction' : transaction,
        'patient_input': {},   # not used in this path
        'db'          : db,
    }

    final_state = agent3_transaction_graph.invoke(initial_state)

    # transaction_node writes agent1_result into state
    # main.py reads directly from the agent1_result dict
    agent1_result = final_state.get('agent1_result', {})

    print("\n" + "=" * 55)
    print("AGENT 3 → TRANSACTION PATH COMPLETE")
    print("=" * 55)
    print(f"  Overall Risk : {agent1_result.get('overall_risk', 'N/A')}")
    print(f"  Is Fraud     : {agent1_result.get('is_fraud', False)}")
    print(f"  Doctor Score : {final_state.get('updated_doctor_score', 'N/A')}")
    print(f"  Report       : {len(agent1_result.get('report', ''))} chars")

    # Return agent1_result directly so main.py can read all fields
    # Also merge top-level Agent 3 fields (doctor_score, status)
    return {
        **agent1_result,
        'updated_doctor_score': final_state.get('updated_doctor_score'),
        'doctor_status'       : final_state.get('doctor_status'),
    }


def run_agent3_recommend(patient_input: dict, db) -> dict:
    """
    Entry point for POST /recommend endpoint.

    Called by Patient only.
    Runs Agent 2 only — no fraud detection.

    Args:
        patient_input : patient dict (patient_id, specialty, max_fee, top_n)
        db            : SQLAlchemy session

    Returns:
        dict with: recommended_doctors, rl_top_pick,
                   recommendation_note, fraud_warning,
                   all_doctor_scores
    """
    print("\n" + "=" * 55)
    print("AGENT 3 → RECOMMENDATION PATH (Agent 2 only)")
    print("=" * 55)
    print(f"  Patient   : {patient_input.get('patient_id', 'Unknown')}")
    print(f"  Specialty : {patient_input.get('required_specialty', 'Any')}")
    print(f"  Max fee   : {patient_input.get('max_fee', 'No limit')}")
    print(f"  Top N     : {patient_input.get('top_n', 5)}")

    initial_state = {
        'transaction'  : {},          # not used in this path
        'patient_input': patient_input,
        'db'           : db,
        # recommendation_node needs these fields from state
        # but they come from transaction_node normally.
        # Since we skip transaction_node, we set safe defaults.
        'doctor_id'            : '',
        'doctor_status'        : 'UNKNOWN',
        'updated_doctor_score' : 100.0,
        'is_fraud'             : False,
        'overall_risk'         : 'NORMAL',
    }

    final_state = agent3_recommend_graph.invoke(initial_state)

    print("\n" + "=" * 55)
    print("AGENT 3 → RECOMMENDATION PATH COMPLETE")
    print("=" * 55)
    print(f"  Recommended : {len(final_state.get('recommended_doctors', []))} doctors")
    print(f"  RL top pick : {final_state.get('rl_top_pick', 'N/A')}")

    return {
        'recommended_doctors': final_state.get('recommended_doctors', []),
        'rl_top_pick'        : final_state.get('rl_top_pick', ''),
        'recommendation_note': final_state.get('recommendation_note', ''),
        'fraud_warning'      : final_state.get('fraud_warning', ''),
        'all_doctor_scores'  : final_state.get('all_doctor_scores', []),
    }


# ── Backward compatibility — kept in case anything still calls run_agent3() ──
def run_agent3(transaction: dict, patient_input: dict, db) -> dict:
    """
    Deprecated: old single-entry function.
    Kept for backward compatibility only.
    New code should call run_agent3_transaction() or run_agent3_recommend().
    """
    print("⚠️  run_agent3() is deprecated. Use run_agent3_transaction() or run_agent3_recommend().")
    return run_agent3_transaction(transaction=transaction, db=db)