# ============================================================
# agent1/agent1_graph.py
#
# CHANGED:
#   Normal transactions no longer stop at END after model_node.
#   Both fraud AND normal transactions always reach rag_node.
#   Only fraud transactions go through investigation + scoring first.
#
# FLOW:
#   Fraud  → model → investigation → scoring → rag → END
#   Normal → model → rag → END
# ============================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END

from agent1.state import Agent1State

from agent1.nodes.model_node         import model_node, should_investigate
from agent1.nodes.investigation_node import investigation_node
from agent1.nodes.scoring_node       import scoring_node
from agent1.nodes.rag_node           import rag_node


def build_agent1_graph():

    graph = StateGraph(Agent1State)

    graph.add_node("model",         model_node)
    graph.add_node("investigation", investigation_node)
    graph.add_node("scoring",       scoring_node)
    graph.add_node("rag",           rag_node)

    graph.set_entry_point("model")

    # ── KEY CHANGE ────────────────────────────────────────────
    # Old: normal → "end" (END)   ← report was never generated
    # New: normal → "rag"         ← report always generated
    graph.add_conditional_edges(
        "model",
        should_investigate,
        {
            "investigation": "investigation",  # fraud path
            "rag"          : "rag"             # normal path → skip to RAG
        }
    )

    graph.add_edge("investigation", "scoring")
    graph.add_edge("scoring",       "rag")
    graph.add_edge("rag",           END)

    compiled = graph.compile()

    print("✅ Agent 1 graph compiled!")
    print("   Fraud  flow: model → investigation → scoring → rag → END")
    print("   Normal flow: model → rag → END")
    return compiled


agent1_graph = build_agent1_graph()


def run_agent1(transaction: dict, db) -> dict:

    print("\n" + "="*55)
    print("AGENT 1 STARTING")
    print("="*55)

    initial_state = {
        'transaction': transaction,
        'db'         : db,
    }

    final_state = agent1_graph.invoke(initial_state)

    print("\n" + "="*55)
    print("AGENT 1 COMPLETE")
    print("="*55)
    print(f"  Overall Risk : {final_state.get('overall_risk', 'N/A')}")
    print(f"  TTS Score    : {final_state.get('ttsgan_result', {}).get('score', 'N/A')}")
    print(f"  DCT Score    : {final_state.get('dctgan_result', {}).get('score', 'N/A')}")
    print(f"  Is Fraud     : {final_state.get('is_fraud', False)}")
    print(f"  Report       : {len(final_state.get('report', ''))} chars")

    if final_state.get('is_fraud'):
        print(f"  Responsible  : {final_state.get('responsible_party', 'N/A')}")
        print(f"  Fraud Type   : {final_state.get('primary_fraud_type', 'N/A')}")
        print(f"  Doctor Score : {final_state.get('updated_doctor_score', 'N/A')}")
    else:
        print("  Normal transaction — investigation + scoring skipped")

    return final_state