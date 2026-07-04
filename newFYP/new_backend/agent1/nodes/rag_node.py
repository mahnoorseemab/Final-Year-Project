# ============================================================
# agent1/nodes/rag_node.py
# NODE 4 OF AGENT 1 — HYBRID RAG REPORT GENERATION
#
# ARCHITECTURE: FAISS + Neo4j Graph RAG (Hybrid)
#
# THIS NODE IS THE ONLY FILE THAT CHANGES when replacing
# simple RAG with Graph RAG. Everything else in Agent 1
# (agent1_graph.py, state.py, model_node, investigation_node,
#  scoring_node) stays exactly the same.
#
# HOW THE HYBRID WORKS (two retrievers, one LLM call):
#
#   FAISS (text similarity):
#     - Loads faiss_index/ built by kb_loader.py
#     - Takes a query string → returns 5 most similar policy chunks
#     - Good at broad policy coverage
#
#   Neo4j Graph RAG (entity relationships):
#     - Connects to local Neo4j at bolt://127.0.0.1:7687
#     - Matches specialty name → finds Specialty node in graph
#     - Checks service against CAN_ORDER edges (direct permission)
#     - Checks service against INHERITS → CAN_ORDER (inherited from
#       General Physician — meaning ANY doctor can order it)
#     - Checks diagnosis against TREATS edges
#     - Returns fraud rules + permission verdict
#
#   retriever.py combines both into one context string.
#   llm_analyzer.py uses the combined context in the Groq prompt.
#
# KEY UPGRADE vs Simple RAG:
#   Simple RAG: "find chunks similar to Emergency+FEVER+HBA1C"
#               → may miss rules, no structure, no inheritance
#
#   Graph RAG adds: "Is HBA1C directly allowed for Emergency?
#                    OR is it inherited from General Physician?
#                    If inherited → NOT a specialty mismatch fraud"
#   This prevents false positives where general lab tests get
#   flagged as fraud just because a specialist ordered them.
#
# FLOW IN THIS NODE:
#   Step 1 → load_retriever()
#     Loads FAISS index from rag/faiss_index/ (cached after first call)
#
#   Step 2 → retrieve_policies(retriever, state)
#     Calls retriever.py which runs:
#       A) FAISS search → 5 policy chunks
#       B) graph_retriever.retrieve_graph_context(state)
#          → Neo4j traversal → service/diagnosis permission check
#     Returns combined context string (FAISS + Graph)
#     NOTE: If Neo4j is down, graph_retriever.py handles the error
#     gracefully and returns "[Graph RAG] Graph retrieval failed".
#     retrieve_policies() still returns FAISS-only context — no crash.
#
#   Step 3 → analyze_claim(state, retrieved_policies)
#     Calls llm_analyzer.py with the combined context.
#     Groq LLM generates the report using structured prompts
#     that tell it exactly how to use Graph RAG context:
#       - INHERITS from General Physician → NOT specialty fraud
#       - NOT FOUND in graph → specialty mismatch IS fraud evidence
#
# PREREQUISITES (run these once before starting the server):
#   1. python rag/kb_loader.py       → builds rag/faiss_index/
#   2. python rag/neo4j_builder.py   → builds Neo4j knowledge graph
#   3. Neo4j Desktop must be running at bolt://127.0.0.1:7687
#
# FALLBACK:
#   If FAISS index missing OR Groq API fails → template report.
#   Neo4j being down alone does NOT trigger fallback (FAISS still runs).
# ============================================================

import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', '..')
)

# ── FAISS retriever cache ─────────────────────────────────────
# Loaded ONCE on first call, reused for every request after that.
# Avoids re-loading the embedding model on every transaction.
_retriever = None


def _get_retriever():
    """
    Returns the FAISS retriever.
    Loads it from disk on first call, returns cached on all others.
    """
    global _retriever
    if _retriever is None:
        from rag.retriever import load_retriever
        _retriever = load_retriever()
    return _retriever


def rag_node(state: dict) -> dict:
    """
    LangGraph Node 4 — Hybrid RAG Report Generation.

    Runs for EVERY transaction (fraud and normal alike).
    Called by agent1_graph.py as the final node in both paths:
      Fraud  → model → investigation → scoring → rag → END
      Normal → model → rag → END

    For normal transactions:
      Generates a clearance report confirming legitimate billing.
      FAISS retrieves general compliance policies.
      Graph RAG confirms service is allowed for the specialty.

    For fraud transactions:
      Generates a full investigation report with specific violations.
      FAISS retrieves fraud-specific policy text.
      Graph RAG tells the LLM whether specialty mismatch is genuine
      fraud OR just an inherited general service (not fraud).
    """

    print("\n" + "=" * 55)
    print("NODE 4 — HYBRID RAG REPORT GENERATION")
    print("         (FAISS + Neo4j Graph RAG)")
    print("=" * 55)

    transaction  = state.get('transaction', {})
    is_fraud     = state.get('is_fraud', False)
    overall_risk = state.get('overall_risk', 'NORMAL')
    doctor_id    = transaction.get('DOCTOR_ID', 'Unknown')
    speciality   = transaction.get('SPECIALITY_NAME', 'Unknown')
    diagnosis    = transaction.get('DIAGNOSIS', 'Unknown')
    service      = transaction.get('SERVICE_DESCRIPTION', 'Unknown')

    print(f"Doctor     : {doctor_id}")
    print(f"Specialty  : {speciality}")
    print(f"Diagnosis  : {diagnosis}")
    print(f"Service    : {service}")
    print(f"Is Fraud   : {is_fraud}")
    print(f"Risk Level : {overall_risk}")
    print(f"Report type: {'FRAUD INVESTIGATION' if is_fraud else 'NORMAL CLEARANCE'}")

    try:
        # ── Step 1: Load FAISS retriever ──────────────────────
        # First call loads from disk (slow ~2s).
        # All subsequent calls return the cached retriever instantly.
        print("\nStep 1: Loading FAISS retriever...")
        retriever = _get_retriever()
        print("  ✅ FAISS retriever ready")

        # ── Step 2: Hybrid retrieval ──────────────────────────
        # retrieve_policies() from retriever.py does TWO things:
        #
        # A) FAISS: builds a query string from the transaction
        #    fields (specialty, service, diagnosis, fraud_type)
        #    and retrieves top 5 similar policy text chunks.
        #
        # B) Graph RAG: calls graph_retriever.retrieve_graph_context()
        #    which connects to Neo4j and traverses:
        #      specialty → CAN_ORDER → service     (direct permission)
        #      specialty → INHERITS  → CAN_ORDER   (inherited permission)
        #      specialty → TREATS    → diagnosis   (valid diagnosis check)
        #      specialty → FRAUD_IF_NO → FraudRule (relevant fraud rules)
        #
        # Both results are concatenated into one context string and
        # returned. If Neo4j is unavailable, graph_retriever catches
        # the error and returns a "[Graph RAG] failed" placeholder —
        # retrieve_policies() still returns the FAISS context.
        print("Step 2: Retrieving policies (FAISS + Neo4j Graph RAG)...")
        from rag.retriever import retrieve_policies
        retrieved_policies = retrieve_policies(retriever, state)
        print("  ✅ Policy context retrieved")

        # ── Step 3: Generate report via Groq LLM ─────────────
        # analyze_claim() in llm_analyzer.py receives the full
        # combined context. The prompt tells Groq exactly how to
        # interpret the Graph RAG section:
        #
        # Normal report prompt:
        #   Check Graph RAG — if service is ALLOWED (direct or
        #   inherited) → confirm billing is legitimate.
        #
        # Fraud report prompt:
        #   Check Graph RAG — if service INHERITS from General
        #   Physician → do NOT use specialty mismatch as fraud
        #   evidence. Evaluate all other signals normally.
        #   If service NOT FOUND → specialty mismatch IS fraud.
        print("Step 3: Generating report via Groq LLM...")
        from rag.llm_analyzer import analyze_claim
        report = analyze_claim(state, retrieved_policies)

        print(f"\n✅ Hybrid RAG Node complete!")
        print(f"   Report length : {len(report)} chars")

    except Exception as e:
        print(f"\n⚠️  Hybrid RAG failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Falling back to template report...")
        report = _fallback_report(state)

    return {'report': report}


# ─────────────────────────────────────────────────────────────
# FALLBACK TEMPLATE REPORT
#
# Used when FAISS index is missing or Groq API call fails.
# Neo4j being down alone does NOT reach here — FAISS still
# runs and Groq still generates a report (FAISS-only context).
# ─────────────────────────────────────────────────────────────

def _fallback_report(state: dict) -> str:
    """
    Generates a template report from state data alone.
    No FAISS, no Graph RAG, no LLM — pure state fields.

    Two templates:
      Normal transaction → clearance template
      Fraud transaction  → investigation template
    """
    t            = state.get('transaction', {})
    is_fraud     = state.get('is_fraud', False)
    overall_risk = state.get('overall_risk', 'NORMAL')

    # ── Normal clearance fallback ──────────────────────────────
    if not is_fraud:
        return f"""
HEALTHCARE TRANSACTION CLEARANCE REPORT
========================================
VERDICT: LEGITIMATE

EXECUTIVE SUMMARY:
  Transaction for Doctor {t.get('DOCTOR_ID')} has been reviewed
  and cleared. No billing irregularities detected.
  Transaction is approved as NORMAL.

TRANSACTION DETAILS:
  Patient ID  : {t.get('PATIENT_ID')}
  Doctor ID   : {t.get('DOCTOR_ID')}
  Speciality  : {t.get('SPECIALITY_NAME')}
  Diagnosis   : {t.get('DIAGNOSIS')}
  Service     : {t.get('SERVICE_DESCRIPTION')}
  Service ID  : {t.get('SERVICE_ID')}
  Verdict     : {overall_risk}

POLICY COMPLIANCE:
  The billing pattern for this transaction is consistent
  with normal doctor behavior. The service, specialty, and
  diagnosis combination is clinically appropriate.

RECOMMENDATION: APPROVE

⚠️  DISCLAIMER:
  This is an automated fallback report. The full hybrid RAG
  report (FAISS + Neo4j) was unavailable. Likely cause:
  FAISS index missing (run kb_loader.py) or Groq API error.
  Neo4j being down does NOT cause this fallback.
""".strip()

    # ── Fraud investigation fallback ───────────────────────────
    responsible     = state.get('responsible_party', 'Unknown')
    fraud_type      = state.get('primary_fraud_type', 'Unknown')
    fraud_types     = state.get('fraud_types', [])
    confidence      = state.get('confidence', 0.0)
    score           = state.get('updated_doctor_score', 'N/A')
    penalty         = state.get('doctor_score_penalty', 0)
    overbilling     = state.get('overbilling_flag', False)
    reasons         = state.get('reasons', [])
    score_breakdown = state.get('score_breakdown', {})
    score_status    = score_breakdown.get('score_status', 'Unknown')
    previous_frauds = score_breakdown.get('previous_frauds', 0)

    fraud_types_text = "\n".join(
        f"  • {ft}" for ft in fraud_types
    ) or "  • Unknown"

    reasons_text = "\n".join(
        f"  • {r[:200]}" for r in reasons[:4]
    ) or "  • No reasons captured"

    return f"""
HEALTHCARE FRAUD INVESTIGATION REPORT
======================================
VERDICT: FRAUDULENT

EXECUTIVE SUMMARY:
  Doctor {t.get('DOCTOR_ID')} has been flagged for {fraud_type}.
  {responsible} identified as responsible party.
  Confidence Level: {confidence * 100:.1f}%

TRANSACTION DETAILS:
  Patient ID  : {t.get('PATIENT_ID')}
  Doctor ID   : {t.get('DOCTOR_ID')}
  Speciality  : {t.get('SPECIALITY_NAME')}
  Diagnosis   : {t.get('DIAGNOSIS')}
  Service     : {t.get('SERVICE_DESCRIPTION')}
  Service ID  : {t.get('SERVICE_ID')}
  Risk Level  : {overall_risk}

INVESTIGATION FINDINGS:
  Responsible Party  : {responsible}
  Primary Fraud Type : {fraud_type}

  All Fraud Types Detected:
{fraud_types_text}

  Detailed Reasons:
{reasons_text}

OVERBILLING ASSESSMENT:
  {"⚠️  OVERBILLING SUSPECTED — Consistent anomalous billing detected." if overbilling else "No overbilling pattern detected."}

DOCTOR ACCOUNTABILITY:
  Previous Fraud Count : {previous_frauds}
  Penalty Applied      : -{penalty} points
  Updated Score        : {score} / 100
  Doctor Status        : {score_status}

RECOMMENDATION: REJECT & INVESTIGATE FURTHER

ACTION ITEMS:
  1. Suspend doctor billing privileges pending investigation
  2. Audit all transactions for this doctor for past 6 months
  3. Refer case to fraud investigation team immediately

⚠️  DISCLAIMER:
  This is an automated fallback report. The full hybrid RAG
  report (FAISS + Neo4j) was unavailable. Likely cause:
  FAISS index missing (run kb_loader.py) or Groq API error.
  Neo4j being down does NOT cause this fallback.
""".strip()