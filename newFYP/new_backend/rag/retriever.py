# ============================================================
# rag/retriever.py
# HYBRID POLICY RETRIEVER — FAISS + Graph RAG
#
# UPDATED:
#   - retrieve_policies now combines:
#     1. FAISS vector search (existing)
#     2. Graph RAG Neo4j traversal (new)
#   - Both contexts passed to LLM together
# ============================================================

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


def load_retriever():
    """
    Loads the FAISS index from disk.
    Called ONCE — reused for every transaction.
    """
    print("Loading FAISS retriever...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "faiss_index"
    )

    vectorstore = FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    print("✅ Retriever loaded!")
    return retriever


def retrieve_policies(retriever, agent1_state: dict) -> str:
    """
    Hybrid retrieval — FAISS + Graph RAG combined.

    Step 1: FAISS retrieves top 5 policy chunks
    Step 2: Graph RAG traverses Neo4j for relational context
    Step 3: Both combined and returned to LLM

    Args:
        retriever:    loaded FAISS retriever
        agent1_state: full Agent 1 state dict

    Returns:
        str: combined FAISS + Graph context for LLM
    """

    transaction        = agent1_state.get('transaction', {})
    speciality         = transaction.get('SPECIALITY_NAME', 'Unknown')
    service            = transaction.get('SERVICE_DESCRIPTION', 'Unknown')
    diagnosis          = transaction.get('DIAGNOSIS', 'Unknown')

    rel_feats          = agent1_state.get('relationship_features', {})
    doc_spec           = rel_feats.get('doc_spec_match', 0.5)
    svc_spec           = rel_feats.get('svc_spec_match', 0.5)
    diag_svc           = rel_feats.get('diag_svc_match', 0.5)

    overall_risk       = agent1_state.get('overall_risk', 'Unknown')
    responsible_party  = agent1_state.get('responsible_party', 'Unknown')
    primary_fraud_type = agent1_state.get('primary_fraud_type', 'Unknown')

    # ── STEP 1: FAISS Retrieval (existing) ──────────────────
    query = f"""
    Speciality: {speciality}
    Service: {service}
    Diagnosis: {diagnosis}
    Risk Level: {overall_risk}
    Responsible Party: {responsible_party}
    Fraud Type: {primary_fraud_type}
    Doctor billing outside specialty: {doc_spec == 0.0}
    Service specialty mismatch: {svc_spec == 0.0}
    Diagnosis service mismatch: {diag_svc == 0.0}
    """

    print(f"[FAISS] Searching policies for: {speciality} | {primary_fraud_type} | {diagnosis}")

    relevant_chunks = retriever.invoke(query)
    faiss_context   = "\n\n".join(
        chunk.page_content for chunk in relevant_chunks
    )
    print(f"✅ FAISS retrieved {len(relevant_chunks)} policy chunks")

    # ── STEP 2: Graph RAG Retrieval (new) ───────────────────
    graph_context = ""
    try:
        from rag.graph_retriever import retrieve_graph_context
        graph_context = retrieve_graph_context(agent1_state)
        print("✅ Graph RAG context retrieved!")
    except Exception as e:
        print(f"⚠️  Graph RAG skipped: {e}")
        graph_context = "[Graph RAG] Not available\n"

    # ── STEP 3: Combine both contexts ───────────────────────
    combined_context = f"""
{'='*60}
FAISS POLICY CONTEXT
{'='*60}
{faiss_context}

{'='*60}
GRAPH RAG CONTEXT
{'='*60}
{graph_context}
"""

    return combined_context