# ============================================================
# rag/graph_retriever.py
# GRAPH RAG — Neo4j Graph Retriever
#
# WHAT THIS DOES:
#   - Connects to Neo4j knowledge graph
#   - Given a claim (specialty, diagnosis, service)
#   - Traverses graph to find relevant nodes + relationships
#   - Checks inheritance from General Physician
#   - Returns graph context string for LLM
#
# Used by: retriever.py (combined with FAISS)
# ============================================================

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()

# ── Neo4j Credentials ────────────────────────────────────────
NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ── Specialty Mapping ────────────────────────────────────────
SPECIALTY_MAPPING = {
    "emergency": "EMERGENCY / CRITICAL CARE",
    "neurosurgery": "NEUROLOGIST / NEUROSURGEON",
    "neurologist": "NEUROLOGIST / NEUROSURGEON",
    "orthopedic": "ORTHOPEDIC / PHYSICAL MED & REHABILITATION",
    "physical med & rehabilitation": "ORTHOPEDIC / PHYSICAL MED & REHABILITATION",
    "medical specialist": "BLOOD BANK STAFF / MEDICAL SPECIALIST",
    "blood bank staff": "BLOOD BANK STAFF / MEDICAL SPECIALIST",
    "cardiologist": "CARDIOLOGIST / SURGERY - CARDIAC / PEDIATRIC CARDIOLOGIST",
    "surgery - cardiac": "CARDIOLOGIST / SURGERY - CARDIAC / PEDIATRIC CARDIOLOGIST",
    "pediatric cardiologist": "CARDIOLOGIST / SURGERY - CARDIAC / PEDIATRIC CARDIOLOGIST",
    "infectious diseases": "INFECTIOUS DISEASES SPECIALIST",
    "surgery - plastic": "GENERAL SURGEON / PEDIATRIC SURGEON / SURGERY - PLASTIC",
    "pediatric surgeon": "GENERAL SURGEON / PEDIATRIC SURGEON / SURGERY - PLASTIC",
    "general surgeon": "GENERAL SURGEON / PEDIATRIC SURGEON / SURGERY - PLASTIC",
    "gastroenterogist": "GASTROENTEROLOGIST",
    "critical care": "EMERGENCY / CRITICAL CARE",
    "cardiology": "CARDIOLOGIST / SURGERY - CARDIAC / PEDIATRIC CARDIOLOGIST",
    "pediatrics": "PEDIATRICIAN",
    "pediatrician": "PEDIATRICIAN",
}

# ============================================================
# NEO4J CONNECTION
# ============================================================

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run(self, query, parameters=None):
        with self.driver.session() as session:
            return list(session.run(query, parameters or {}))


# Singleton connection — load once, reuse
_graph_conn = None

def get_graph_connection():
    global _graph_conn
    if _graph_conn is None:
        _graph_conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        print("✅ Graph retriever connected to Neo4j!")
    return _graph_conn


# ============================================================
# GRAPH TRAVERSAL FUNCTIONS
# ============================================================

def get_specialty_node(conn, specialty_name):
    # Check mapping first
    mapped = SPECIALTY_MAPPING.get(specialty_name.lower())
    if mapped:
        return mapped
    
    # Fallback: fuzzy match
    result = conn.run("""
        MATCH (sp:Specialty)
        WHERE toLower(sp.name) CONTAINS toLower($specialty)
        OR toLower($specialty) CONTAINS toLower(sp.name)
        RETURN sp.name as name
        LIMIT 1
    """, {"specialty": specialty_name})

    if result:
        return result[0]["name"]
    return None


def get_direct_services(conn, specialty_node):
    """
    Get all services directly allowed for this specialty (CAN_ORDER).
    """
    result = conn.run("""
        MATCH (sp:Specialty {name: $specialty})-[:CAN_ORDER]->(s:Service)
        RETURN s.name as service
    """, {"specialty": specialty_node})

    return [r["service"] for r in result]


def get_inherited_services(conn, specialty_node):
    """
    Get all services inherited from General Physician via INHERITS.
    """
    result = conn.run("""
        MATCH (sp:Specialty {name: $specialty})-[:INHERITS]->(gp:Specialty)
        -[:CAN_ORDER]->(s:Service)
        RETURN s.name as service, gp.name as inherited_from
    """, {"specialty": specialty_node})

    return [(r["service"], r["inherited_from"]) for r in result]


def get_specialty_diagnoses(conn, specialty_node):
    """
    Get all diagnoses this specialty treats.
    """
    result = conn.run("""
        MATCH (sp:Specialty {name: $specialty})-[:TREATS]->(d:Diagnosis)
        RETURN d.name as diagnosis
    """, {"specialty": specialty_node})

    return [r["diagnosis"] for r in result]


def get_fraud_rules(conn, specialty_node):
    """
    Get fraud rules for this specialty.
    """
    result = conn.run("""
        MATCH (sp:Specialty {name: $specialty})-[:FRAUD_IF_NO]->(fr:FraudRule)
        RETURN fr.description as rule
    """, {"specialty": specialty_node})

    return [r["rule"] for r in result]


def check_service_allowed(conn, specialty_node, service_name):
    """
    Check if a specific service is allowed for this specialty.
    Checks both direct CAN_ORDER and inherited from General Physician.
    Returns: (allowed: bool, source: str)
    """
    # Check direct
    result = conn.run("""
        MATCH (sp:Specialty {name: $specialty})-[:CAN_ORDER]->(s:Service)
        WHERE toLower(s.name) CONTAINS toLower($service)
        RETURN s.name as service
        LIMIT 1
    """, {"specialty": specialty_node, "service": service_name})

    if result:
        return True, "direct"

    # Check inherited
    result = conn.run("""
        MATCH (sp:Specialty {name: $specialty})-[:INHERITS]->(gp:Specialty)
        -[:CAN_ORDER]->(s:Service)
        WHERE toLower(s.name) CONTAINS toLower($service)
        RETURN s.name as service, gp.name as gp_name
        LIMIT 1
    """, {"specialty": specialty_node, "service": service_name})

    if result:
        return True, f"inherited from {result[0]['gp_name']}"

    return False, "not found"


def check_diagnosis_valid(conn, specialty_node, diagnosis_name):
    """
    Check if a diagnosis is valid for this specialty.
    Checks direct + inherited.
    """
    # Direct
    result = conn.run("""
        MATCH (sp:Specialty {name: $specialty})-[:TREATS]->(d:Diagnosis)
        WHERE toLower(d.name) CONTAINS toLower($diagnosis)
        RETURN d.name as diagnosis
        LIMIT 1
    """, {"specialty": specialty_node, "diagnosis": diagnosis_name})

    if result:
        return True, "direct"

    # Inherited
    result = conn.run("""
        MATCH (sp:Specialty {name: $specialty})-[:INHERITS]->(gp:Specialty)
        -[:TREATS]->(d:Diagnosis)
        WHERE toLower(d.name) CONTAINS toLower($diagnosis)
        RETURN d.name as diagnosis, gp.name as gp_name
        LIMIT 1
    """, {"specialty": specialty_node, "diagnosis": diagnosis_name})

    if result:
        return True, f"inherited from {result[0]['gp_name']}"

    return False, "not found"


# ============================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================

def retrieve_graph_context(agent1_state: dict) -> str:
    """
    Main function called by retriever.py.

    Given agent1_state with transaction details,
    traverses Neo4j graph and returns a context string
    that gets combined with FAISS policy chunks for LLM.

    Args:
        agent1_state: full Agent 1 state dict

    Returns:
        str: graph context for LLM
    """
    transaction = agent1_state.get('transaction', {})
    specialty   = transaction.get('SPECIALITY_NAME', '')
    service     = transaction.get('SERVICE_DESCRIPTION', '')
    diagnosis   = transaction.get('DIAGNOSIS', '')

    try:
        conn = get_graph_connection()

        # ── Find specialty node ──────────────────────────────
        specialty_node = get_specialty_node(conn, specialty)

        if not specialty_node:
            return (
                f"[Graph RAG] No matching specialty node found "
                f"for '{specialty}' in knowledge graph.\n"
            )

        # ── Check service allowed ────────────────────────────
        service_allowed, service_source = check_service_allowed(
            conn, specialty_node, service
        )

        # ── Check diagnosis valid ────────────────────────────
        diagnosis_valid, diagnosis_source = check_diagnosis_valid(
            conn, specialty_node, diagnosis
        )

        # ── Get fraud rules ──────────────────────────────────
        fraud_rules = get_fraud_rules(conn, specialty_node)

        # ── Get direct services (top 10) ─────────────────────
        direct_services = get_direct_services(conn, specialty_node)[:10]

        # ── Get inherited services (top 5) ───────────────────
        inherited = get_inherited_services(conn, specialty_node)[:5]

        # ── Build context string ─────────────────────────────
        context_lines = []
        context_lines.append("=" * 50)
        context_lines.append("GRAPH RAG — KNOWLEDGE GRAPH CONTEXT")
        context_lines.append("=" * 50)

        context_lines.append(f"\nSPECIALTY NODE FOUND: {specialty_node}")

        # Service check
        context_lines.append(f"\nSERVICE CHECK: '{service}'")
        if service_allowed:
            context_lines.append(
                f"  ✅ ALLOWED — source: {service_source}"
            )
        else:
            context_lines.append(
                f"  ❌ NOT FOUND in specialty or inherited permissions"
            )

        # Diagnosis check
        context_lines.append(f"\nDIAGNOSIS CHECK: '{diagnosis}'")
        if diagnosis_valid:
            context_lines.append(
                f"  ✅ VALID — source: {diagnosis_source}"
            )
        else:
            context_lines.append(
                f"  ⚠️  NOT IN standard list for this specialty"
            )

        # Direct services
        if direct_services:
            context_lines.append(
                f"\nDIRECT ALLOWED SERVICES (sample):\n  "
                + ", ".join(direct_services)
            )

        # Inherited services
        if inherited:
            context_lines.append("\nINHERITED SERVICES (General Physician):")
            for svc, from_node in inherited:
                context_lines.append(f"  - {svc} (via {from_node})")

        # Fraud rules
        if fraud_rules:
            context_lines.append("\nRELEVANT FRAUD RULES:")
            for rule in fraud_rules[:5]:
                context_lines.append(f"  ⚠️  {rule}")

        # Inheritance note
        context_lines.append(
            "\nNOTE: This specialty inherits general physician "
            "permissions. Basic services (CBC, BMP, ESR etc.) "
            "are allowed regardless of specialty mismatch."
        )

        context_lines.append("=" * 50)

        graph_context = "\n".join(context_lines)
        print(f"✅ Graph context retrieved for: {specialty_node}")
        return graph_context

    except Exception as e:
        print(f"⚠️  Graph retrieval error: {e}")
        return f"[Graph RAG] Graph retrieval failed: {str(e)}\n"