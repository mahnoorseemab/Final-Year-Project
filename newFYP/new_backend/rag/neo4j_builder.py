# ============================================================
# rag/neo4j_builder.py
# GRAPH RAG — Knowledge Graph Builder
#
# WHAT THIS DOES:
#   - Reads policies_knowledge_base.txt
#   - Automatically parses ALL sections (including General Physician)
#   - Builds Neo4j knowledge graph with relationships
#   - Connects all specialties to General Physician via INHERITS
#
# RUN ONCE — like kb_loader.py
# Command: python rag/neo4j_builder.py
# ============================================================

import os
import re
from neo4j import GraphDatabase
import neo4j
from dotenv import load_dotenv
load_dotenv()

# ── Neo4j Credentials ────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ── KB Path ──────────────────────────────────────────────────
KB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "knowledge_base",
    "policies_knowledge_base.txt"
)


# ============================================================
# NEO4J CONNECTION
# ============================================================

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("✅ Neo4j connected!")

    def close(self):
        self.driver.close()

    def run(self, query, parameters=None):
        with self.driver.session() as session:
            return list(session.run(query, parameters or {}))


# ============================================================
# PARSE KNOWLEDGE BASE
# ============================================================

def parse_knowledge_base(kb_path):
    """
    Reads KB file and extracts ALL sections automatically.
    Works for all 35 specialties + General Physician (Section 36).
    Returns list of dicts with specialty, diagnoses, services, frauds.
    """
    with open(kb_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections    = []
    raw_blocks  = re.split(r'={40,}', content)

    current_specialty = None
    current_diagnoses = []
    current_services  = []
    current_frauds    = []
    mode              = None

    for block in raw_blocks:
        lines = block.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # ── Detect new SECTION ──────────────────────────
            if re.match(r'SECTION\s+\d+\s*:', line, re.IGNORECASE):
                # Save previous section
                if current_specialty:
                    sections.append({
                        "specialty" : current_specialty,
                        "diagnoses" : current_diagnoses,
                        "services"  : current_services,
                        "frauds"    : current_frauds
                    })
                # Extract specialty name
                parts = line.split(":", 1)
                current_specialty = parts[1].strip() if len(parts) > 1 else None
                current_diagnoses = []
                current_services  = []
                current_frauds    = []
                mode              = None

            # ── Detect mode ─────────────────────────────────
            elif re.search(r'ALLOWED CONDITIONS', line, re.IGNORECASE):
                mode = "diagnoses"
            elif re.search(r'ALLOWED SERVICES', line, re.IGNORECASE):
                mode = "services"
            elif re.search(r'FRAUD RULES', line, re.IGNORECASE):
                mode = "frauds"
            elif re.search(r'NOT COVERED|NOTE FOR GRAPH', line, re.IGNORECASE):
                mode = None

            # ── Extract items ────────────────────────────────
            elif line.startswith("- ") and mode:
                item = line[2:].strip()
                if item:
                    if mode == "diagnoses":
                        current_diagnoses.append(item)
                    elif mode == "services":
                        current_services.append(item)
                    elif mode == "frauds":
                        current_frauds.append(item)

    # Save last section
    if current_specialty:
        sections.append({
            "specialty" : current_specialty,
            "diagnoses" : current_diagnoses,
            "services"  : current_services,
            "frauds"    : current_frauds
        })

    print(f"✅ Parsed {len(sections)} sections from KB")
    for s in sections:
        print(f"   → {s['specialty']} | "
              f"diagnoses: {len(s['diagnoses'])} | "
              f"services: {len(s['services'])} | "
              f"frauds: {len(s['frauds'])}")
    return sections

# ============================================================
# BUILD GRAPH
# ============================================================

def clear_graph(conn):
    conn.run("MATCH (n) DETACH DELETE n")
    print("✅ Graph cleared!")


def build_full_graph(conn, sections):
    """
    For every section (including General Physician):
    - Create Specialty node
    - Create Diagnosis nodes + TREATS relationships
    - Create Service nodes + CAN_ORDER relationships
    - Create FraudRule nodes + FRAUD_IF_NO relationships

    Then connect all non-general specialties to
    General Physician via INHERITS relationship.
    """

    # ── Find General Physician section ──────────────────────
    general_physician_name = None
    for section in sections:
        if "GENERAL PHYSICIAN" in section["specialty"].upper():
            general_physician_name = section["specialty"]
            break

    if not general_physician_name:
        print("⚠️  WARNING: General Physician section not found in KB!")
        print("   Make sure Section 36 is added to policies_knowledge_base.txt")

    # ── Build graph for ALL sections ────────────────────────
    for section in sections:
        specialty = section["specialty"]
        print(f"  Building: {specialty}")

        # Create Specialty node
        conn.run(
            "MERGE (sp:Specialty {name: $specialty})",
            {"specialty": specialty}
        )

        # Diagnosis nodes + TREATS
        for diagnosis in section["diagnoses"]:
            conn.run("""
                MERGE (d:Diagnosis {name: $diagnosis})
                WITH d
                MATCH (sp:Specialty {name: $specialty})
                MERGE (sp)-[:TREATS]->(d)
            """, {"specialty": specialty, "diagnosis": diagnosis})

        # Service nodes + CAN_ORDER
        for service in section["services"]:
            conn.run("""
                MERGE (s:Service {name: $service})
                WITH s
                MATCH (sp:Specialty {name: $specialty})
                MERGE (sp)-[:CAN_ORDER]->(s)
            """, {"specialty": specialty, "service": service})

        # FraudRule nodes + FRAUD_IF_NO
        for fraud in section["frauds"]:
            conn.run("""
                MERGE (fr:FraudRule {description: $fraud})
                WITH fr
                MATCH (sp:Specialty {name: $specialty})
                MERGE (sp)-[:FRAUD_IF_NO]->(fr)
            """, {"specialty": specialty, "fraud": fraud})

    print(f"✅ All {len(sections)} sections added to graph!")

    # ── INHERITS: all specialties → General Physician ───────
    if general_physician_name:
        for section in sections:
            specialty = section["specialty"]
            if specialty == general_physician_name:
                continue
            conn.run("""
                MATCH (sp:Specialty {name: $specialty})
                MATCH (gp:Specialty {name: $gp_name})
                MERGE (sp)-[:INHERITS]->(gp)
            """, {
                "specialty": specialty,
                "gp_name"  : general_physician_name
            })
        print(f"✅ INHERITS relationships created → {general_physician_name}")


def create_indexes(conn):
    conn.run("CREATE INDEX specialty_name IF NOT EXISTS FOR (n:Specialty) ON (n.name)")
    conn.run("CREATE INDEX service_name IF NOT EXISTS FOR (n:Service) ON (n.name)")
    conn.run("CREATE INDEX diagnosis_name IF NOT EXISTS FOR (n:Diagnosis) ON (n.name)")
    print("✅ Indexes created!")


def print_stats(conn):
    print("\n📊 Graph Statistics:")
    result = conn.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC")
    for record in result:
        print(f"   {record['label']}: {record['count']} nodes")

    print("📊 Relationships:")
    result = conn.run("MATCH ()-[r]->() RETURN type(r) as rel, count(r) as count ORDER BY count DESC")
    for record in result:
        print(f"   {record['rel']}: {record['count']}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("Graph RAG — Neo4j Knowledge Graph Builder")
    print("=" * 55)

    conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    print("\n[1/4] Clearing existing graph...")
    clear_graph(conn)

    print("\n[2/4] Parsing knowledge base...")
    sections = parse_knowledge_base(KB_PATH)

    print("\n[3/4] Building knowledge graph...")
    build_full_graph(conn, sections)

    print("\n[4/4] Creating indexes...")
    create_indexes(conn)

    print_stats(conn)
    conn.close()

    print("\n" + "=" * 55)
    print("✅ Knowledge Graph ready in Neo4j!")
    print("Open Neo4j Aura → Explore to see your graph!")
    print("=" * 55)