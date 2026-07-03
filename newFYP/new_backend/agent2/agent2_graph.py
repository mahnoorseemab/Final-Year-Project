# ============================================================
# agent2/agent2_graph.py — AGENT 2 MAIN FILE
#
# FLOW:
#   load → score → recommend → END
#
#   load_node      : passes globally cached data into state
#   score_node     : reads doctor_scores DB table — gets both
#                    current_score (trust) and review_stars
#                    (quality); adjusts RL Q-values by trust level
#   recommend_node : hybrid scoring (content + collab + RL + trust)
#                    uses DB review_stars for doc_XX doctors instead
#                    of static CSV avg_rating=3.0
#
# INITIALIZATION:
#   initialize_agent2_cache(filepath) must be called ONCE at
#   FastAPI startup (in lifespan). It loads Reviews_new.csv,
#   builds profiles, runs biclustering, pre-trains RL agent,
#   and stores everything in module-level globals.
#
# CHANGES vs previous version:
#   - Agent2State now imported from agent2.state (same filename,
#     updated class with doctor_review_stars field)
#   - get_all_recommendation_doctors() now includes review_stars
#     so the dashboard can show live DB stars for doc_XX doctors
#   - initialize_agent2_cache() prints cleaner startup summary
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from agent2.state             import Agent2State
from agent2.nodes.load_node      import load_node
from agent2.nodes.score_node     import score_node
from agent2.nodes.recommend_node import recommend_node

# ── Global Cache — built ONCE at startup, reused every request ─
_cached_df               = None
_cached_profiles         = None
_cached_pivot            = None
_cached_patient_clusters = None
_cached_doctor_clusters  = None
_cached_rl_agent         = None
_agent2_ready            = False


def initialize_agent2_cache(filepath: str):
    """
    Called ONCE at FastAPI startup (in lifespan).

    Loads Reviews_new.csv which has two doctor groups:
      - 256 real named doctors  (rows 0–528)  — have real patient reviews
      - 118 FYP doc_XX doctors  (rows 529–646) — have realistic stars
                                                  from seed_review_stars.py

    Steps:
      1. load_and_clean()       → adds 'source' column, fills NaN Employee_ID
      2. build_doctor_profiles() → one profile row per doctor
                                   doc_XX get avg_rating=3.0 (neutral default)
                                   real doctors get real avg from their reviews
      3. apply_biclustering()   → Spectral BiClustering on reviewed rows only
                                   gives patient_clusters + doctor_clusters
      4. EpsilonGreedyRecommender → initialized for ALL 374 doctors
                                    doc_XX start at Q=0.6 (neutral, not 0.0)
      5. Pre-train RL            → only on rows with real Reviews (not NaN)
    """
    global _cached_df, _cached_profiles, _cached_pivot
    global _cached_patient_clusters, _cached_doctor_clusters
    global _cached_rl_agent, _agent2_ready

    print("=" * 55)
    print("AGENT 2 — Initializing Recommendation System...")
    print(f"  Loading: {filepath}")
    print("=" * 55)

    sys.path.insert(0, os.path.dirname(os.path.abspath(filepath)))
    from recommendation import (
        load_and_clean,
        build_doctor_profiles,
        apply_biclustering,
        EpsilonGreedyRecommender,
    )

    # ── Step 1: Load & clean ──────────────────────────────────
    print("  Step 1: Loading and cleaning CSV...")
    df = load_and_clean(filepath)
    print(f"    Total rows       : {len(df)}")
    print(f"    Unique doctors   : {df['Doctor_Name'].nunique()}")
    print(f"    Reviews doctors  : {(df['source'] == 'reviews').sum()} rows")
    print(f"    Agent1 doctors   : {(df['source'] == 'agent1').sum()} rows")

    # ── Step 2: Build profiles ────────────────────────────────
    # profiles_raw has all 374 doctors.
    # doc_XX get avg_rating=3.0 (neutral) from build_doctor_profiles.
    # recommend_node will OVERRIDE this with DB review_stars at runtime
    # for doc_XX doctors — so the 3.0 here is just a safe default.
    print("  Step 2: Building doctor profiles...")
    profiles_raw, le_dept, le_hosp = build_doctor_profiles(df)

    # Merge source column into profiles (reviews vs agent1)
    source_map = (
        df.groupby("Doctor_Name")["source"]
        .first()
        .reset_index()
    )
    profiles = profiles_raw.merge(source_map, on="Doctor_Name", how="left")
    profiles["source"] = profiles["source"].fillna("reviews")

    print(f"    Profiles built   : {len(profiles)}")
    print(f"    Reviews profiles : {(profiles['source'] == 'reviews').sum()}")
    print(f"    Agent1 profiles  : {(profiles['source'] == 'agent1').sum()}")

    # ── Step 3: Bi-clustering ─────────────────────────────────
    # apply_biclustering() filters to reviewed rows internally.
    # doc_XX doctors (NaN Reviews) are excluded from biclustering —
    # they get collab_score=0 in recommend_node, which is correct
    # since we have no real patient-doctor interaction data for them.
    # Their content score + trust score carry their recommendations.
    print("  Step 3: Applying Spectral Bi-Clustering...")
    patient_clusters, doctor_clusters, pivot = apply_biclustering(df, n_clusters=5)
    print(f"    Patients clustered : {len(patient_clusters)}")
    print(f"    Doctors clustered  : {len(doctor_clusters)}")
    print(f"    Pivot shape        : {pivot.shape}")

    # ── Step 4: Initialize RL agent ───────────────────────────
    # All 374 doctors are in the RL agent.
    # doc_XX start at Q=0.6 (neutral 3/5) — not 0.0.
    # Real doctors start at Q=0.0, then get updated in Step 5.
    print("  Step 4: Initializing Epsilon-Greedy RL Agent...")
    rl_agent = EpsilonGreedyRecommender(
        doctor_names=profiles["Doctor_Name"].tolist(),
        epsilon=0.1
    )
    print(f"    Doctors in RL    : {len(rl_agent.q_values)}")
    print(f"    doc_XX Q start   : 0.6 (neutral)")
    print(f"    Real doc Q start : 0.0 (updated in Step 5)")

    # ── Step 5: Pre-train RL on existing reviews ──────────────
    # Only rows with real Reviews (not NaN).
    # Each row = one patient visited one doctor → reward = stars/5.
    # This builds up Q-values for real doctors from real data.
    # doc_XX rows are skipped here — their Q-values come from
    # score_node at runtime (adjusted by trust score from DB).
    print("  Step 5: Pre-training RL on existing reviews...")
    reviewed = df[df["Reviews"].notna()]
    for _, row in reviewed.iterrows():
        rl_agent.update(row["Doctor_Name"], row["Reviews"] / 5.0)
    print(f"    RL trained on    : {len(reviewed)} review records")

    # ── Store in globals ──────────────────────────────────────
    _cached_df               = df
    _cached_profiles         = profiles
    _cached_pivot            = pivot
    _cached_patient_clusters = patient_clusters
    _cached_doctor_clusters  = doctor_clusters
    _cached_rl_agent         = rl_agent
    _agent2_ready            = True

    print("=" * 55)
    print("✅ Agent 2 ready!")
    print(f"   Total doctors  : {profiles['Doctor_Name'].nunique()}")
    print(f"   Specialties    : {profiles['department'].nunique()}")
    print(f"   RL Q-table     : {len(rl_agent.q_values)} entries")
    print("=" * 55)
    print()
    print("NOTE: doc_XX avg_rating starts at 3.0 (neutral default).")
    print("      recommend_node overrides it with DB review_stars at")
    print("      runtime — run seed_review_stars.py first if not done.")


def update_rl_feedback(doctor_name: str, actual_rating: float):
    """
    Called from POST /feedback endpoint.
    Patient visited a recommended doctor and gave a real rating.
    Updates the RL Q-value for that doctor via incremental mean.

    Args:
        doctor_name  : e.g. "doc_64" or "Dr. Ali Khan"
        actual_rating: 1.0 – 5.0 (patient's star rating)
    """
    if _cached_rl_agent is None:
        raise RuntimeError("Agent 2 not initialized! Call initialize_agent2_cache() first.")

    reward = actual_rating / 5.0
    _cached_rl_agent.update(doctor_name, reward)
    new_q = _cached_rl_agent.q_values.get(doctor_name, 0.0)
    print(f"✅ RL feedback: {doctor_name} rated {actual_rating}/5 → Q={new_q:.4f}")


def get_available_specialties() -> list:
    """Returns sorted list of all specialties for frontend dropdowns."""
    if _cached_profiles is None:
        return []
    return sorted(_cached_profiles["department"].unique().tolist())


def get_all_recommendation_doctors() -> list:
    """
    Returns all doctor profiles for dashboard display.
    Includes source field so frontend can distinguish
    Agent1 (doc_XX) doctors from real named doctors.
    """
    if _cached_profiles is None:
        return []
    cols = [
        "Doctor_Name", "department", "hospital",
        "avg_rating", "avg_fee", "total_visits", "source"
    ]
    return _cached_profiles[cols].to_dict(orient="records")


# ── Build LangGraph ───────────────────────────────────────────

def build_agent2_graph():
    """
    Compiles the Agent 2 LangGraph.
    Flow: load → score → recommend → END

    load_node      : injects cached data into state
    score_node     : reads DB for trust scores + review_stars,
                     adjusts RL Q-values by fraud status
    recommend_node : hybrid scoring returns top N doctors
    """
    graph = StateGraph(Agent2State)

    graph.add_node("load",      load_node)
    graph.add_node("score",     score_node)
    graph.add_node("recommend", recommend_node)

    graph.set_entry_point("load")
    graph.add_edge("load",      "score")
    graph.add_edge("score",     "recommend")
    graph.add_edge("recommend", END)

    compiled = graph.compile()

    print("✅ Agent 2 graph compiled!")
    print("   Flow: load → score → recommend → END")
    return compiled


agent2_graph = build_agent2_graph()


def run_agent2(patient_input: dict, fraud_context: dict, db=None) -> dict:
    """
    Main entry point for Agent 2.
    Called by Agent 3's recommendation_node after Agent 1 completes.

    Args:
        patient_input  : patient request dict
            {
              "patient_id"        : "E1" or "P123"
              "required_specialty": "Cardiology"   (optional)
              "max_fee"           : 3000            (optional)
              "top_n"             : 5               (default 5)
            }

        fraud_context  : Agent 1 result for the transaction doctor
            {
              "doctor_id"    : "doc_64"
              "doctor_score" : 45.0
              "doctor_status": "FLAGGED"
              "is_fraud"     : True
              "overall_risk" : "HIGH RISK"
            }

        db  : SQLAlchemy session (passed from FastAPI endpoint)
              Used by score_node to query doctor_scores table.

    Returns:
        final LangGraph state dict with:
          recommended_doctors, rl_top_pick,
          recommendation_note, fraud_warning
    """
    if not _agent2_ready:
        raise RuntimeError(
            "Agent 2 not initialized! "
            "Call initialize_agent2_cache() at FastAPI startup."
        )

    print("\n" + "=" * 55)
    print("AGENT 2 (RECOMMENDATION) STARTING")
    print(f"  Patient   : {patient_input.get('patient_id', 'Unknown')}")
    print(f"  Specialty : {patient_input.get('required_specialty', 'Any')}")
    print(f"  Top N     : {patient_input.get('top_n', 5)}")
    print(f"  Fraud Doc : {fraud_context.get('doctor_id', 'N/A')} "
          f"({fraud_context.get('doctor_status', 'N/A')})")
    print("=" * 55)

    initial_state = {
        "patient_input": patient_input,
        "fraud_context": fraud_context,
        "_db"          : db,
    }

    final_state = agent2_graph.invoke(initial_state)

    print("\n" + "=" * 55)
    print("AGENT 2 COMPLETE")
    recs = final_state.get('recommended_doctors', [])
    print(f"  Recommendations : {len(recs)}")
    print(f"  RL Top Pick     : {final_state.get('rl_top_pick', 'N/A')}")
    if recs:
        best = recs[0]
        print(f"  Best Doctor     : {best['Doctor_Name']} "
              f"(Rating:{best['Avg_Rating']}, "
              f"Trust:{best['Trust_Score']:.0f}, "
              f"Hybrid:{best['Hybrid_Score']})")
    print("=" * 55)

    return final_state