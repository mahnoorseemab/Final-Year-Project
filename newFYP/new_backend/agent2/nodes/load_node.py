# ============================================================
# agent2/nodes/load_node.py — NODE 1 OF AGENT 2
#
# WHAT IT DOES:
#   Injects the globally cached recommendation data into the
#   LangGraph state so score_node and recommend_node can read it.
#
#   The cache was built ONCE at FastAPI startup by
#   initialize_agent2_cache() in agent2_graph.py.
#   This node does NOT reload anything from disk — it just
#   passes references to the already-built in-memory objects.
#
# WHY A SEPARATE NODE FOR THIS?
#   LangGraph nodes must return dict updates to state.
#   The cached objects (df, profiles, pivot, clusters) need to
#   be in state so subsequent nodes can access them as state fields
#   rather than importing globals directly.
#   This keeps each node's dependencies explicit and testable.
#
# DATA IN CACHE (built by initialize_agent2_cache):
#   _cached_df               : full Reviews_new.csv DataFrame (647 rows)
#   _cached_profiles         : one profile row per doctor (374 rows)
#   _cached_pivot            : patient×doctor matrix (reviewed rows only)
#   _cached_patient_clusters : { patient_id: cluster_number }
#   _cached_doctor_clusters  : { doctor_name: cluster_number }
#   _cached_rl_agent         : EpsilonGreedyRecommender (374 doctors)
#
# NOTE ON rl_agent:
#   The RL agent is NOT passed into state — it stays in the global
#   cache and is accessed directly by score_node and recommend_node
#   via import. This is because LangGraph serialises state between
#   nodes and EpsilonGreedyRecommender is not JSON-serialisable.
#   Accessing it as a global is intentional and correct here.
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def load_node(state: dict) -> dict:
    """
    LangGraph Node 1 — Data Loader.

    Reads from global cache (built at startup).
    Writes df, profiles, pivot, patient_clusters, doctor_clusters
    into state for use by score_node and recommend_node.

    Raises RuntimeError if initialize_agent2_cache() was not called.
    """

    print("\n" + "=" * 55)
    print("AGENT 2 — NODE 1: DATA LOADER")
    print("=" * 55)

    from agent2.agent2_graph import (
        _cached_df,
        _cached_profiles,
        _cached_pivot,
        _cached_patient_clusters,
        _cached_doctor_clusters,
    )

    if _cached_df is None:
        raise RuntimeError(
            "Agent 2 cache is empty! "
            "initialize_agent2_cache() must be called at FastAPI startup "
            "before any recommendation requests."
        )

    total_doctors   = _cached_profiles['Doctor_Name'].nunique()
    reviews_doctors = (_cached_profiles['source'] == 'reviews').sum()
    agent1_doctors  = (_cached_profiles['source'] == 'agent1').sum()
    specialties     = _cached_profiles['department'].nunique()

    print(f"  Total doctors      : {total_doctors}")
    print(f"    Reviews doctors  : {reviews_doctors} (real named — have real patient reviews)")
    print(f"    Agent1 doctors   : {agent1_doctors} (doc_XX — fraud detection dataset)")
    print(f"  Specialties        : {specialties}")
    print(f"  Patients clustered : {len(_cached_patient_clusters)}")
    print(f"  Doctors in pivot   : {len(_cached_doctor_clusters)} (reviewed only)")
    print(f"  Pivot shape        : {_cached_pivot.shape}")

    return {
        'df'              : _cached_df,
        'profiles'        : _cached_profiles,
        'pivot'           : _cached_pivot,
        'patient_clusters': _cached_patient_clusters,
        'doctor_clusters' : _cached_doctor_clusters,
    }