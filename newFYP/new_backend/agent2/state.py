# ============================================================
# agent2/state.py — Agent 2 Shared Whiteboard
#
# FLOW:
#   Node 1 (load_node)      → loads cached data into state
#   Node 2 (score_node)     → fetches trust scores + review_stars
#                             from DB, updates RL Q-values
#   Node 3 (recommend_node) → hybrid scoring, returns top N doctors
#
# TWO DOCTOR GROUPS IN THIS SYSTEM:
#
#   "reviews" doctors (256 unique):
#     Real named doctors from the hospital reviews dataset.
#     Multiple rows per doctor in CSV (different patients).
#     avg_rating computed from real patient reviews.
#     collab_score works for them (they appear in pivot matrix).
#     review_stars in DB = not needed (CSV rating is richer).
#
#   "agent1" doctors (118 unique, doc_01..doc_118):
#     FYP doctors from FYPDATA.csv — these are the same doctors
#     that Agent 1 runs fraud detection on.
#     One row per doctor in Reviews_new.csv.
#     avg_rating defaults to 3.0 in profiles (neutral).
#     collab_score = 0 (not in pivot — no multi-patient review data).
#     review_stars from DB overrides the 3.0 at recommendation time.
#     trust_score from DB is their main differentiator.
# ============================================================

from typing import TypedDict, Any


class Agent2State(TypedDict):

    # ── INPUTS ────────────────────────────────────────────────
    patient_input: dict
    # {
    #   "patient_id"        : str   — "E1", "P5", any format
    #   "required_specialty": str   — "Cardiology" (optional)
    #   "max_fee"           : int   — max fee in PKR (optional)
    #   "top_n"             : int   — how many doctors to return (default 5)
    # }

    fraud_context: dict
    # Passed from Agent 3 after Agent 1 completes.
    # Contains fraud detection result for the transaction doctor.
    # {
    #   "doctor_id"    : str   — "doc_64"
    #   "doctor_score" : float — current trust score e.g. 45.0
    #   "doctor_status": str   — "TRUSTED" / "WATCH LIST" / "FLAGGED" / "SUSPENDED"
    #   "is_fraud"     : bool
    #   "overall_risk" : str   — "HIGH RISK" / "MEDIUM RISK" / "NORMAL"
    # }

    # ── NODE 1 OUTPUT: Cached Data ────────────────────────────
    # Written by load_node. Read by score_node and recommend_node.

    df: Any
    # Full Reviews_new.csv DataFrame after load_and_clean().
    # 647 rows, 374 unique doctors.
    # 'source' column added: 'reviews' or 'agent1'.

    profiles: Any
    # One row per doctor — doctor feature profile.
    # Columns: Doctor_Name, avg_rating, total_visits, avg_fee,
    #          rating_std, department, hospital,
    #          dept_encoded, hospital_encoded, source
    # doc_XX doctors have avg_rating=3.0 (neutral default).
    # recommend_node overrides avg_rating with DB review_stars for doc_XX.

    pivot: Any
    # Patient × Doctor rating matrix (pandas DataFrame).
    # Built only from reviewed rows (real Reviews, not NaN).
    # Rows = Employee_IDs, Columns = Doctor_Names.
    # doc_XX doctors NOT in pivot — collab_score = 0 for them.

    patient_clusters: dict
    # { "E1": 2, "E5": 0, ... }
    # Maps each patient_id to their bi-cluster number.
    # Used by recommend_node collab_score to find similar patients.

    doctor_clusters: dict
    # { "Dr. Ali Khan": 3, "doc_64": N/A, ... }
    # Maps each doctor to their bi-cluster number.
    # Only contains real doctors (from pivot columns).
    # doc_XX doctors not here — collab_score returns 0 for them.

    # ── NODE 2 OUTPUT: Trust + Quality Scores from DB ─────────
    # Written by score_node. Read by recommend_node.

    doctor_trust_scores: dict
    # { "doc_64": 45.0, "doc_01": 100.0, "Dr. Ali": 100.0, ... }
    # Agent 1 fraud trust score per doctor (0–100).
    # Only doc_XX doctors appear here (real doctors not in doctor_scores table).
    # Real doctors default to 100.0 in recommend_node (no fraud history).
    # Doctors below 40 → SUSPENDED → excluded from recommendations entirely.
    # Used as 15% of hybrid score.

    doctor_review_stars: dict
    # { "doc_64": 4.0, "doc_01": 3.0, ... }
    # Star rating from DB (doctor_scores.review_stars column).
    # Seeded by seed_review_stars.py from Reviews_new.csv.
    # Only contains doc_XX doctors (real doctors not in doctor_scores table).
    #
    # HOW THIS IS USED IN recommend_node:
    #   For doc_XX doctors:
    #     If doctor_id is in this dict → use this star as avg_rating
    #     This overrides the static 3.0 default from profiles.
    #     Result: a doc_XX with 4 stars ranks higher than one with 2 stars,
    #     independent of their fraud trust score.
    #
    #   For real named doctors:
    #     Ignored — their CSV avg_rating (from real multi-row reviews) is used.
    #     Real reviews are richer and more accurate than a single DB value.
    #
    # Default: 5.0 for any doc_XX not in DB (safe default = no penalty)

    # ── NODE 3 OUTPUT: Recommendation Results ─────────────────
    # Written by recommend_node. Returned to Agent 3.

    recommended_doctors: list
    # Top N doctors sorted by hybrid_score descending.
    # Each dict contains:
    # {
    #   "Doctor_Name"  : str    — e.g. "doc_64" or "Dr. Ali Khan"
    #   "Department"   : str    — specialty
    #   "Hospital"     : str    — hospital name
    #   "Avg_Fee"      : int|None — None for doc_XX (fee unknown)
    #   "Avg_Rating"   : float  — effective rating (DB stars for doc_XX,
    #                             CSV avg for real doctors)
    #   "Total_Visits" : int    — number of review rows in dataset
    #   "Hybrid_Score" : float  — weighted sum of 4 components
    #   "Trust_Score"  : float  — Agent 1 fraud score (100 if not in DB)
    #   "Trust_Status" : str    — TRUSTED/WATCH LIST/FLAGGED/SUSPENDED
    #   "Source"       : str    — "agent1" or "reviews"
    #   "RL_Top_Pick"  : str    — "⭐ RL Pick" or ""
    # }

    rl_top_pick: str
    # Doctor name that RL agent selected (explore or exploit).

    recommendation_note: str
    # Human-readable summary of the top recommendation.

    fraud_warning: str
    # Warning message about the transaction doctor's fraud status.
    # Empty string if doctor is TRUSTED.