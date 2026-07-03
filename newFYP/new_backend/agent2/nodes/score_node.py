# ============================================================
# agent2/nodes/score_node.py — NODE 2 OF AGENT 2
#
# WHAT IT DOES:
#   Bridge between Agent 1 (fraud detection) and Agent 2
#   (recommendation). Reads the doctor_scores DB table and
#   makes trust data available for recommend_node.
#
#   READS FROM DB:
#     current_score  → fraud trust score (0–100)
#     review_stars   → patient satisfaction stars (1.0–5.0)
#                      seeded from Reviews_new.csv by seed_review_stars.py
#                      decremented by fraud penalties in scoring_node (Agent 1)
#
#   WRITES TO STATE:
#     doctor_trust_scores  : { doctor_id: current_score }
#     doctor_review_stars  : { doctor_id: review_stars  }  ← NEW
#
#   WHY BOTH FIELDS?
#     current_score (trust)  = fraud safety signal from Agent 1
#       → used as 15% weight in hybrid score
#       → doctors below 40 are fully suspended from results
#
#     review_stars (quality) = patient satisfaction signal
#       → used in recommend_node to OVERRIDE the static avg_rating
#         from CSV for doc_XX doctors, so fraud-penalised doctors
#         also get a lower content score (not just lower trust score)
#       → real named doctors (non doc_XX) keep their CSV avg_rating
#         since their reviews come from real multi-row patient data
#
#   HOW RL IS ADJUSTED (unchanged from before):
#     SUSPENDED (score < 40) → RL Q = 0.0     (never recommend)
#     FLAGGED   (40–59)      → RL Q × 0.7     (30% reduction)
#     WATCH LIST(60–79)      → RL Q × 0.9     (10% reduction)
#     TRUSTED   (80–100)     → RL Q × 1.1     (10% boost, capped 1.0)
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def score_node(state: dict) -> dict:

    print("\n" + "=" * 55)
    print("AGENT 2 — NODE 2: TRUST SCORE INTEGRATION")
    print("=" * 55)

    db = state.get('_db')

    # ── Step 1: Fetch trust scores AND review stars from DB ───
    trust_scores, review_stars = _fetch_scores_from_db(db)

    print(f"  Trust scores loaded  : {len(trust_scores)} doctors")
    print(f"  Review stars loaded  : {len(review_stars)} doctors")

    # ── Step 2: Update RL Q-values using trust scores ─────────
    from agent2.agent2_graph import _cached_rl_agent

    updated_count = 0
    for doctor_id, score in trust_scores.items():
        if doctor_id not in _cached_rl_agent.q_values:
            # Doctor is in DB but not in RL agent — initialize
            _cached_rl_agent.q_values[doctor_id]     = score / 100.0
            _cached_rl_agent.visit_counts[doctor_id] = 0

        current_q = _cached_rl_agent.q_values[doctor_id]

        if score < 40:
            new_q = 0.0            # SUSPENDED — never recommend
        elif score < 60:
            new_q = current_q * 0.7   # FLAGGED — reduce 30%
        elif score < 80:
            new_q = current_q * 0.9   # WATCH LIST — reduce 10%
        else:
            new_q = min(current_q * 1.1, 1.0)  # TRUSTED — boost 10%

        _cached_rl_agent.q_values[doctor_id] = round(new_q, 4)
        updated_count += 1

    print(f"  RL Q-values adjusted : {updated_count} doctors")

    # Print summary of flagged/suspended
    flagged = {d: s for d, s in trust_scores.items() if s < 60}
    if flagged:
        print(f"  ⚠️  Flagged/Suspended ({len(flagged)}):")
        for d, s in sorted(flagged.items(), key=lambda x: x[1]):
            status = "SUSPENDED" if s < 40 else "FLAGGED"
            print(f"     {d}: {s:.1f}/100 ({status})")

    # Print a few review stars for visibility
    if review_stars:
        sample = dict(list(review_stars.items())[:5])
        print(f"  Review stars sample  : {sample}")

    return {
        'doctor_trust_scores': trust_scores,
        'doctor_review_stars': review_stars,   # ← NEW
    }


def _fetch_scores_from_db(db) -> tuple:
    """
    Reads doctor_scores table.

    Returns:
        trust_scores  : { doctor_id: current_score  }
        review_stars  : { doctor_id: review_stars   }

    Returns ({}, {}) if DB unavailable.
    """
    if db is None:
        print("  ⚠️  No DB session — skipping trust score integration")
        return {}, {}

    try:
        from models_db import DoctorScore
        rows = db.query(DoctorScore).all()

        trust_scores = {}
        review_stars = {}

        for row in rows:
            trust_scores[row.doctor_id] = float(row.current_score)
            # review_stars default 5.0 if column missing or None
            stars = getattr(row, 'review_stars', None)
            review_stars[row.doctor_id] = float(stars) if stars is not None else 5.0

        return trust_scores, review_stars

    except Exception as e:
        print(f"  ⚠️  DB fetch failed: {e}")
        return {}, {}