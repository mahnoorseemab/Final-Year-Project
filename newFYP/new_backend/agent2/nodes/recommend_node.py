# ============================================================
# agent2/nodes/recommend_node.py — NODE 3 OF AGENT 2
#
# SCORING FORMULA (4 components):
#   1. Content Score  (35%): doctor quality from profile
#   2. Collab Score   (35%): similar patients' ratings
#   3. RL Score       (15%): epsilon-greedy Q-value
#   4. Trust Score    (15%): Agent 1 fraud trust score (0–100 → 0–1)
#
# CHANGE: review_stars from DB now used for doc_XX doctors.
#
#   HOW avg_rating IS NOW DETERMINED:
#     doc_XX doctors  → avg_rating = review_stars from DB
#                       (seeded from Reviews_new.csv, updated by fraud
#                        penalties via scoring_node in Agent 1)
#     Real doctors    → avg_rating = from CSV profiles (unchanged)
#                       (real named doctors have rich multi-row review
#                        data from real patients — that is more accurate
#                        than a single DB star value)
#
#   WHY THIS MATTERS:
#     Before this change, a fraud-penalised doc_XX doctor still had
#     avg_rating=3.0 (neutral from CSV default) in their content score.
#     Only the 15% trust weight was lowering their hybrid score.
#     Now their content score (35% weight) also reflects the star
#     rating from DB — which is tied to the real distribution
#     (mostly 3–4 stars from seed_review_stars.py).
#     A doctor caught committing fraud with review_stars=3.0 will
#     score lower in content than a clean doctor with stars=4.0.
#
# SUSPENDED DOCTORS:
#   trust_score < 40 → excluded completely. Never appear in results.
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def recommend_node(state: dict) -> dict:

    print("\n" + "=" * 55)
    print("AGENT 2 — NODE 3: RECOMMENDATION ENGINE")
    print("=" * 55)

    patient_input    = state.get('patient_input', {})
    fraud_context    = state.get('fraud_context', {})
    profiles         = state['profiles']
    pivot            = state['pivot']
    patient_clusters = state['patient_clusters']
    doctor_clusters  = state['doctor_clusters']
    trust_scores     = state.get('doctor_trust_scores', {})
    review_stars_db  = state.get('doctor_review_stars', {})   # ← NEW

    from agent2.agent2_graph import _cached_rl_agent
    rl_agent = _cached_rl_agent

    # ── Patient inputs ────────────────────────────────────────
    patient_id         = str(patient_input.get('patient_id', ''))
    required_specialty = patient_input.get('required_specialty', None) or None
    max_fee            = patient_input.get('max_fee', None) or None
    top_n              = int(patient_input.get('top_n', 5))

    print(f"  Patient ID         : {patient_id}")
    print(f"  Required Specialty : {required_specialty or 'Any'}")
    print(f"  Max Fee            : {max_fee or 'No limit'}")

    # ── Fraud context ─────────────────────────────────────────
    fraud_doctor_id = fraud_context.get('doctor_id', '')
    fraud_score     = fraud_context.get('doctor_score', 100.0)
    fraud_status    = fraud_context.get('doctor_status', 'TRUSTED')

    print(f"  Transaction Doctor : {fraud_doctor_id} ({fraud_status})")

    # ── Filter candidates ─────────────────────────────────────
    if required_specialty:
        candidates_df = profiles[
    profiles['department'].str.strip().str.lower() == required_specialty.strip().lower()
]
    else:
        candidates_df = profiles.copy()

    # Exclude SUSPENDED doctors (trust_score < 40)
    suspended = {d for d, s in trust_scores.items() if s < 40}
    candidates_df = candidates_df[~candidates_df['Doctor_Name'].isin(suspended)]

    # Apply fee filter (skip if fee=0, meaning unknown)
    if max_fee:
        candidates_df = candidates_df[
            (candidates_df['avg_fee'] <= max_fee) |
            (candidates_df['avg_fee'] == 0)
        ]

    candidates = candidates_df['Doctor_Name'].tolist()
    print(f"  Candidates after filters : {len(candidates)}")

    if not candidates:
        print("  ⚠️  No candidates found!")
        return {
            'recommended_doctors': [],
            'rl_top_pick'        : '',
            'recommendation_note': f"No doctors found for '{required_specialty}'. Try a different specialty.",
            'fraud_warning'      : _build_fraud_warning(fraud_doctor_id, fraud_status, fraud_score),
        }

    # ── Score every candidate ─────────────────────────────────
    scored = []
    for doc_name in candidates:
        row = profiles[profiles['Doctor_Name'] == doc_name].iloc[0]

        # ── Determine effective avg_rating ────────────────────
        # For doc_XX (FYP/Agent1 doctors): use DB review_stars if available.
        # These doctors have only one synthetic profile row in CSV —
        # the DB star is the better quality signal for them.
        #
        # For real named doctors: use CSV avg_rating (multi-row real data).
        is_fyp_doctor = str(doc_name).startswith('doc_')
        if is_fyp_doctor and doc_name in review_stars_db:
            effective_rating = review_stars_db[doc_name]
        else:
            effective_rating = float(row['avg_rating'])

        # 1. Content score (using effective rating)
        cb = _content_score(row, effective_rating)

        # 2. Collaborative score
        cf = _collab_score(
            patient_id, doc_name, pivot,
            patient_clusters, doctor_clusters
        )

        # 3. RL score
        rl = rl_agent.q_values.get(doc_name, 0.0)

        # 4. Trust score (0–1)
        raw_trust = trust_scores.get(doc_name, 100.0)
        ts = raw_trust / 100.0

        # Weighted hybrid
        hybrid = round(
            0.35 * cb +
            0.35 * cf +
            0.15 * rl +
            0.15 * ts,
            4
        )

        if hybrid <= 0:
            continue

        status = _score_to_status(raw_trust)
        scored.append({
            'Doctor_Name'  : doc_name,
            'Department'   : row['department'],
            'Hospital'     : row['hospital'],
            'Avg_Fee'      : int(row['avg_fee']) if row['avg_fee'] > 0 else None,
            'Avg_Rating'   : round(effective_rating, 2),
            'Total_Visits' : int(row['total_visits']),
            'Hybrid_Score' : hybrid,
            'Trust_Score'  : raw_trust,
            'Trust_Status' : status,
            'Source'       : row.get('source', 'reviews'),
            'RL_Top_Pick'  : '',
        })

    if not scored:
        return {
            'recommended_doctors': [],
            'rl_top_pick'        : '',
            'recommendation_note': "No eligible doctors found after scoring.",
            'fraud_warning'      : _build_fraud_warning(fraud_doctor_id, fraud_status, fraud_score),
        }

    # Sort by hybrid score descending
    scored.sort(key=lambda x: x['Hybrid_Score'], reverse=True)
    top = scored[:top_n]

    # RL explore/exploit
    top_names = [d['Doctor_Name'] for d in top]
    rl_pick   = rl_agent.recommend(top_names)
    for d in top:
        if d['Doctor_Name'] == rl_pick:
            d['RL_Top_Pick'] = '⭐ RL Pick'

    print(f"  Top {len(top)} recommendations:")
    for i, d in enumerate(top, 1):
        pick     = ' ⭐' if d['RL_Top_Pick'] else ''
        src_tag  = ' [A1]' if d['Source'] == 'agent1' else ''
        print(f"    {i}. {d['Doctor_Name']}{src_tag} ({d['Department']}) "
              f"Rating:{d['Avg_Rating']} Trust:{d['Trust_Score']:.0f} "
              f"Hybrid:{d['Hybrid_Score']}{pick}")

    # ── Build recommendation note ─────────────────────────────
    best    = top[0]
    fee_str = f"Rs.{best['Avg_Fee']}" if best['Avg_Fee'] else "Fee N/A"
    recommendation_note = (
        f"Top {len(top)} doctor(s) recommended"
        f"{' for ' + required_specialty if required_specialty else ''}. "
        f"Best match: {best['Doctor_Name']} "
        f"({best['Department']}, {best['Hospital']}) — "
        f"Rating: {best['Avg_Rating']}/5, {fee_str}, "
        f"Trust: {best['Trust_Score']}/100 ({best['Trust_Status']}), "
        f"Hybrid Score: {best['Hybrid_Score']}."
        + (f" RL top pick: {rl_pick}." if rl_pick else "")
    )

    fraud_warning = _build_fraud_warning(fraud_doctor_id, fraud_status, fraud_score)

    return {
        'recommended_doctors': top,
        'rl_top_pick'        : rl_pick or '',
        'recommendation_note': recommendation_note,
        'fraud_warning'      : fraud_warning,
    }


# ── Content-Based Score ───────────────────────────────────────
def _content_score(row, effective_rating: float) -> float:
    """
    Doctor quality score from their profile.
    Uses effective_rating instead of row['avg_rating'] directly —
    this allows DB review_stars to override CSV rating for doc_XX doctors.

    avg_rating  (50%): patient satisfaction signal
    total_visits(30%): popularity / track record
    consistency (20%): low rating variance = reliable doctor
    """
    rating_norm = effective_rating / 5.0
    visit_norm  = min(row['total_visits'] / 50.0, 1.0)
    consistency = 1.0 - min(row.get('rating_std', 0) / 2.0, 1.0)

    return round(
        0.5 * rating_norm +
        0.3 * visit_norm  +
        0.2 * consistency,
        4
    )


# ── Collaborative Filtering Score ────────────────────────────
def _collab_score(
    patient_id, doctor_name, pivot,
    patient_clusters, doctor_clusters
) -> float:
    """
    What did patients in the same bi-cluster rate this doctor?
    Returns 0 if patient unknown or doctor not in pivot.
    doc_XX doctors will typically return 0 here because they
    don't appear in the pivot (no real multi-patient reviews).
    Their content + trust scores carry the recommendation weight.
    """
    if patient_id not in patient_clusters:
        return 0.0
    if doctor_name not in doctor_clusters:
        return 0.0

    my_cluster = patient_clusters[patient_id]
    similar    = [p for p, c in patient_clusters.items()
                  if c == my_cluster and p != patient_id]

    if not similar or doctor_name not in pivot.columns:
        return 0.0

    ratings = pivot.loc[
        [p for p in similar if p in pivot.index],
        doctor_name
    ]
    ratings = ratings[ratings > 0]

    if ratings.empty:
        return 0.0

    return round(float(ratings.mean() / 5.0), 4)


# ── Score → Status Label ──────────────────────────────────────
def _score_to_status(score: float) -> str:
    if score >= 80: return "TRUSTED"
    if score >= 60: return "WATCH LIST"
    if score >= 40: return "FLAGGED"
    return "SUSPENDED"


# ── Fraud Warning ─────────────────────────────────────────────
def _build_fraud_warning(doctor_id, status, score) -> str:
    if status == 'SUSPENDED':
        return (
            f"🚨 CRITICAL: Doctor {doctor_id} is SUSPENDED "
            f"(Trust Score: {score}/100). Do NOT assign patients. "
            f"Please choose from the recommended alternatives below."
        )
    if status == 'FLAGGED':
        return (
            f"⚠️  WARNING: Doctor {doctor_id} is FLAGGED "
            f"(Trust Score: {score}/100). "
            f"Suspicious billing detected. Choose a safer alternative."
        )
    if status == 'WATCH LIST':
        return (
            f"ℹ️  NOTE: Doctor {doctor_id} is on Watch List "
            f"(Trust Score: {score}/100). Monitor billing carefully."
        )
    return ''