# ============================================================
# TTSGAN_INFERENCE.PY — UPDATED FOR TTSWGAN___DCTGAN_FINAL.ipynb
# (8-pair-type, single-scalar-feature design)
#
# WHY THIS VERSION:
#   The FINAL training notebook builds 8 separate single-feature
#   sequence types (Chunk 6), combined into one training pool and
#   normalized by ONE shared scalar. The discriminator's
#   input_projection is Linear(1, 64) -- it scores ONE pair-type
#   sequence at a time, shape (1, 10, 1).
#
#   This inference module builds all 8 pair-type tensors for the
#   incoming transaction (using sequence_pairs.py, shared with
#   dctgan_inference.py), scores each one separately, and combines
#   them with a weighted average into a single TTS score -- exactly
#   matching what the discriminator was trained to judge.
#
# REPLACES the previous "4-grouping, 5-feature-per-timestep" design,
# which built (1, 10, 5) tensors that do not match the FINAL
# notebook's trained weight shapes at all.
#
# sequence_tensors IN RETURN DICT:
#   Returns all 8 pair-type tensors (not just one) so
#   investigation_node can run ablation across all 8 pair-types,
#   matching feature_medians.json's 8 keys from the FINAL notebook.
# ============================================================

import torch
from sqlalchemy.orm import Session

from crud import (
    get_last_10_transactions,
    get_last_10_by_speciality,
    get_last_10_by_service,
    get_last_10_by_diagnosis,
)
from sequence_pairs import (
    SEQUENCE_PAIRS,
    PAIR_WEIGHTS,
    pair_key,
    build_pair_tensor,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN    = 10
N_FEATURES = 1

# ── THRESHOLDS — paste from Colab threshold cell after training ──
# NOTE: these thresholds were calibrated for the OLD 4-grouping,
# 5-feature combined score. They MUST be recalibrated against the
# new 8-pair-type weighted-average score distribution before this
# goes to production -- pull fresh HIGH/MED values from a validation
# run of the FINAL notebook's discriminator on real held-out data.
HIGH_THRESHOLD = -0.6381
MED_THRESHOLD  = -0.3831
# ────────────────────────────────────────────────────────────────

tts_gap = MED_THRESHOLD - HIGH_THRESHOLD
print("[TTSGAN] inference module loaded (8-pair-type design)")
print(f"   HIGH threshold : {HIGH_THRESHOLD:.4f}")
print(f"   MED  threshold : {MED_THRESHOLD:.4f}")
print(f"   SEQ_LEN        : {SEQ_LEN}")
print(f"   N_FEATURES     : {N_FEATURES} (single scalar per timestep)")
print(f"   Pair types     : {len(SEQUENCE_PAIRS)}")
if tts_gap < 0.05:
    print(f"   WARNING: gap ({tts_gap:.4f}) is very small")


# ─────────────────────────────────────────────
# HELPER — Resolve the current transaction's IDs
# (reuses id_mappings, same as the rest of the codebase)
# ─────────────────────────────────────────────
def _resolve_current_ids(transaction: dict) -> dict:
    from id_mappings import names_to_ids
    resolved = names_to_ids(
        doctor_id       = transaction.get('DOCTOR_ID')       or 'Unknown',
        speciality_name = transaction.get('SPECIALITY_NAME') or 'Unknown',
        service_desc    = transaction.get('SERVICE_DESCRIPTION'),
        diagnosis       = transaction.get('DIAGNOSIS'),
        patient_id      = transaction.get('PATIENT_ID')      or 0,
    )
    return {
        'DOCTOR_ID'    : resolved['DOCTOR_ID'],
        'SPECIALITY_ID': resolved['SPECIALITY_ID'] if resolved['SPECIALITY_ID'] is not None else 0,
        'SERVICE_ID'   : transaction.get('SERVICE_ID') or resolved.get('SERVICE_ID'),
        'DIAGNOSIS_ID' : resolved['DIAGNOSIS_ID'],
    }


# ─────────────────────────────────────────────
# HELPER — Resolve a history ORM row's IDs the same way
# ─────────────────────────────────────────────
def _resolve_history_ids(t) -> dict:
    from id_mappings import names_to_ids
    resolved = names_to_ids(
        doctor_id       = t.doctor_id           or 'Unknown',
        speciality_name = t.speciality_name     or 'Unknown',
        service_desc    = t.service_description or 'Unknown',
        diagnosis       = t.diagnosis,
        patient_id      = t.patient_id          or 0,
    )
    return {
        'DOCTOR_ID'    : resolved['DOCTOR_ID'],
        'SPECIALITY_ID': resolved['SPECIALITY_ID'] if resolved['SPECIALITY_ID'] is not None else 0,
        'SERVICE_ID': resolved.get('SERVICE_ID'),
        'DIAGNOSIS_ID' : resolved['DIAGNOSIS_ID'],
    }


# ─────────────────────────────────────────────
# HELPER — Fetch DB history for a given group_col, matching the
# entity referenced by the current transaction.
# Returns list of resolved-ID dicts, oldest-first.
# ─────────────────────────────────────────────
def _fetch_group_history(group_col: str, transaction: dict, current_ids: dict, db: Session) -> list:
    if group_col == 'DOCTOR_ID':
        rows = get_last_10_transactions(db, current_ids['DOCTOR_ID'])
    elif group_col == 'SPECIALITY_ID':
        rows = get_last_10_by_speciality(db, transaction.get('SPECIALITY_NAME', ''))
    elif group_col == 'SERVICE_ID':
        rows = get_last_10_by_service(db, current_ids['SERVICE_ID'])
    elif group_col == 'DIAGNOSIS_ID':
        diag = transaction.get('DIAGNOSIS', '')
        if not diag or str(diag).strip() in ('', 'None', 'Unknown', 'unknown'):
            diag = 'No Diagnosis'
        rows = get_last_10_by_diagnosis(db, diag)
    else:
        raise ValueError(f"Unknown group_col: {group_col}")

    # ORM rows come back most-recent-first (order_by desc(id)); we need
    # oldest-first to match training's chronological CSV row order.
    rows = list(reversed(rows))
    return [_resolve_history_ids(t) for t in rows]


# ═══════════════════════════════════════════════════════
# MAIN INFERENCE FUNCTION
# ═══════════════════════════════════════════════════════
def predict_ttsgan(transaction: dict, db: Session) -> dict:
    """
    Runs TTS-WGAN-GP discriminator on all 8 pair-type sequences,
    exactly matching how TTSWGAN___DCTGAN_FINAL.ipynb built its
    training data (Chunk 6, SEQUENCE_PAIRS).

    Flow:
      1. Resolve current transaction's IDs.
      2. For each of the 8 (group_col, feature_col) pairs:
           - fetch DB history for group_col's entity
           - build a (1, 10, 1) tensor of feature_col values
           - score it with the discriminator
      3. Weighted average of the 8 scores -> combined_score
      4. Apply thresholds -> risk_level, is_fraud

    Returns full result dict including all 8 pair tensors (for
    investigation_node's feature-ablation step) and per-pair scores.
    """
    import ttsgan_loader
    if ttsgan_loader.ttsgan_model is None:
        ttsgan_loader.load_ttsgan()
    discriminator = ttsgan_loader.ttsgan_model['discriminator']
    discriminator.eval()

    current_ids = _resolve_current_ids(transaction)

    print(f"\n   [TTSGAN] Running 8-pair-type inference...")
    print(f"   doctor={current_ids['DOCTOR_ID']} | specialty={current_ids['SPECIALITY_ID']} | "
          f"service={current_ids['SERVICE_ID']} | diagnosis={current_ids['DIAGNOSIS_ID']}")

    pair_scores   = {}
    pair_tensors  = {}
    pad_info      = {}
    all_warnings  = []

    # Cache history fetches per group_col -- several pairs share the
    # same group_col (e.g. SERVICE_ID is group_col for two pairs)
    history_cache = {}

    for group_col, feature_col in SEQUENCE_PAIRS:
        key = pair_key(group_col, feature_col)

        if group_col not in history_cache:
            history_cache[group_col] = _fetch_group_history(
                group_col, transaction, current_ids, db
            )
        history_ids = history_cache[group_col]
        history_feature_values = [h[feature_col] for h in history_ids]

        tensor, warning, is_padded = build_pair_tensor(
            feature_col            = feature_col,
            current_feature_value  = current_ids[feature_col],
            history_feature_values = history_feature_values,
        )

        with torch.no_grad():
            score = discriminator(tensor).item()

        pair_scores[key]  = round(score, 6)
        pair_tensors[key] = tensor
        pad_info[key]     = is_padded
        if warning:
            all_warnings.append(warning)

        print(f"   TTSGAN {key:35s} score={score:.6f}  padded={is_padded}")

    # ══════════════════════════════════════════════════════════
    # COMBINE 8 SCORES — Weighted Average
    # ══════════════════════════════════════════════════════════
    combined_score = sum(
        pair_scores[k] * PAIR_WEIGHTS[k] for k in pair_scores
    )

    if combined_score < HIGH_THRESHOLD:
        risk_level = "HIGH RISK"
    elif combined_score < MED_THRESHOLD:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "NORMAL"

    is_fraud = combined_score < MED_THRESHOLD

    print(f"   TTSGAN combined  score={combined_score:.6f} -> {risk_level}")

    # is_cold_start: True if ANY pair-type needed zero-padding
    is_cold_start = any(pad_info.values())

    return {
        "model"     : "TTS-WGAN-GP",
        "status"    : "FRAUD" if is_fraud else "NORMAL",
        "is_fraud"  : bool(is_fraud),
        "risk_level": risk_level,

        # Combined score (used for weighted voting in agent1)
        "score": round(combined_score, 6),

        # Per-pair-type scores (for detailed report / ablation)
        "pair_scores" : pair_scores,
        "pair_weights": PAIR_WEIGHTS,

        # All 8 tensors -- consumed by investigation_node for ablation.
        # Popped out of the dict by model_node before dctgan_result is
        # stored, same pattern as before (see model_node.py).
        "sequence_tensors": pair_tensors,

        "cold_start_per_pair": pad_info,
        "is_cold_start"      : is_cold_start,

        # Kept for backward compatibility with rag_node template /
        # old report formatting code that may still read this key.
        "relationship_features": {},
        "sequence_used"        : SEQ_LEN,

        "threshold": {
            "high"  : HIGH_THRESHOLD,
            "medium": MED_THRESHOLD,
            "gap"   : round(tts_gap, 4)
        },
        "warnings": list(set(all_warnings))
    }
