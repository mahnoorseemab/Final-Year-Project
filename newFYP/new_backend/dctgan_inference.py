# ============================================================
# DCTGAN_INFERENCE.PY — UPDATED FOR TTSWGAN___DCTGAN_FINAL.ipynb
# (8-pair-type, single-scalar-feature design)
#
# WHY THIS VERSION:
#   Same reasoning as ttsgan_inference.py -- the FINAL notebook
#   trains the discriminator to score ONE (1, 10, 1) pair-type
#   sequence at a time. This module builds all 8 pair-type tensors
#   (shared logic in sequence_pairs.py), scores each, and combines
#   with a weighted average.
#
# sequence_tensors IN RETURN DICT:
#   This is the PRIMARY model_node uses to populate
#   state['sequence_tensors'] (plural now -- one tensor per pair
#   type) for investigation_node's feature ablation. DCT-GAN is
#   used for ablation (matches the original design rationale: DCT
#   is the more reliable model and is used by investigation_node).
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
# NOTE: same caveat as ttsgan_inference.py -- these were calibrated
# for the OLD combined-score distribution. Recalibrate against the
# new 8-pair-type weighted-average score before production use.
HIGH_THRESHOLD = 0.1959
MED_THRESHOLD  = 0.2986

dct_gap = MED_THRESHOLD - HIGH_THRESHOLD
print("[DCTGAN] inference module loaded (8-pair-type design)")
print(f"   HIGH threshold : {HIGH_THRESHOLD:.4f}")
print(f"   MED  threshold : {MED_THRESHOLD:.4f}")
print(f"   SEQ_LEN        : {SEQ_LEN}")
print(f"   N_FEATURES     : {N_FEATURES} (single scalar per timestep)")
print(f"   Pair types     : {len(SEQUENCE_PAIRS)}")
if dct_gap < 0.05:
    print(f"   WARNING: gap ({dct_gap:.4f}) is very small")


# ─────────────────────────────────────────────
# HELPER — Resolve the current transaction's IDs
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
# HELPER — Fetch DB history for a given group_col
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

    rows = list(reversed(rows))  # oldest-first, matches training order
    return [_resolve_history_ids(t) for t in rows]


# ═══════════════════════════════════════════════════════
# MAIN INFERENCE FUNCTION
# ═══════════════════════════════════════════════════════
def predict_dctgan(transaction: dict, db: Session) -> dict:
    """
    Runs DCT-GAN discriminator on all 8 pair-type sequences,
    exactly matching TTSWGAN___DCTGAN_FINAL.ipynb's training data
    construction (Chunk 6, SEQUENCE_PAIRS).

    sequence_tensors in the return dict holds all 8 pair tensors --
    used by investigation_node for feature ablation across pair
    types (replaces the old single-tensor "doctor grouping" design).
    """
    import dctgan_loader
    if dctgan_loader.dctgan_model is None:
        dctgan_loader.load_dctgan()
    discriminator = dctgan_loader.dctgan_model['discriminator']
    discriminator.eval()

    current_ids = _resolve_current_ids(transaction)

    print(f"\n   [DCTGAN] Running 8-pair-type inference...")
    print(f"   doctor={current_ids['DOCTOR_ID']} | specialty={current_ids['SPECIALITY_ID']} | "
          f"service={current_ids['SERVICE_ID']} | diagnosis={current_ids['DIAGNOSIS_ID']}")

    pair_scores   = {}
    pair_tensors  = {}
    pad_info      = {}
    all_warnings  = []

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

        print(f"   DCTGAN {key:35s} score={score:.6f}  padded={is_padded}")

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

    print(f"   DCTGAN combined  score={combined_score:.6f} -> {risk_level}")

    is_cold_start = any(pad_info.values())

    return {
        "model"     : "DCT-GAN",
        "status"    : "FRAUD" if is_fraud else "NORMAL",
        "is_fraud"  : bool(is_fraud),
        "risk_level": risk_level,

        "score": round(combined_score, 6),

        "pair_scores" : pair_scores,
        "pair_weights": PAIR_WEIGHTS,

        # All 8 tensors -- popped out by model_node and stored in
        # state['sequence_tensors'] (plural) for investigation_node.
        "sequence_tensors": pair_tensors,

        "cold_start_per_pair": pad_info,
        "is_cold_start"      : is_cold_start,

        "relationship_features": {},
        "sequence_used"        : SEQ_LEN,

        "threshold": {
            "high"  : HIGH_THRESHOLD,
            "medium": MED_THRESHOLD,
            "gap"   : round(dct_gap, 4)
        },
        "warnings": list(set(all_warnings))
    }
