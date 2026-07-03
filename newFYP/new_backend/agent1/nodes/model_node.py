# ============================================================
# agent1/nodes/model_node.py
# NODE 1 OF AGENT 1
#
# CHANGE 1: relationship_features is no longer read from GAN
#   results. Both inference files return relationship_features
#   = {} because population_stats.pkl was removed from new
#   training. investigation_node uses feature ablation instead.
#
# CHANGE 2: Equal majority voting replaced with WEIGHTED voting.
#   DCT-GAN  → 60% weight (more reliable; used by investigation_node)
#   TTS-WGAN → 40% weight (supporting signal)
#
# CHANGE 3: sequence_tensors (PLURAL — dict of 8 pair-type tensors)
#   is now extracted from dctgan_result and written to state.
#   investigation_node reads it directly from state — no second
#   DB query, no tensor rebuild.
#
#   HOW IT FLOWS:
#     dctgan_inference builds 8 tensors → returns them in result dict
#     model_node pulls them out          → writes to state
#     investigation_node reads state     → runs ablation on each
#
#   The tensors dict is removed from dctgan_result before it is
#   stored in state (to keep dctgan_result a clean JSON-serialisable
#   dict). sequence_tensors lives in its own state field.
#
# HOW WEIGHTED VOTING WORKS:
#   Each risk level is mapped to a numeric value:
#     NORMAL      = 0
#     MEDIUM RISK = 1
#     HIGH RISK   = 2
#
#   weighted_score = (tts_numeric × 0.40) + (dct_numeric × 0.60)
#
#   Thresholds:
#     >= 1.40  → HIGH RISK
#     >= 0.40  → MEDIUM RISK
#     <  0.40  → NORMAL
# ============================================================

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ttsgan_inference import predict_ttsgan
from dctgan_inference import predict_dctgan


# ── Voting weights ────────────────────────────────────────────
TTS_WEIGHT = 0.40   # TTS-WGAN-GP  — supporting signal
DCT_WEIGHT = 0.60   # DCT-GAN      — primary signal (also used by investigation_node)

# ── Risk level → numeric score ────────────────────────────────
RISK_TO_SCORE = {
    'NORMAL'     : 0,
    'MEDIUM RISK': 1,
    'HIGH RISK'  : 2,
}

# ── Weighted score → risk level thresholds ────────────────────
HIGH_THRESHOLD   = 1.40
MEDIUM_THRESHOLD = 0.40


def _weighted_risk(tts_risk: str, dct_risk: str) -> tuple:
    """
    Converts individual model risk levels to a single weighted verdict.

    Returns:
        overall_risk  : str   — "HIGH RISK" / "MEDIUM RISK" / "NORMAL"
        weighted_score: float — the raw combined score (for logging)
    """
    tts_score = RISK_TO_SCORE.get(tts_risk, 0)
    dct_score = RISK_TO_SCORE.get(dct_risk, 0)

    weighted = round((tts_score * TTS_WEIGHT) + (dct_score * DCT_WEIGHT), 4)

    if weighted >= HIGH_THRESHOLD:
        overall_risk = "HIGH RISK"
    elif weighted >= MEDIUM_THRESHOLD:
        overall_risk = "MEDIUM RISK"
    else:
        overall_risk = "NORMAL"

    return overall_risk, weighted


def model_node(state: dict) -> dict:
    """
    LangGraph Node 1 — GAN Fraud Detection.

    Runs TTS-WGAN-GP + DCT-GAN on the doctor's last 10 transactions.
    Applies weighted voting (DCT=60%, TTS=40%) to decide overall risk.

    Also extracts the sequence_tensor from dctgan_result and stores
    it separately in state so investigation_node can use it directly
    for feature ablation — no second DB query needed.
    """

    print("\n" + "=" * 55)
    print("NODE 1 — GAN MODEL DETECTION")
    print("=" * 55)

    transaction = state['transaction']
    db          = state['db']

    doctor_id = transaction.get('DOCTOR_ID', 'Unknown')
    print(f"Doctor ID  : {doctor_id}")
    print(f"Specialty  : {transaction.get('SPECIALITY_NAME', 'Unknown')}")
    print(f"Service    : {transaction.get('SERVICE_DESCRIPTION', 'Unknown')}")
    print(f"Diagnosis  : {transaction.get('DIAGNOSIS', 'Unknown')}")

    # ── Run TTS-WGAN-GP ───────────────────────────────────────
    print("\nRunning TTS-WGAN-GP...")
    ttsgan_result = predict_ttsgan(transaction, db)
    print(f"  TTS Score : {ttsgan_result['score']:.6f} → {ttsgan_result['risk_level']}")
    if ttsgan_result.get('is_cold_start'):
        print(f"  ⚠️  Cold start — doctor has fewer than 10 records in DB")

    # ── Run DCT-GAN ───────────────────────────────────────────
    print("Running DCT-GAN...")
    dctgan_result = predict_dctgan(transaction, db)
    print(f"  DCT Score : {dctgan_result['score']:.6f} → {dctgan_result['risk_level']}")

    # ── Extract sequence_tensors BEFORE storing results ───────
    # predict_dctgan() returns a dict of 8 pair-type tensors so
    # investigation_node can reuse them directly. We pop them out
    # here and store them in their own state field. This keeps
    # dctgan_result a clean dict (no tensors inside it) and
    # sequence_tensors lives in state['sequence_tensors'].
    #
    # We use the DCT-GAN tensors for ablation (DCT is the more
    # reliable model — see investigation_node.py). The TTS-GAN
    # result also carries its own copy of sequence_tensors; we
    # discard that one since it's not used downstream, to avoid
    # storing duplicate tensors in state.
    ttsgan_result.pop('sequence_tensors', None)
    sequence_tensors = dctgan_result.pop('sequence_tensors', None)

    if sequence_tensors:
        print(f"  Sequence tensors extracted: {len(sequence_tensors)} pair-types")
    else:
        print(f"  ⚠️  sequence_tensors missing from dctgan_result — investigation will be unreliable")

    # ── Weighted voting ───────────────────────────────────────
    tts_risk = ttsgan_result['risk_level']
    dct_risk = dctgan_result['risk_level']

    overall_risk, weighted_score = _weighted_risk(tts_risk, dct_risk)

    is_fraud = overall_risk != "NORMAL"

    risk_votes = {
        "tts_vote"      : tts_risk,
        "dct_vote"      : dct_risk,
        "tts_weight"    : TTS_WEIGHT,
        "dct_weight"    : DCT_WEIGHT,
        "weighted_score": weighted_score,
        "high_threshold": HIGH_THRESHOLD,
        "med_threshold" : MEDIUM_THRESHOLD,
    }

    print(f"\nWEIGHTED VOTING RESULT:")
    print(f"  TTS vote      : {tts_risk}  (weight={TTS_WEIGHT})")
    print(f"  DCT vote      : {dct_risk}  (weight={DCT_WEIGHT})")
    print(f"  Weighted score: {weighted_score:.4f}  "
          f"(HIGH>={HIGH_THRESHOLD}, MEDIUM>={MEDIUM_THRESHOLD})")
    print(f"  Overall risk  : {overall_risk}")
    print(f"  Is Fraud      : {is_fraud}")

    # ── relationship_features: always empty now ───────────────
    relationship_features = {}

    return {
        'is_fraud'             : is_fraud,
        'overall_risk'         : overall_risk,
        'ttsgan_result'        : ttsgan_result,
        'dctgan_result'        : dctgan_result,        # tensors already removed
        'sequence_tensors'     : sequence_tensors,     # dict of 8, stored separately in state
        'relationship_features': relationship_features,
        'risk_votes'           : risk_votes,
    }


def should_investigate(state: dict) -> str:
    """
    Conditional edge router after model_node.

    Fraud   → investigation → scoring → rag
    Normal  → rag directly (clearance report, no scoring)
    """
    is_fraud     = state.get('is_fraud', False)
    overall_risk = state.get('overall_risk', 'NORMAL')

    if is_fraud:
        print(f"\n⚠️  FRAUD DETECTED ({overall_risk})")
        print("→ Routing: model → investigation → scoring → rag")
        return "investigation"
    else:
        print(f"\n✅ NORMAL transaction ({overall_risk})")
        print("→ Routing: model → rag (clearance report)")
        return "rag"