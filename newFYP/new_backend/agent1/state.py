# ============================================================
# agent1/state.py
# THE SHARED WHITEBOARD FOR AGENT 1
#
# HOW IT WORKS:
#   Node 1 (model_node)    → runs TTS-WGAN + DCT-GAN,
#                            writes: is_fraud, ttsgan_result,
#                            dctgan_result, overall_risk,
#                            relationship_features, risk_votes,
#                            sequence_tensors   ← UPDATED (plural)
#   Node 2 (investigation) → reads above, writes:
#                            responsible_party, fraud_types,
#                            primary_fraud_type, reasons,
#                            confidence, overbilling_flag
#   Node 3 (scoring)       → reads above, writes:
#                            doctor_score_penalty,
#                            updated_doctor_score, score_breakdown
#   Node 4 (rag)           → reads ALL above, writes: report
#
# RULE:
#   Every field starts as None.
#   Each node fills in its OWN fields only.
#   No node overwrites another node's fields.
# ============================================================

from typing import TypedDict, Optional, List, Any


class Agent1State(TypedDict):

    # ── INPUT ─────────────────────────────────────────────────
    # Set ONCE at the start in run_agent1(). Never changed.
    transaction: dict
    # Example:
    # {
    #   "PATIENT_ID": 1,
    #   "DOCTOR_ID": "doc_64",
    #   "SPECIALITY_NAME": "Emergency",
    #   "DIAGNOSIS": "FEVER",
    #   "SERVICE_DESCRIPTION": "HBA1C (HS16)",
    #   "SERVICE_ID": "S_01"
    # }

    db: object
    # SQLAlchemy session — passed through state so nodes
    # can query DB without needing FastAPI dependency injection

    # ── NODE 1 OUTPUT: Model Detection Results ────────────────
    # Written by model_node.py
    # Read by investigation_node.py and all nodes after

    is_fraud: bool
    # True  = fraud detected → investigation node will run
    # False = normal         → skip investigation, go to rag

    overall_risk: str
    # Final verdict after weighted vote
    # "HIGH RISK" / "MEDIUM RISK" / "NORMAL"

    ttsgan_result: dict
    # Full result dict from predict_ttsgan()
    # Keys: model, status, is_fraud, risk_level, score,
    #        sequence_used, is_cold_start,
    #        relationship_features, threshold, warnings
    # Example score: 0.0533 (raw WGAN-GP logit)

    dctgan_result: dict
    # Full result dict from predict_dctgan()
    # Same keys as ttsgan_result above

    sequence_tensors: Any
    # Dict of 8 (1, 10, 1) float32 torch.Tensors, one per pair-type,
    # built by dctgan_inference during model_node. Keyed by pair name,
    # e.g. "DOCTOR_ID__to__SERVICE_ID", "SERVICE_ID__to__DOCTOR_ID", etc.
    # Matches feature_medians.json's 8 keys saved by
    # TTSWGAN___DCTGAN_FINAL.ipynb Chunk 8b.
    #
    # Passed directly to investigation_node so it can run pair-type
    # ablation WITHOUT re-querying the DB.
    #
    # Type is Any because TypedDict doesn't support a dict-of-Tensors
    # annotation cleanly — the actual value is always
    # dict[str, torch.Tensor] or None.
    #
    # Written by: model_node  (via dctgan_inference return value)
    # Read by   : investigation_node  (ablation input)
    # Never modified after model_node writes it.

    relationship_features: dict
    # Always {} now — population_stats.pkl removed from new training.
    # Kept in state for backward compatibility with rag_node template.

    risk_votes: dict
    # Weighted vote breakdown from model_node
    # Keys: tts_vote, dct_vote, tts_weight, dct_weight,
    #       weighted_score, high_threshold, med_threshold

    # ── NODE 2 OUTPUT: Investigation Results ──────────────────
    # Written by investigation_node.py
    # Read by scoring_node.py and rag_node.py

    responsible_party: str
    # "Doctor" / "Hospital" / "Multiple: Doctor + Hospital"

    primary_fraud_type: str
    # The single most likely fraud type based on ablation signal
    # Example: "Diagnosis-Service Mismatch Fraud"

    fraud_types: list
    # All fraud types detected (can be multiple)

    confidence: float
    # 0.0 to 1.0 — ablation clarity(40%) + GAN agreement(35%)
    #              + risk weight(25%)

    reasons: list
    # Human-readable list explaining each fraud signal
    # Each item maps one ablated feature to a fraud explanation

    overbilling_flag: bool
    # True if both GANs flag HIGH RISK simultaneously
    # This means the pattern is very consistent → overbilling likely

    overbilling_note: str
    # Explanation of overbilling signal for RAG report

    # ── NODE 3 OUTPUT: Doctor Scoring Results ─────────────────
    # Written by scoring_node.py
    # Read by rag_node.py

    doctor_score_penalty: float
    # Points deducted from doctor score

    updated_doctor_score: float
    # Doctor's new score after penalty (floor = 0)
    # Used by Agent 2 for doctor recommendations

    display_score: float          # ← ADD THIS
# DTS × (review_stars / 5)
# Used by Agent 2 for ranking in recommendations
# Integrates fraud score + patient review star

    score_breakdown: dict
    # How the penalty was calculated — for transparency in report

    # ── NODE 4 OUTPUT: RAG Report ─────────────────────────────
    # Written by rag_node.py — final output of Agent 1

    report: str
    # Full LLM-generated fraud investigation report