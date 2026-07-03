# ============================================================
# agent1/nodes/investigation_node.py
# NODE 2 OF AGENT 1
#
# HOW THIS WORKS — PAIR-TYPE ABLATION (Paper-backed method):
#
#   The GAN discriminator is trained on thousands of NORMAL
#   transaction sequences, built from 8 separate (group_col,
#   feature_col) "relationship" types (e.g. doctor->service,
#   service->doctor, specialty->diagnosis, etc — see
#   sequence_pairs.py). Each sequence is a single scalar per
#   timestep: "what does a normal doctor->service relationship
#   look like?"
#
#   When the combined score flags a transaction as fraud, we ask:
#   "WHICH relationship is making this transaction look wrong?"
#
#   We find out by replacing each pair-type's ENTIRE sequence
#   tensor (not a column — a whole separate tensor) with an
#   all-median "neutral" sequence and re-running the DCT-GAN
#   discriminator on it, recomputing the combined weighted score.
#
#   The pair whose replacement causes the BIGGEST improvement in
#   the combined score = the anomalous relationship = the fraud
#   signal.
#
#   WHY DCT-GAN DISCRIMINATOR ONLY:
#     - DCT-GAN is our more reliable model
#     - Node 1 already decided fraud/normal using both models
#     - Node 2 only needs to EXPLAIN why — use the better model
#
#   SEQUENCE TENSORS — NO LONGER REBUILT HERE:
#     The 8 (1, 10, 1) pair-type tensors are built ONCE in
#     dctgan_inference during model_node, then passed through
#     state['sequence_tensors'] (a dict keyed by pair name).
#     investigation_node reads it directly — zero extra DB
#     queries, zero extra encoding work. These are the same
#     tensors the discriminator already scored during model_node,
#     so the ablation scores will be perfectly consistent with
#     the original detection score.
#
#     Fallback: if state['sequence_tensors'] is None/empty for any
#     reason (e.g. very old code path), the node falls back to
#     zero tensors for all 8 pairs and logs a warning. Results will
#     be unreliable in that case but the system will not crash.
#
#   HOW ABLATION WORKS IN SIMPLE ENGLISH:
#     8 sequence tensors, each shape (1, 10, 1) — one per pair-type
#
#     Run discriminator on all 8, weighted-average -> combined score
#     e.g. combined score = -0.85  (fraud)
#
#     Now for each of the 8 pair-types:
#       Replace that ENTIRE pair's tensor with an all-median
#       neutral tensor (a "what if this relationship looked
#       average" simulation)
#       Recompute the combined weighted score with this swap
#       Improvement = new_combined_score - original_combined_score
#
#     Pair-type with HIGHEST improvement = most anomalous
#     relationship = the thing that was making the transaction
#     look fraudulent
#
#   NO PERSONAL HISTORY NEEDED:
#     Works for brand new doctors (1 transaction, zero-padded)
#     Works for doctors with 1000 transactions
#     The population median comes from training data — saved once
#     per pair-type in feature_medians.json
#
# FRAUD TYPES DETECTED (9 types — 8 pair-types + 1 pattern-level):
#
#   1. DOCTOR_ID -> SERVICE_ID anomaly
#      -> "Doctor Service-Pattern Anomaly"  |  Responsible: Doctor
#
#   2. SERVICE_ID -> DOCTOR_ID anomaly
#      -> "Service Access Anomaly"          |  Responsible: Hospital
#
#   3. SPECIALITY_ID -> SERVICE_ID anomaly
#      -> "Specialty-Service Mismatch Fraud" |  Responsible: Hospital
#
#   4. SERVICE_ID -> SPECIALITY_ID anomaly
#      -> "Service Cross-Specialty Misuse"   |  Responsible: Hospital
#
#   5. SPECIALITY_ID -> DIAGNOSIS_ID anomaly
#      -> "Specialty-Diagnosis Mismatch Fraud" | Responsible: Doctor
#
#   6. DIAGNOSIS_ID -> SPECIALITY_ID anomaly
#      -> "Diagnosis Cross-Specialty Pattern Anomaly" | Responsible: Doctor
#
#   7. SERVICE_ID -> DIAGNOSIS_ID anomaly
#      -> "Phantom Billing / Upcoding"       |  Responsible: Doctor
#
#   8. DIAGNOSIS_ID -> SERVICE_ID anomaly
#      -> "Diagnosis-Service Mismatch Fraud" |  Responsible: Doctor
#
#   9. No single pair dominant
#      -> "Anomalous Billing Sequence (Pattern-Level Fraud)"
#      -> The whole multi-relationship pattern is wrong
#      -> Responsible: Doctor (controls overall billing behaviour)
#
#   NOTE — PATIENT_ID REMOVED:
#     TTSWGAN___DCTGAN_FINAL.ipynb's 8 SEQUENCE_PAIRS do not
#     include PATIENT_ID at all (confirmed directly from the
#     notebook's Chunk 6). The old "Patient-Doctor Collusion /
#     Repeat Billing Fraud" fraud type has no corresponding
#     trained signal in this design and has been removed. If
#     patient-level collusion detection is still wanted, it would
#     need a 9th training pair-type (e.g. DOCTOR_ID -> PATIENT_ID)
#     added back into the notebook and retrained.
#
# INPUT  (reads from state — written by Node 1):
#   state['ttsgan_result']    → score, risk_level
#   state['dctgan_result']    → score, risk_level
#   state['sequence_tensors'] → dict of 8 (1,10,1) tensors
#   state['overall_risk']     → weighted vote result
#   state['risk_votes']       → individual model votes
#   state['transaction']      → current transaction dict
#   state['db']                → DB session (used only for history display)
#
# OUTPUT (writes to state — read by Node 3 and Node 4):
#   responsible_party, primary_fraud_type, fraud_types,
#   confidence, reasons, overbilling_flag, overbilling_note
# ============================================================

import sys
import os
import json
import copy
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sequence_pairs import SEQUENCE_PAIRS, PAIR_KEYS, pair_key


# ── Pair-type metadata ──────────────────────────────────────────
# REPLACES the old 5-single-ID FEATURE_ORDER / FEATURE_INDEX design.
#
# TTSWGAN___DCTGAN_FINAL.ipynb trains on 8 separate (group_col,
# feature_col) sequence types, not 5 stacked ID columns. Ablation
# now works at the PAIR level: each of the 8 sequence_tensors
# (one per pair) is replaced wholesale with its population median
# and re-scored. The pair whose replacement most improves the
# score is the "anomalous pair" — i.e. the relationship that looks
# wrong, not a single isolated column.
#
# feature_medians.json (saved by the FINAL notebook's Chunk 8b) is
# keyed by these same 8 pair names, e.g.:
#   "DOCTOR_ID__to__SERVICE_ID": 0.0088, "SERVICE_ID__to__DOCTOR_ID": 0.0, ...


# ── Fraud type metadata per PAIR (replaces per-column DIMENSION_META) ──
# Each entry defines:
#   fraud_type   → the name used in reports and scoring_node
#   responsible  → who is accountable for this pair anomaly
#   explanation  → human-readable explanation for the report
#
# WHY EACH RESPONSIBILITY IS ASSIGNED THIS WAY:
#
#   DOCTOR_ID -> SERVICE_ID:
#     This doctor's sequence of billed services doesn't match any
#     legitimate doctor's service-billing pattern. The doctor
#     chooses what to bill. -> Doctor responsible.
#
#   SERVICE_ID -> DOCTOR_ID:
#     This service's sequence of billing doctors doesn't match
#     which doctors normally bill it (e.g. a niche specialist
#     service suddenly billed by many unrelated doctors) -- a
#     hospital-side credentialing/access-control issue.
#     -> Hospital responsible.
#
#   SPECIALITY_ID -> SERVICE_ID:
#     The services billed under this specialty don't match what
#     this specialty normally bills. Specialty codes are assigned
#     by hospital admin staff. -> Hospital responsible.
#
#   SERVICE_ID -> SPECIALITY_ID:
#     This service is being billed by an unusual mix of specialties
#     -- suggests the service is being misused/upcoded across
#     specialty lines, a billing-department-level issue.
#     -> Hospital responsible.
#
#   SPECIALITY_ID -> DIAGNOSIS_ID:
#     The diagnoses associated with this specialty don't match
#     normal clinical patterns for that specialty.
#     -> Doctor responsible (diagnosis is a clinical decision).
#
#   DIAGNOSIS_ID -> SPECIALITY_ID:
#     This diagnosis is being handled by an unusual mix of
#     specialties -- may indicate diagnosis shopping across
#     specialties to justify billing. -> Doctor responsible.
#
#   SERVICE_ID -> DIAGNOSIS_ID:
#     This service is being justified by an unusual mix of
#     diagnoses -- classic upcoding/unjustified-service signal.
#     -> Doctor responsible (diagnosis written by doctor).
#
#   DIAGNOSIS_ID -> SERVICE_ID:
#     This diagnosis is being used to justify an unusual mix of
#     services -- diagnosis-service mismatch fraud.
#     -> Doctor responsible.

DIMENSION_META = {
    'DOCTOR_ID__to__SERVICE_ID': {
        'fraud_type' : 'Doctor Service-Pattern Anomaly',
        'responsible': 'Doctor',
        'explanation': (
            'This doctor\'s sequence of recently billed services does not '
            'match any legitimate doctor billing pattern the GAN learned '
            'from the training population. The doctor controls what '
            'services they bill — an anomalous service sequence under '
            'this doctor\'s ID points to the doctor as the responsible party.'
        ),
    },
    'SERVICE_ID__to__DOCTOR_ID': {
        'fraud_type' : 'Service Access Anomaly',
        'responsible': 'Hospital',
        'explanation': (
            'The sequence of doctors recently billing this service code '
            'does not match the normal population of doctors who '
            'legitimately bill it. This suggests the service is being '
            'billed by doctors who should not have access to it — a '
            'hospital-side credentialing or claim-processing control issue.'
        ),
    },
    'SPECIALITY_ID__to__SERVICE_ID': {
        'fraud_type' : 'Specialty-Service Mismatch Fraud',
        'responsible': 'Hospital',
        'explanation': (
            'The services recently billed under this specialty code do '
            'not match the normal service mix the GAN learned for that '
            'specialty across the population. Specialty codes are '
            'assigned by hospital administrative staff when a claim is '
            'processed — an inconsistent specialty-service pairing '
            'points to a hospital-side coding error or manipulation.'
        ),
    },
    'SERVICE_ID__to__SPECIALITY_ID': {
        'fraud_type' : 'Service Cross-Specialty Misuse',
        'responsible': 'Hospital',
        'explanation': (
            'This service is being billed across an unusual mix of '
            'specialties compared to its normal usage pattern. This '
            'suggests the service code is being misapplied or upcoded '
            'across specialty lines — a billing department-level issue.'
        ),
    },
    'SPECIALITY_ID__to__DIAGNOSIS_ID': {
        'fraud_type' : 'Specialty-Diagnosis Mismatch Fraud',
        'responsible': 'Doctor',
        'explanation': (
            'The diagnoses recently associated with this specialty do '
            'not match the normal clinical pattern the GAN learned for '
            'that specialty. Diagnoses are clinical decisions made by '
            'the treating doctor — an inconsistent specialty-diagnosis '
            'pairing points to the doctor as the responsible party.'
        ),
    },
    'DIAGNOSIS_ID__to__SPECIALITY_ID': {
        'fraud_type' : 'Diagnosis Cross-Specialty Pattern Anomaly',
        'responsible': 'Doctor',
        'explanation': (
            'This diagnosis is being handled by an unusual mix of '
            'specialties compared to its normal clinical pattern. This '
            'may indicate diagnosis shopping across specialties to '
            'justify billing for services the diagnosis would not '
            'normally warrant. The doctor selects the diagnosis on the '
            'clinical record. -> Doctor responsible.'
        ),
    },
    'SERVICE_ID__to__DIAGNOSIS_ID': {
        'fraud_type' : 'Phantom Billing / Upcoding',
        'responsible': 'Doctor',
        'explanation': (
            'This service is being justified by an unusual mix of '
            'diagnoses compared to its normal clinical usage. The GAN '
            'learned which diagnoses typically justify this service '
            'across the population — this pattern does not match. The '
            'doctor writes the diagnosis to justify the service billed, '
            'a classic upcoding / unjustified-service signal.'
        ),
    },
    'DIAGNOSIS_ID__to__SERVICE_ID': {
        'fraud_type' : 'Diagnosis-Service Mismatch Fraud',
        'responsible': 'Doctor',
        'explanation': (
            'This diagnosis is being used to justify an unusual mix of '
            'services compared to its normal clinical pattern. The GAN '
            'learned which services are typically billed for this '
            'diagnosis across the population — this combination does '
            'not match. The doctor attaches the diagnosis to justify '
            'the service billed. -> Doctor responsible.'
        ),
    },
}

assert set(DIMENSION_META.keys()) == set(PAIR_KEYS), (
    "DIMENSION_META keys must exactly match the 8 pair-types in sequence_pairs.PAIR_KEYS"
)


# ── Load feature medians (saved from training — one time) ─────
# These are the population median values, ONE PER PAIR TYPE,
# computed across ALL training sequences for that pair (Chunk 8b
# in TTSWGAN___DCTGAN_FINAL.ipynb). Used as the "neutral
# replacement" value during feature ablation.
# Shape: {'DOCTOR_ID__to__SERVICE_ID': 0.0088, 'SERVICE_ID__to__DOCTOR_ID': 0.0, ...}

_feature_medians = None

def _load_feature_medians() -> dict:
    global _feature_medians
    if _feature_medians is not None:
        return _feature_medians

    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR  = os.path.join(BASE_DIR, '..', '..', 'models')
    median_path = os.path.join(MODELS_DIR, 'feature_medians.json')

    if not os.path.exists(median_path):
        # Fallback: use 0.5 for all features (midpoint of [0,1] normalized range)
        # This is less accurate but will not crash the system
        print(
            f"[investigation] ⚠️  feature_medians.json not found at {median_path}\n"
            f"   Using 0.5 fallback for all features.\n"
            f"   Run the median-saving cell in Colab to fix this."
        )
        _feature_medians = {key: 0.5 for key in PAIR_KEYS}
        return _feature_medians

    with open(median_path, 'r') as f:
        _feature_medians = json.load(f)

    print(f"[investigation] ✅ Feature medians loaded: {_feature_medians}")
    return _feature_medians


# ── Core: Pair-Type Ablation ────────────────────────────────────
def _run_feature_ablation(
    sequence_tensors: dict,
    discriminator,
    device: torch.device,
    feature_medians: dict,
    pair_weights: dict
) -> dict:
    """
    Runs ablation across the 8 pair-type sequence tensors using the
    DCT-GAN discriminator to find which RELATIONSHIP (pair-type) is
    most anomalous.

    REPLACES the old per-column ablation on a single (1,10,5) tensor.
    TTSWGAN___DCTGAN_FINAL.ipynb trains the discriminator on 8
    separate (1,10,1) sequences, not 5 stacked columns in one
    sequence — so ablation now works at the pair level: each pair's
    tensor is swapped wholesale for a "neutral" all-median tensor,
    and we measure how much the COMBINED weighted-average score
    improves.

    HOW IT WORKS:
      1. Compute original_score = weighted average of discriminator
         scores on all 8 real pair tensors (same formula as inference).
      2. For each of the 8 pair-types (one at a time):
           - Replace that pair's tensor with an all-median tensor
             (shape (1,10,1), every value = that pair's population
             median from feature_medians.json)
           - Recompute the combined weighted-average score using
             this one swapped-out pair + the other 7 real pairs
           - improvement = new_combined_score - original_score
      3. The pair with the highest improvement = the relationship
         that was most responsible for the fraud signal.

    Args:
        sequence_tensors : dict {pair_key: (1,10,1) tensor} — the 8
                            real pair-type sequences for this transaction
        discriminator    : loaded DCT-GAN discriminator model
        device           : torch device (cpu or cuda)
        feature_medians  : dict {pair_key: median_value} from
                            feature_medians.json
        pair_weights     : dict {pair_key: weight}, sums to 1.0 —
                            same weights used in dctgan_inference

    Returns:
        dict: {
            'original_score'  : float,
            'ablation_scores' : {pair_key: combined_score_after_ablation},
            'improvements'    : {pair_key: improvement_value},
            'ranked_features' : [(pair_key, improvement), ...] sorted desc
        }
    """
    discriminator.eval()

    # Step 1: score all 8 real pairs once, compute combined baseline
    real_scores = {}
    with torch.no_grad():
        for key, tensor in sequence_tensors.items():
            real_scores[key] = discriminator(tensor).item()

    original_score = sum(
        real_scores[k] * pair_weights[k] for k in real_scores
    )

    ablation_scores = {}
    improvements    = {}

    # Step 2: for each pair-type, swap in an all-median tensor and
    # recompute the combined score using the other 7 real scores
    for key in PAIR_KEYS:
        median = feature_medians.get(key, 0.5)

        neutral_tensor = torch.full(
            (1, sequence_tensors[key].shape[1], 1),
            float(median),
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            neutral_score = discriminator(neutral_tensor).item()

        swapped_scores = dict(real_scores)
        swapped_scores[key] = neutral_score

        combined_after = sum(
            swapped_scores[k] * pair_weights[k] for k in swapped_scores
        )

        ablation_scores[key] = round(combined_after, 6)

        # Higher improvement = replacing this pair with a neutral
        # value made the combined score LESS fraudulent (higher) =
        # this pair was the anomalous relationship driving the signal
        improvement = combined_after - original_score
        improvements[key] = round(improvement, 6)

    # Step 3: rank pairs by improvement (highest first = most anomalous)
    ranked = sorted(
        improvements.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        'original_score' : round(original_score, 6),
        'ablation_scores': ablation_scores,
        'improvements'   : improvements,
        'ranked_features': ranked,
    }


# ── _get_sequence_tensor REMOVED ─────────────────────────────
# The sequence tensors are no longer rebuilt here.
# They are built ONCE in dctgan_inference (predict_dctgan),
# returned in the result dict, extracted by model_node,
# and stored in state['sequence_tensors'] (dict of 8, plural).
# investigation_node reads state['sequence_tensors'] directly.
# This eliminates the duplicate DB queries and tensor rebuild.


# ── Main Node ─────────────────────────────────────────────────
def investigation_node(state: dict) -> dict:
    """
    LangGraph Node 2 — Investigation Module

    Uses pair-type ablation on the DCT-GAN discriminator to
    identify which (group_col -> feature_col) relationship is
    causing the fraud signal, then maps that to a fraud type and
    responsible party.

    Reads state['sequence_tensors'] — dict of 8 (1, 10, 1) tensors
    built by dctgan_inference during model_node. No DB query, no
    tensor rebuild.

    Paper reference: Inspired by AnoGAN (Schlegl et al. 2017) and
    SHAP-based GAN explanation methods — replacing features with
    neutral values to measure each feature's contribution to anomaly score.
    """

    print("\n" + "=" * 55)
    print("NODE 2 — INVESTIGATION MODULE (Pair-Type Ablation)")
    print("=" * 55)

    # ── Read from state ───────────────────────────────────────
    transaction   = state['transaction']
    ttsgan_result = state.get('ttsgan_result', {})
    dctgan_result = state.get('dctgan_result', {})
    overall_risk  = state.get('overall_risk', 'MEDIUM RISK')
    risk_votes    = state.get('risk_votes', {})
    db            = state.get('db')

    doctor_id  = transaction.get('DOCTOR_ID', 'Unknown')
    speciality = transaction.get('SPECIALITY_NAME',    transaction.get('SPECIALITY_ID', 'Unknown'))
    service    = transaction.get('SERVICE_DESCRIPTION',transaction.get('SERVICE_ID',    'Unknown'))
    diagnosis  = transaction.get('DIAGNOSIS',          transaction.get('DIAGNOSIS_ID',  'Unknown'))

    tts_score = ttsgan_result.get('score', 0.0)
    dct_score = dctgan_result.get('score', 0.0)
    tts_risk  = ttsgan_result.get('risk_level', 'NORMAL')
    dct_risk  = dctgan_result.get('risk_level', 'NORMAL')

    print(f"Doctor     : {doctor_id}")
    print(f"Specialty  : {speciality}")
    print(f"Service    : {service}")
    print(f"Diagnosis  : {diagnosis}")
    print(f"TTS Score  : {tts_score:.6f} → {tts_risk}")
    print(f"DCT Score  : {dct_score:.6f} → {dct_risk}")
    print(f"Overall    : {overall_risk}")

    # ── Load DCT-GAN discriminator ────────────────────────────
    print("\nLoading DCT-GAN discriminator for ablation...")
    try:
        import dctgan_loader
        if dctgan_loader.dctgan_model is None:
            dctgan_loader.load_dctgan()
        discriminator = dctgan_loader.dctgan_model['discriminator']
        device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("  ✅ DCT-GAN discriminator ready")
    except Exception as e:
        print(f"  ⚠️  Could not load DCT-GAN discriminator: {e}")
        print("  Falling back to GAN-score-only investigation")
        discriminator = None
        device        = torch.device('cpu')

    # ── Load feature medians ──────────────────────────────────
    print("Loading feature medians...")
    feature_medians = _load_feature_medians()

    # ── Run ablation if discriminator is available ────────────
    ablation_result  = None
    ranked_features  = []

    if discriminator is not None:
        print("\nStep 1: Reading sequence_tensors (8 pair-types) from state...")
        seq_tensors = state.get('sequence_tensors', None)

        if seq_tensors:
            print(f"  ✅ {len(seq_tensors)} pair tensors received from model_node")
        else:
            # Fallback: should not happen in normal flow, but
            # protects against edge cases (e.g. testing a node in isolation)
            print(f"  ⚠️  sequence_tensors not in state — using zero-tensor fallback for all 8 pairs")
            print(f"     Ablation results will be unreliable for this transaction")
            seq_tensors = {
                key: torch.zeros(1, 10, 1).to(device) for key in PAIR_KEYS
            }

        print("\nStep 2: Running pair-type ablation...")
        print(f"  Original DCT score: {dct_score:.6f}")
        print(f"  (Lower = more fraudulent, Higher = more normal)")
        print(f"  Replacing each pair-type's sequence with its population median and re-scoring:\n")

        from sequence_pairs import PAIR_WEIGHTS

        ablation_result = _run_feature_ablation(
            seq_tensors, discriminator, device, feature_medians, PAIR_WEIGHTS
        )

        ranked_features = ablation_result['ranked_features']

        print(f"  {'Pair':<32} {'Median':>8}  {'Score After':>12}  {'Improvement':>12}")
        print(f"  {'-'*32} {'-'*8}  {'-'*12}  {'-'*12}")
        for feat, improvement in ranked_features:
            median      = feature_medians.get(feat, 0.5)
            score_after = ablation_result['ablation_scores'][feat]
            marker      = " ← ANOMALOUS" if feat == ranked_features[0][0] else ""
            print(f"  {feat:<32} {median:>8.4f}  {score_after:>12.6f}  {improvement:>+12.6f}{marker}")

    # ── Determine anomalous dimension ─────────────────────────
    # We use a MINIMUM IMPROVEMENT THRESHOLD to avoid false attribution:
    # If the best improvement is tiny (< 0.05), no single feature is
    # clearly responsible — it's a pattern-level anomaly.
    # 0.05 means: replacing this feature improved the score by at least
    # 5% of the discriminator's score range — a meaningful signal.

    MIN_IMPROVEMENT_THRESHOLD = 0.05

    detected_fraud_types = []
    responsible_parties  = []
    reasons              = []
    anomalous_features   = []

    if ranked_features and ranked_features[0][1] >= MIN_IMPROVEMENT_THRESHOLD:

        # Find ALL features above the threshold (can be more than one)
        for feat, improvement in ranked_features:
            if improvement >= MIN_IMPROVEMENT_THRESHOLD:
                anomalous_features.append((feat, improvement))

        print(f"\nStep 3: Anomalous features detected (improvement >= {MIN_IMPROVEMENT_THRESHOLD}):")
        for feat, imp in anomalous_features:
            print(f"  → {feat}: improvement = {imp:+.6f}")

        for feat, improvement in anomalous_features:
            meta = DIMENSION_META[feat]
            detected_fraud_types.append(meta['fraud_type'])
            responsible_parties.append(meta['responsible'])
            reasons.append(
                f"[{feat} | ablation improvement = {improvement:+.4f}] "
                f"{meta['explanation']}"
            )

    else:
        # No feature clearly dominates — GAN detected a pattern anomaly
        # The whole sequence is wrong, not any individual dimension
        print(f"\nStep 3: No single feature dominates (all improvements < {MIN_IMPROVEMENT_THRESHOLD})")
        print(f"  → Pattern-level anomaly detected by GAN")

        if tts_risk == "HIGH RISK" and dct_risk == "HIGH RISK":
            fraud_label = "Consistent Anomalous Billing Sequence"
            reason_text = (
                f"[Both models HIGH RISK | TTS={tts_score:.4f}, DCT={dct_score:.4f}] "
                f"Both GAN models independently flagged this transaction's billing "
                f"sequence as highly anomalous. The complete sequence of doctor, "
                f"specialty, service, and diagnosis across the 10 recent transactions "
                f"does not resemble any legitimate billing pattern learned during "
                f"training. No single feature dominates — the combination as a whole "
                f"is the anomaly signal. This indicates a systematic billing fraud "
                f"pattern rather than a single incorrect field."
            )
        else:
            fraud_label = "Anomalous Billing Pattern"
            reason_text = (
                f"[TTS={tts_score:.4f} {tts_risk} | DCT={dct_score:.4f} {dct_risk}] "
                f"The GAN models detected an anomalous billing sequence pattern. "
                f"The doctor's recent billing history does not match any legitimate "
                f"billing pattern seen during training, but no single relationship "
                f"pair-type is clearly more anomalous than the others. This may "
                f"indicate a complex multi-relationship fraud pattern."
            )

        detected_fraud_types.append(fraud_label)
        responsible_parties.append("Doctor")
        reasons.append(reason_text)
        anomalous_features = []

    # ── Determine responsible party ───────────────────────────
    # If multiple pair-types are flagged, report all unique parties.
    # NOTE: with PATIENT_ID removed from the 8 trained pair-types
    # (see DIMENSION_META), only 'Doctor' and 'Hospital' can ever
    # appear here now — the old 'Patient' branch is unreachable
    # and has been removed for clarity. If a 9th pair-type
    # involving PATIENT_ID is added back later, restore that branch.
    unique_parties = list(dict.fromkeys(responsible_parties))

    if len(unique_parties) == 1:
        responsible = unique_parties[0]
    elif set(unique_parties) == {'Doctor', 'Hospital'}:
        responsible = "Multiple: Doctor + Hospital"
    else:
        responsible = " + ".join(unique_parties)

    primary_fraud_type = detected_fraud_types[0]

    # ── Confidence score ──────────────────────────────────────
    # Components:
    #   1. Ablation clarity (40%): how clearly one feature dominates
    #      = improvement of top feature / sum of all improvements
    #      High clarity = one feature is clearly the cause
    #
    #   2. GAN agreement (35%): do both models agree on risk level?
    #      Both HIGH = 1.0, Both MEDIUM = 0.85, Disagreement = 0.60
    #
    #   3. Risk level weight (25%): how severe is the overall verdict?
    #      HIGH RISK = 1.0, MEDIUM RISK = 0.75

    if ablation_result and ranked_features:
        all_improvements = [max(0, imp) for _, imp in ranked_features]
        total_imp        = sum(all_improvements)

        if total_imp > 0:
            top_improvement  = max(0, ranked_features[0][1])
            ablation_clarity = round(top_improvement / total_imp, 4)
        else:
            ablation_clarity = 0.5
    else:
        ablation_clarity = 0.5

    if tts_risk == "HIGH RISK" and dct_risk == "HIGH RISK":
        gan_agreement = 1.0
    elif tts_risk == dct_risk:
        gan_agreement = 0.85
    else:
        gan_agreement = 0.60

    risk_weight = 1.0 if overall_risk == "HIGH RISK" else 0.75

    confidence = round(
        min(
            ablation_clarity * 0.40
            + gan_agreement  * 0.35
            + risk_weight    * 0.25,
            1.0
        ),
        4
    )

    print(f"\nConfidence calculation:")
    print(f"  Ablation clarity ({ablation_clarity:.4f}) × 0.40 = {ablation_clarity * 0.40:.4f}")
    print(f"  GAN agreement   ({gan_agreement:.4f}) × 0.35 = {gan_agreement * 0.35:.4f}")
    print(f"  Risk weight     ({risk_weight:.4f}) × 0.25 = {risk_weight * 0.25:.4f}")
    print(f"  Total confidence: {confidence:.4f}")

    # ── Overbilling flag ──────────────────────────────────────
    # Set when both GAN models independently say HIGH RISK.
    # Two different architectures (Transformer-only vs CNN+Transformer)
    # agreeing on HIGH RISK = very strong signal of persistent fraud.
    overbilling_flag = (tts_risk == "HIGH RISK" and dct_risk == "HIGH RISK")
    overbilling_note = ""

    if overbilling_flag:
        overbilling_note = (
            f"Overbilling suspected: TTS-WGAN-GP (score={tts_score:.4f}) and "
            f"DCT-GAN (score={dct_score:.4f}) both independently flagged HIGH RISK. "
            f"Two architecturally different models agreeing indicates a persistent "
            f"anomalous billing pattern across this doctor's recent transaction "
            f"sequence — not a one-off error. A full manual billing audit is "
            f"recommended for this doctor covering at least the last 6 months."
        )
        print(f"\n⚠️  OVERBILLING FLAG RAISED — both models HIGH RISK")

    # ── Print final result ────────────────────────────────────
    print(f"\nINVESTIGATION RESULT:")
    print(f"  Responsible     : {responsible}")
    print(f"  Primary Fraud   : {primary_fraud_type}")
    print(f"  All Types       : {detected_fraud_types}")
    print(f"  Confidence      : {confidence}")
    print(f"  Overbilling     : {overbilling_flag}")
    if ablation_result:
        print(f"  Top anomalous   : {ranked_features[0][0] if ranked_features else 'N/A'}")
        print(f"  Improvement     : {ranked_features[0][1]:+.6f}" if ranked_features else "")

    # ── Write to state ────────────────────────────────────────
    return {
        'responsible_party' : responsible,
        'primary_fraud_type': primary_fraud_type,
        'fraud_types'       : detected_fraud_types,
        'confidence'        : confidence,
        'reasons'           : reasons,
        'overbilling_flag'  : overbilling_flag,
        'overbilling_note'  : overbilling_note,
    }