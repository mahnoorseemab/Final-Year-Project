# ============================================================
# agent1/nodes/scoring_node.py
# NODE 3 OF AGENT 1  —  DOCTOR TRUST SCORE (DTS)
#
# ────────────────────────────────────────────────────────────
# ACADEMIC FOUNDATION
# ────────────────────────────────────────────────────────────
# This node implements a multi-factor, responsibility-gated
# trust scoring system for healthcare fraud detection.
#
# The core design principle is directly aligned with the
# U.S. Office of Inspector General (OIG) compliance framework
# and the False Claims Act attribution model:
#
#   "Penalties and accountability are assigned ONLY to the
#    party whose action (or inaction) caused the fraudulent
#    claim to be submitted."
#
# Investigation Node (Node 2) uses Feature Ablation
# (inspired by AnoGAN — Schlegl et al. 2017, and SHAP-based
# GAN explanation methods) to identify which of the 5 feature
# dimensions caused the GAN to flag fraud, and then assigns
# a responsible party per dimension:
#
#   DOCTOR_ID    anomaly → Responsible: Doctor
#   SPECIALITY_ID anomaly → Responsible: Hospital
#   SERVICE_ID   anomaly → Responsible: Hospital
#   DIAGNOSIS_ID anomaly → Responsible: Doctor
#   PATIENT_ID   anomaly → Responsible: Doctor + Patient
#   No dominant feature  → Responsible: Doctor (pattern-level)
#
# This node reads that responsible_party from state and
# applies a RESPONSIBILITY GATE:
#   → If Doctor is responsible (in any form) → penalise DTS
#   → If Hospital alone is responsible       → NO penalty at all
#
# ────────────────────────────────────────────────────────────
# DOCTOR TRUST SCORE (DTS)
# ────────────────────────────────────────────────────────────
# Every doctor starts at DTS = 100.
# The score is CUMULATIVE — stored in doctor_scores table,
# carries across all fraud events. Never resets. Floor = 0.
#
# DISPLAY SCORE (used in Agent 2 recommendation UI):
#   display_score = DTS × (review_stars / 5)
#
#   This integrates star-rating reviews (from Reviews_new.csv,
#   used by Agent 2) with the fraud trust score. Both dimensions
#   matter for patient recommendations:
#     - A fraud-free doctor with 2 stars → lower display score
#     - A flagged doctor with 5 stars    → still low display score
#   The DTS remains the authoritative fraud score.
#   The display score is only for Agent 2 UI ranking.
#
# ────────────────────────────────────────────────────────────
# PENALTY FORMULA
# ────────────────────────────────────────────────────────────
# penalty = base_penalty
#           × risk_multiplier
#           × confidence
#           × repeat_multiplier
#           + overbilling_extra
#
# Clamped: min 1 pt, max 50 pt per transaction.
#
#   base_penalty     — fraud type severity (see table below)
#   risk_multiplier  — HIGH RISK=1.0, MEDIUM RISK=0.80
#   confidence       — from investigation_node (0.0–1.0):
#                      ablation_clarity(40%) + gan_agreement(35%)
#                      + risk_weight(25%). Low confidence = lighter
#                      penalty; the system is less certain.
#   repeat_multiplier— 1.0 + (prior_fraud_count × 0.05), max 1.20
#                      Repeat offenders penalised more severely.
#                      (Recidivism weighting — used in insurance
#                       fraud scoring literature.)
#   overbilling_extra— +8 pts if BOTH GANs independently flag
#                      HIGH RISK (overbilling_flag = True from
#                      investigation_node). Two architecturally
#                      different models agreeing = very strong
#                      persistent fraud signal.
#
# ────────────────────────────────────────────────────────────
# DOCTOR STATUS THRESHOLDS
# ────────────────────────────────────────────────────────────
#   DTS 80–100  → TRUSTED      → Agent 2 recommends freely
#   DTS 60–79   → WATCH LIST   → Agent 2 recommends with note
#   DTS 40–59   → FLAGGED      → Agent 2 recommends only if no alt
#   DTS  0–39   → SUSPENDED    → Agent 2 does NOT recommend
#
# ────────────────────────────────────────────────────────────
# STATE CONTRACT
# ────────────────────────────────────────────────────────────
# INPUT  (reads from state — written by Node 2 investigation_node):
#   state['responsible_party']  — e.g. "Doctor", "Hospital",
#                                 "Doctor + Patient",
#                                 "Multiple: Doctor + Hospital"
#   state['primary_fraud_type'] — exact string from DIMENSION_META
#   state['confidence']         — float 0.0–1.0
#   state['overbilling_flag']   — bool
#   state['overall_risk']       — "HIGH RISK"/"MEDIUM RISK"/"NORMAL"
#   state['transaction']        — dict with DOCTOR_ID etc.
#   state['db']                 — SQLAlchemy session
#
# OUTPUT (writes to state — read by Node 4 rag_node):
#   state['doctor_score_penalty']  — float: points deducted
#   state['updated_doctor_score']  — float: new DTS (raw fraud score)
#   state['display_score']         — float: DTS × (stars/5)
#   state['score_breakdown']       — dict: full audit trail
# ============================================================

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from models_db import DoctorScore


# ─────────────────────────────────────────────────────────────
# RESPONSIBILITY GATE LOGIC
# ─────────────────────────────────────────────────────────────
# investigation_node writes responsible_party as one of:
#   "Doctor"
#   "Hospital"
#   "Doctor + Patient"
#   "Multiple: Doctor + Hospital"
#   "Multiple: Doctor + Hospital + Patient"
#
# Rule: penalise the doctor if and only if the string contains
# the word "Doctor". "Hospital" alone → skip entirely.
#
# "Multiple: Doctor + Hospital" → still penalise doctor because
# the doctor participated (e.g. DOCTOR_ID + SPECIALITY_ID both
# fired). The doctor is not protected just because the hospital
# also shares blame.

def _doctor_is_responsible(responsible_party: str) -> bool:
    return 'Doctor' in responsible_party


# ─────────────────────────────────────────────────────────────
# BASE PENALTY TABLE
# ─────────────────────────────────────────────────────────────
# These strings EXACTLY match the fraud_type values written by
# investigation_node via DIMENSION_META['<feature>']['fraud_type']
# and the pattern-level fallback labels.
#
# All penalties are BEFORE confidence and risk multipliers.
# Final effective penalty will be lower for borderline cases.

FRAUD_PENALTY_MAP = {

    # ── Doctor-responsible ────────────────────────────────────

    'Specialty Mismatch Fraud': 28,
    # DOCTOR_ID anomaly.
    # Doctor billing under a specialty inconsistent with their
    # normal profile. GAN learned per-doctor billing identity —
    # a mismatch here is a direct doctor-controlled act.
    # Base: 28 pts (high severity — identity-level fraud).

    'Diagnosis-Service Mismatch Fraud': 32,
    # DIAGNOSIS_ID anomaly.
    # Only the doctor writes the diagnosis. Attaching a diagnosis
    # that does not justify the service = deliberate upcoding.
    # Base: 32 pts (highest single-feature penalty — direct deception).

    'Patient-Doctor Collusion / Repeat Billing Fraud': 30,
    # PATIENT_ID anomaly.
    # Same patient dominating recent sequence → collusion.
    # Doctor initiates every billing event; both parties benefit.
    # Base: 30 pts (severe — active scheme involving two parties).

    'Consistent Anomalous Billing Sequence': 22,
    # Pattern-level: both GANs HIGH RISK, no single feature dominant.
    # Entire billing sequence is anomalous → systematic behaviour.
    # Base: 22 pts (lower than single-feature because cause is diffuse).

    'Anomalous Billing Pattern': 15,
    # Pattern-level: one GAN flagged, no clear dominant feature.
    # Borderline / ambiguous detection.
    # Base: 15 pts (lightest — system is least certain here).

    # ── Hospital-responsible ──────────────────────────────────
    # These entries exist for completeness and documentation.
    # They will NEVER be reached in practice because the
    # responsibility gate returns before penalty calculation
    # whenever the doctor is not responsible.

    'Specialty-Service Mismatch Fraud': 0,
    # SPECIALITY_ID anomaly → Hospital billing dept.

    'Phantom Billing / Upcoding': 0,
    # SERVICE_ID anomaly → Hospital coding dept.

    # ── Fallback ──────────────────────────────────────────────
    'Unknown Fraud': 10,
}


# ─────────────────────────────────────────────────────────────
# RISK MULTIPLIER
# ─────────────────────────────────────────────────────────────
# Applied on top of base penalty before confidence.
# Reflects how certain the majority-vote outcome is.
# HIGH RISK  = both or strongest model votes agree → full penalty
# MEDIUM RISK = mixed votes → 80% penalty

RISK_MULTIPLIER = {
    'HIGH RISK'  : 1.00,
    'MEDIUM RISK': 0.80,
    'NORMAL'     : 0.50,   # safety fallback — should never reach here
}

# Extra penalty added when overbilling_flag = True
# (both GAN models independently output HIGH RISK)
OVERBILLING_EXTRA = 8

# Score constants
DTS_START   = 100.0
DTS_FLOOR   = 0.0
MIN_PENALTY = 1.0    # at least 1 pt deducted per confirmed doctor fraud
MAX_PENALTY = 50.0   # no single transaction can wipe out a doctor's score


# ─────────────────────────────────────────────────────────────
# MAIN NODE
# ─────────────────────────────────────────────────────────────

def scoring_node(state: dict) -> dict:
    """
    LangGraph Node 3 — Doctor Trust Score (DTS) Update.

    Reads investigation results, applies responsibility gate,
    calculates penalty, updates doctor score in DB.

    Returns dict to merge into Agent1State.
    """

    print("\n" + "=" * 55)
    print("NODE 3 — DOCTOR TRUST SCORE (DTS)")
    print("=" * 55)

    # ── Read from state ───────────────────────────────────────
    transaction        = state['transaction']
    doctor_id          = transaction.get('DOCTOR_ID', 'Unknown')
    primary_fraud_type = state.get('primary_fraud_type', 'Unknown Fraud')
    responsible_party  = state.get('responsible_party', 'Unknown')
    confidence         = state.get('confidence', 0.5)
    overbilling_flag   = state.get('overbilling_flag', False)
    overall_risk       = state.get('overall_risk', 'MEDIUM RISK')
    db                 = state.get('db')

    print(f"Doctor ID      : {doctor_id}")
    print(f"Primary Fraud  : {primary_fraud_type}")
    print(f"Responsible    : {responsible_party}")
    print(f"Confidence     : {confidence:.4f}")
    print(f"Overall Risk   : {overall_risk}")
    print(f"Overbilling    : {overbilling_flag}")

    # ── Load existing doctor record from DB ───────────────────
    current_dts     = DTS_START
    previous_frauds = 0
    review_stars    = 5.0   # default if reviews not yet in DB

    if db:
        try:
            existing = db.query(DoctorScore).filter(
                DoctorScore.doctor_id == doctor_id
            ).first()
            if existing:
                current_dts     = float(existing.current_score)
                previous_frauds = int(existing.fraud_count)
                # review_stars column now exists in DoctorScore (added in models_db.py).
                # Populated by seed_review_stars.py from Reviews_new.csv.
                # Default 5.0 means no penalty from missing data (new/unknown doctors).
                review_stars = float(existing.review_stars) if existing.review_stars is not None else 5.0
                print(f"\nExisting DTS    : {current_dts}")
                print(f"Previous frauds : {previous_frauds}")
                print(f"Review stars    : {review_stars} / 5")
            else:
                print(f"\nNew doctor in scoring DB — starting at {DTS_START}")
        except Exception as e:
            print(f"\n⚠️  Could not read DB score: {e}")

    # ─────────────────────────────────────────────────────────
    # RESPONSIBILITY GATE
    # ─────────────────────────────────────────────────────────
    # If the responsible party does NOT contain "Doctor" at all
    # (i.e. this is a pure Hospital fraud: SPECIALITY_ID or
    # SERVICE_ID anomaly), we do NOT touch the doctor's DTS.
    #
    # We still compute display_score and return score_breakdown
    # so the RAG report can explain why no penalty was applied.
    # ─────────────────────────────────────────────────────────

    if not _doctor_is_responsible(responsible_party):

        print(f"\n🏥  HOSPITAL FRAUD DETECTED")
        print(f"   Responsible party : '{responsible_party}'")
        print(f"   Doctor NOT penalised — DTS unchanged: {current_dts}")

        display_score = round(current_dts * (review_stars / 5.0), 2)
        score_status  = _get_status(current_dts)

        score_breakdown = {
            'doctor_id'            : doctor_id,
            'responsible_party'    : responsible_party,
            'primary_fraud_type'   : primary_fraud_type,
            'doctor_penalised'     : False,
            'reason_not_penalised' : (
                f"Fraud type '{primary_fraud_type}' is attributed to the "
                f"hospital ({responsible_party}). The doctor did not control "
                f"the anomalous feature. DTS is unchanged per OIG attribution "
                f"principle — penalties apply only to the accountable party."
            ),
            'previous_score'       : current_dts,
            'updated_score'        : current_dts,
            'total_penalty'        : 0.0,
            'previous_frauds'      : previous_frauds,
            'review_stars'         : review_stars,
            'display_score'        : display_score,
            'score_status'         : score_status,
            'overall_risk'         : overall_risk,
        }

        return {
            'doctor_score_penalty' : 0.0,
            'updated_doctor_score' : current_dts,
            'display_score'        : display_score,
            'score_breakdown'      : score_breakdown,
        }

    # ─────────────────────────────────────────────────────────
    # DOCTOR IS RESPONSIBLE — CALCULATE PENALTY
    # ─────────────────────────────────────────────────────────

    print(f"\n👨‍⚕️  Doctor IS responsible — calculating penalty...")

    # Step 1 — Base penalty from fraud type
    base_penalty = FRAUD_PENALTY_MAP.get(
        primary_fraud_type,
        FRAUD_PENALTY_MAP['Unknown Fraud']
    )
    print(f"\n  Step 1 | base_penalty ({primary_fraud_type}): {base_penalty}")

    # Step 2 — Risk multiplier
    # Reflects majority-vote confidence in the fraud verdict.
    # HIGH RISK = both or dominant votes agree → full penalty.
    risk_mult  = RISK_MULTIPLIER.get(overall_risk, 0.80)
    after_risk = round(base_penalty * risk_mult, 4)
    print(f"  Step 2 | × risk_multiplier ({overall_risk} → {risk_mult}): {after_risk}")

    # Step 3 — Confidence multiplier
    # investigation_node computes confidence from three components:
    #   ablation_clarity (40%) — how clearly one feature dominated
    #   gan_agreement    (35%) — do both models agree on risk level?
    #   risk_weight      (25%) — how severe is the overall verdict?
    # A lower confidence means the cause is ambiguous → lighter penalty.
    after_conf = round(after_risk * confidence, 4)
    print(f"  Step 3 | × confidence ({confidence:.4f}): {after_conf}")

    # Step 4 — Repeat offender multiplier
    # Each prior verified fraud (fraud_count in DB) adds +5% penalty,
    # capped at +20% (equivalent to 4 prior frauds).
    # Rationale: recidivism weighting used in insurance fraud scoring.
    # First-time offenders get lighter effective penalty.
    repeat_bonus = min(previous_frauds * 0.05, 0.20)
    repeat_mult  = round(1.0 + repeat_bonus, 4)
    after_repeat = round(after_conf * repeat_mult, 4)
    if previous_frauds > 0:
        print(f"  Step 4 | × repeat_mult ({previous_frauds} prior frauds → {repeat_mult}): {after_repeat}")
    else:
        after_repeat = after_conf
        print(f"  Step 4 | repeat_mult = 1.0 (first offence)")

    # Step 5 — Overbilling extra penalty
    # Both GAN models (TTS-WGAN-GP and DCT-GAN) independently output
    # HIGH RISK → overbilling_flag = True (set by investigation_node).
    # Two architecturally different models agreeing = very strong
    # persistent anomaly across the entire 10-transaction sequence.
    overbilling_extra = OVERBILLING_EXTRA if overbilling_flag else 0
    if overbilling_flag:
        print(f"  Step 5 | + overbilling_extra (both GANs HIGH RISK): +{overbilling_extra}")

    # Step 6 — Total penalty (clamped between 1 and 50)
    raw_penalty   = after_repeat + overbilling_extra
    total_penalty = round(
        max(MIN_PENALTY, min(raw_penalty, MAX_PENALTY)), 2
    )
    print(f"\n  Raw penalty  : {raw_penalty:.4f}")
    print(f"  Clamped [{MIN_PENALTY}, {MAX_PENALTY}]: {total_penalty}")

    # Step 7 — New DTS (floor = 0)
    updated_dts = round(
        max(current_dts - total_penalty, DTS_FLOOR), 2
    )
    print(f"\n  DTS: {current_dts} − {total_penalty} = {updated_dts}")

    # Step 8 — Display score (integrates review stars for Agent 2)
    # display_score = DTS × (review_stars / 5)
    # Review stars come from Reviews_new.csv via Agent 2's pipeline.
    # If stored on DoctorScore model → used here.
    # If not available → defaults to 5 (no penalty from missing reviews).
    display_score = round(updated_dts * (review_stars / 5.0), 2)
    print(f"  display_score = {updated_dts} × ({review_stars}/5) = {display_score}")

    # Step 9 — Status label
    score_status = _get_status(updated_dts)
    print(f"  Doctor status: {score_status}")

    # Step 10 — Full audit breakdown for RAG report
    score_breakdown = {
        'doctor_id'           : doctor_id,
        'responsible_party'   : responsible_party,
        'primary_fraud_type'  : primary_fraud_type,
        'doctor_penalised'    : True,
        'overall_risk'        : overall_risk,
        'base_penalty'        : base_penalty,
        'risk_multiplier'     : risk_mult,
        'after_risk'          : after_risk,
        'confidence'          : confidence,
        'after_confidence'    : after_conf,
        'repeat_multiplier'   : repeat_mult,
        'previous_frauds'     : previous_frauds,
        'after_repeat'        : after_repeat,
        'overbilling_extra'   : overbilling_extra,
        'total_penalty'       : total_penalty,
        'previous_score'      : current_dts,
        'updated_score'       : updated_dts,
        'review_stars'        : review_stars,
        'display_score'       : display_score,
        'score_status'        : score_status,
    }

    # Step 11 — Save to DB
    if db:
        try:
            row = db.query(DoctorScore).filter(
                DoctorScore.doctor_id == doctor_id
            ).first()

            if not row:
                # First fraud event for this doctor
                row = DoctorScore(
                    doctor_id       = doctor_id,
                    current_score   = DTS_START,
                    total_penalties = 0.0,
                    fraud_count     = 0,
                )
                db.add(row)
                db.flush()   # get PK before update

            # Update fields
            row.current_score    = updated_dts
            row.total_penalties  = round(
                (row.total_penalties or 0.0) + total_penalty, 2
            )
            row.fraud_count     += 1
            row.last_updated     = datetime.utcnow()

            db.commit()
            print(f"\n  ✅ DTS saved to DB")

        except Exception as e:
            print(f"\n  ⚠️  Failed to save DTS: {e}")
            try:
                db.rollback()
            except Exception:
                pass
    else:
        print("\n  ⚠️  No DB session — DTS not persisted")

    # ── Final summary ─────────────────────────────────────────
    print(f"\nSCORING COMPLETE:")
    print(f"  {doctor_id}:")
    print(f"    DTS        : {current_dts} → {updated_dts}  ({score_status})")
    print(f"    Penalty    : −{total_penalty} pts")
    print(f"    Display    : {display_score}  (with {review_stars}★ reviews)")

    return {
        'doctor_score_penalty' : total_penalty,
        'updated_doctor_score' : updated_dts,
        'display_score'        : display_score,
        'score_breakdown'      : score_breakdown,
    }


# ─────────────────────────────────────────────────────────────
# HELPER — Status label from raw DTS
# ─────────────────────────────────────────────────────────────

def _get_status(dts: float) -> str:
    """
    Maps raw DTS to a status label used by Agent 2.

    Note: thresholds are applied to the RAW DTS, not display_score.
    Agent 2 should gate recommendations on DTS status, not display score.
    display_score is only for UI ranking/ordering.
    """
    if dts >= 80:
        return 'TRUSTED'
    elif dts >= 60:
        return 'WATCH LIST'
    elif dts >= 40:
        return 'FLAGGED'
    else:
        return 'SUSPENDED'