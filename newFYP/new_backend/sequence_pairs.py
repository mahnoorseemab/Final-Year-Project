# ============================================================
# sequence_pairs.py
# SHARED MODULE — builds the 8 (group_col, feature_col) sequences
# used by BOTH ttsgan_inference.py and dctgan_inference.py.
#
# WHY THIS EXISTS:
#   TTSWGAN___DCTGAN_FINAL.ipynb (Chunk 6) trains on 8 separate
#   single-scalar sequence types, combined into one pool and
#   normalized by ONE shared scalar (raw_max):
#
#     SEQUENCE_PAIRS = [
#       ('DOCTOR_ID',     'SERVICE_ID'),
#       ('SERVICE_ID',    'DOCTOR_ID'),
#       ('SPECIALITY_ID', 'SERVICE_ID'),
#       ('SERVICE_ID',    'SPECIALITY_ID'),
#       ('SPECIALITY_ID', 'DIAGNOSIS_ID'),
#       ('DIAGNOSIS_ID',  'SPECIALITY_ID'),
#       ('SERVICE_ID',    'DIAGNOSIS_ID'),
#       ('DIAGNOSIS_ID',  'SERVICE_ID'),
#     ]
#
#   Each sequence is: group by `group_col`, take that entity's
#   values of `feature_col` (last 10, zero-padded at the START
#   if fewer than 10), normalize by the single scalar raw_max.
#   Final tensor shape per pair: (1, 10, 1).
#
#   At inference, for a single incoming transaction we cannot
#   "group by" anything -- instead we look up the DB history of
#   the relevant entity (e.g. last 10 records for this DOCTOR_ID)
#   and read off the feature_col values in the same order training
#   used (oldest -> newest, current transaction is most recent).
#
# PADDING — MUST MATCH TRAINING:
#   Training pads with ZEROS at the START for short sequences
#   (see create_one_sequence_per_entity in the notebook). This
#   module follows the same rule -- it does NOT cycle/repeat
#   existing records like crud.handle_cold_start_10 does for the
#   old design. Zero-padding here is the correct match to training.
#
# NORMALIZATION:
#   raw_max is the single scalar saved by the notebook to
#   feature_maxes.npy (shape (1,)). This is NOT the same as the
#   old per-column feature_maxes.npy (shape (5,)) used by the
#   pre-FINAL inference code -- that file must be regenerated
#   from the FINAL notebook and replaced in models/.
# ============================================================

import numpy as np
import torch
import pickle
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN = 10

# ── The 8 pair-types, in the exact order trained ──────────────
SEQUENCE_PAIRS = [
    ('DOCTOR_ID',     'SERVICE_ID'),
    ('SERVICE_ID',    'DOCTOR_ID'),
    ('SPECIALITY_ID', 'SERVICE_ID'),
    ('SERVICE_ID',    'SPECIALITY_ID'),
    ('SPECIALITY_ID', 'DIAGNOSIS_ID'),
    ('DIAGNOSIS_ID',  'SPECIALITY_ID'),
    ('SERVICE_ID',    'DIAGNOSIS_ID'),
    ('DIAGNOSIS_ID',  'SERVICE_ID'),
]

PAIR_KEYS = [f"{g}__to__{f}" for g, f in SEQUENCE_PAIRS]

# ── Weights for combining the 8 pair-type scores ───────────────
# Must sum to 1.0. Doctor- and service-centred pairs weighted
# slightly higher since they carry the strongest fraud signal in
# healthcare billing; symmetric reverse-direction pairs and the
# specialty/diagnosis pairs share the remaining weight evenly.
PAIR_WEIGHTS = {
    'DOCTOR_ID__to__SERVICE_ID'    : 0.16,
    'SERVICE_ID__to__DOCTOR_ID'    : 0.14,
    'SPECIALITY_ID__to__SERVICE_ID': 0.12,
    'SERVICE_ID__to__SPECIALITY_ID': 0.12,
    'SPECIALITY_ID__to__DIAGNOSIS_ID': 0.11,
    'DIAGNOSIS_ID__to__SPECIALITY_ID': 0.11,
    'SERVICE_ID__to__DIAGNOSIS_ID' : 0.12,
    'DIAGNOSIS_ID__to__SERVICE_ID' : 0.12,
}
assert abs(sum(PAIR_WEIGHTS.values()) - 1.0) < 1e-9, "PAIR_WEIGHTS must sum to 1.0"


# ── Load encoders + shared normalization scalar ────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

with open(os.path.join(MODELS_DIR, 'encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)

# feature_maxes.npy from the FINAL notebook is a single-value array:
# np.save('feature_maxes.npy', np.array([raw_max]))
_feature_maxes_arr = np.load(os.path.join(MODELS_DIR, 'feature_maxes.npy'))
RAW_MAX = float(_feature_maxes_arr[0]) if _feature_maxes_arr.size > 0 else 1.0
if RAW_MAX <= 0:
    RAW_MAX = 1.0

print(f"[sequence_pairs] Loaded RAW_MAX = {RAW_MAX} (shared normalization scalar)")
print(f"[sequence_pairs] {len(SEQUENCE_PAIRS)} pair-types configured")


# ─────────────────────────────────────────────
# HELPER — Normalize an ID value string for LabelEncoder lookup
# Matches exact string format used during training (Chunk 5)
# ─────────────────────────────────────────────
def normalize_id_for_lookup(col: str, val) -> str:
    if col == 'DIAGNOSIS_ID':
        try:
            return str(float(val))
        except (ValueError, TypeError):
            return str(val)
    elif col in ('SPECIALITY_ID', 'PATIENT_ID'):
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return str(val)
    else:
        return str(val)


def _encode_id(col: str, val) -> tuple:
    """
    Encodes a single raw ID value using the trained LabelEncoder
    for that column. Returns (encoded_int, warning_or_None).
    Unseen values fall back to the median encoded index, same
    policy as the previous inference code.
    """
    le  = encoders[col]
    key = normalize_id_for_lookup(col, val)

    if key in le.classes_:
        return int(le.transform([key])[0]), None

    median_idx = len(le.classes_) // 2
    warning = f"Unseen {col}: '{key}' -> median fallback ({median_idx})"
    return median_idx, warning


# ─────────────────────────────────────────────
# CORE — Build one pair-type's (1, 10, 1) tensor for inference
#
# group_col / feature_col   : the pair, e.g. ('DOCTOR_ID', 'SERVICE_ID')
# group_value                : the raw value of group_col for this
#                               transaction (e.g. the doctor's ID)
# current_feature_value      : the raw value of feature_col for the
#                               CURRENT (incoming) transaction
# history_feature_values     : list of feature_col raw values from
#                               the entity's DB history, OLDEST FIRST,
#                               NOT including the current transaction
#
# Sequence order matches training: chronological, most recent last.
# Padding: zeros at the START if fewer than SEQ_LEN values exist --
# this matches create_one_sequence_per_entity() in the notebook,
# NOT crud.handle_cold_start_10()'s cycling behaviour.
# ─────────────────────────────────────────────
def build_pair_tensor(
    feature_col           : str,
    current_feature_value,
    history_feature_values: list,
) -> tuple:
    """
    Returns:
        tensor   : torch.FloatTensor shape (1, SEQ_LEN, 1)
        warning  : str or None (unseen-value fallback message)
        is_padded: bool -- True if zero-padding was applied
    """
    # Encode every value (history is oldest-first; append current last)
    encoded_values = []
    warning = None

    for raw_val in history_feature_values:
        enc, w = _encode_id(feature_col, raw_val)
        encoded_values.append(enc)
        if w and warning is None:
            warning = w

    current_enc, w = _encode_id(feature_col, current_feature_value)
    if w and warning is None:
        warning = w
    encoded_values.append(current_enc)  # current transaction = most recent = last

    # Keep only the last SEQ_LEN values (most recent), matching training
    encoded_values = encoded_values[-SEQ_LEN:]

    is_padded = len(encoded_values) < SEQ_LEN
    if is_padded:
        pad_len = SEQ_LEN - len(encoded_values)
        encoded_values = [0] * pad_len + encoded_values  # zero-pad at START

    arr = np.array(encoded_values, dtype=np.float32) / RAW_MAX  # (10,)
    arr = arr.reshape(1, SEQ_LEN, 1)

    tensor = torch.FloatTensor(arr).to(device)
    return tensor, warning, is_padded


def pair_key(group_col: str, feature_col: str) -> str:
    return f"{group_col}__to__{feature_col}"
