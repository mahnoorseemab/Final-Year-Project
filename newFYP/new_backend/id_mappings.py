# ============================================================
# id_mappings.py — Name ↔ ID Lookup Tables
#
# PURPOSE:
#   The GAN models were trained on IDs only (SPECIALITY_ID,
#   SERVICE_ID, DIAGNOSIS_ID, DOCTOR_ID). But the API receives
#   human-readable names from the user. This file bridges the gap.
#
# FLOW:
#   User sends names
#       ↓
#   names_to_ids()   ← converts to IDs for GAN inference
#       ↓
#   GAN runs on IDs → fraud result
#       ↓
#   ids_to_names()   ← converts back to names for the report
#       ↓
#   Result saved with real names (readable)
#
# DUPLICATE HANDLING:
#   Some names map to multiple IDs in the raw data (e.g. "Emergency"
#   had 3 different SPECIALITY_IDs due to data entry inconsistency).
#   We resolve this by always picking the most-frequent ID for that
#   name — i.e. whichever ID appeared most often in FYPDATA.csv.
#   This matches what the model saw most during training.
#
# UNKNOWN VALUES:
#   If the user sends a name not in our dataset (new specialty, new
#   service etc.), names_to_ids() returns None for that field and
#   the inference file will handle it as an unknown (median fallback).
# ============================================================

import os
import pandas as pd

# ── Module-level cache ─────────────────────────────────────────
# Built once at startup, reused for every request
_MAPS = None


def _build_maps(csv_path: str) -> dict:
    """
    Reads FYPDATA.csv once and builds all lookup dictionaries.
    Called automatically the first time get_maps() is called.

    Strategy for duplicates: take the most frequently occurring
    ID for each name (most representative of training data).
    """
    df = pd.read_csv(csv_path)

    # ── Helper: name → ID (most frequent ID for that name) ────
    def name_to_id_map(df, name_col, id_col):
        return (
            df.dropna(subset=[name_col, id_col])
            .groupby([name_col, id_col])
            .size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
            .drop_duplicates(name_col)
            .set_index(name_col)[id_col]
            .to_dict()
        )

    # ── Helper: ID → name (most frequent name for that ID) ────
    def id_to_name_map(df, id_col, name_col):
        return (
            df.dropna(subset=[name_col, id_col])
            .groupby([id_col, name_col])
            .size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
            .drop_duplicates(id_col)
            .set_index(id_col)[name_col]
            .to_dict()
        )

    # ── SPECIALITY ────────────────────────────────────────────
    spec_name_to_id = name_to_id_map(df, 'SPECIALITY_NAME', 'SPECIALITY_ID')
    spec_id_to_name = id_to_name_map(df, 'SPECIALITY_ID',   'SPECIALITY_NAME')
    # Convert ID keys to int for consistency
    spec_id_to_name = {int(k): str(v) for k, v in spec_id_to_name.items()}

    # ── SERVICE ───────────────────────────────────────────────
    svc_desc_to_id = name_to_id_map(df, 'SERVICE_DESCRIPTION', 'SERVICE_ID')
    svc_id_to_desc = id_to_name_map(df, 'SERVICE_ID',          'SERVICE_DESCRIPTION')
    # SERVICE_ID is already a string like "S_01"
    svc_id_to_desc = {str(k): str(v) for k, v in svc_id_to_desc.items()}

    # ── DIAGNOSIS ─────────────────────────────────────────────
    # Nulls mean "no diagnosis recorded" → DIAGNOSIS_ID = 0
    diag_name_to_id = name_to_id_map(df, 'DIAGNOSIS', 'DIAGNOSIS_ID')
    diag_name_to_id = {str(k): int(v) for k, v in diag_name_to_id.items()}
    diag_name_to_id['Unknown']     = 0   # explicit unknowns → 0
    diag_name_to_id['No Diagnosis']= 0

    diag_id_to_name = id_to_name_map(df, 'DIAGNOSIS_ID', 'DIAGNOSIS')
    diag_id_to_name = {int(k): str(v) for k, v in diag_id_to_name.items()}
    diag_id_to_name[0] = 'No Diagnosis'  # 0 → readable label

    # ── DOCTOR_ID ─────────────────────────────────────────────
    # DOCTOR_ID is already the ID itself (e.g. "doc_01")
    # No mapping needed — user sends doc_01, model uses doc_01
    # We still store the list of known doctors for validation
    known_doctors = set(df['DOCTOR_ID'].dropna().unique().tolist())

    return {
        'spec_name_to_id': spec_name_to_id,    # "Emergency" → 1
        'spec_id_to_name': spec_id_to_name,    # 1 → "Emergency"
        'svc_desc_to_id' : svc_desc_to_id,     # "HBA1C (HS16)" → "S_01"
        'svc_id_to_desc' : svc_id_to_desc,     # "S_01" → "HBA1C (HS16)"
        'diag_name_to_id': diag_name_to_id,    # "FEVER" → 16
        'diag_id_to_name': diag_id_to_name,    # 16 → "FEVER"
        'known_doctors'  : known_doctors,       # {"doc_01", "doc_02", ...}
    }


def get_maps(csv_path: str = None) -> dict:
    """
    Returns the lookup maps. Builds them on first call, then cached.

    Args:
        csv_path: path to FYPDATA.csv. Only needed on first call.
                  After that, the cache is returned without re-reading.

    Returns:
        dict with keys: spec_name_to_id, spec_id_to_name,
                        svc_desc_to_id, svc_id_to_desc,
                        diag_name_to_id, diag_id_to_name,
                        known_doctors
    """
    global _MAPS
    if _MAPS is None:
        if csv_path is None:
            # Default: look for FYPDATA.csv next to this file
            csv_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'FYPDATA.csv'
            )
        print(f"[id_mappings] Building lookup tables from: {csv_path}")
        _MAPS = _build_maps(csv_path)
        print(f"[id_mappings] Ready:")
        print(f"  Specialties : {len(_MAPS['spec_name_to_id'])} names")
        print(f"  Services    : {len(_MAPS['svc_desc_to_id'])} descriptions")
        print(f"  Diagnoses   : {len(_MAPS['diag_name_to_id'])} names")
        print(f"  Doctors     : {len(_MAPS['known_doctors'])} known IDs")
    return _MAPS


# ── PUBLIC FUNCTIONS ───────────────────────────────────────────

def names_to_ids(
    doctor_id        : str,   # already an ID: "doc_01"
    speciality_name  : str,   # "Emergency"
    service_desc     : str,   # "HBA1C (HS16)"
    diagnosis        : str,   # "FEVER" or "Unknown" / None
    patient_id       : int,   # just passed through, not mapped
    csv_path         : str = None
) -> dict:
    """
    Converts human-readable names to IDs for GAN inference.

    Returns a dict with:
        DOCTOR_ID     : str  (unchanged, e.g. "doc_01")
        SPECIALITY_ID : int  (e.g. 1 for Emergency)
        SERVICE_ID    : str  (e.g. "S_01")
        DIAGNOSIS_ID  : int  (e.g. 16 for FEVER, 0 for unknown)
        PATIENT_ID    : int  (unchanged)

        _names: dict with the original names (for saving results)
        _warnings: list of any unknown values that needed fallback
    """
    maps     = get_maps(csv_path)
    warnings = []

    # ── SPECIALITY_ID ─────────────────────────────────────────
    speciality_id = maps['spec_name_to_id'].get(speciality_name)
    if speciality_id is None:
        warnings.append(f"Unknown SPECIALITY_NAME: '{speciality_name}' — not in dataset")
        speciality_id = -1  # inference will use median fallback

    # ── SERVICE_ID ────────────────────────────────────────────
    service_id = maps['svc_desc_to_id'].get(service_desc)
    if service_id is None:
        warnings.append(f"Unknown SERVICE_DESCRIPTION: '{service_desc}' — not in dataset")
        service_id = 'UNKNOWN'  # inference will use median fallback

    # ── DIAGNOSIS_ID ──────────────────────────────────────────
    if diagnosis is None or str(diagnosis).strip() in ('', 'Unknown', 'unknown', 'None'):
        diagnosis_id = 0  # no diagnosis recorded
        diagnosis    = 'No Diagnosis'
    else:
        diagnosis_id = maps['diag_name_to_id'].get(str(diagnosis).strip())
        if diagnosis_id is None:
            warnings.append(f"Unknown DIAGNOSIS: '{diagnosis}' — not in dataset")
            diagnosis_id = 0

    # ── DOCTOR_ID ─────────────────────────────────────────────
    # Doctor ID is already the raw ID — just validate it's known
    if doctor_id not in maps['known_doctors']:
        warnings.append(f"Unknown DOCTOR_ID: '{doctor_id}' — not seen in training")

    if warnings:
        for w in warnings:
            print(f"[id_mappings] ⚠️  {w}")

    return {
        # IDs for GAN model
        'DOCTOR_ID'    : doctor_id,
        'SPECIALITY_ID': int(speciality_id) if speciality_id != -1 else None,
        'SERVICE_ID'   : service_id,
        'DIAGNOSIS_ID' : int(diagnosis_id),
        'PATIENT_ID'   : int(patient_id),

        # Original names (saved alongside results)
        '_names': {
            'doctor_id'          : doctor_id,
            'speciality_name'    : speciality_name,
            'service_description': service_desc,
            'diagnosis'          : diagnosis,
            'patient_id'         : patient_id,
        },
        '_warnings': warnings,
    }


def ids_to_names(
    doctor_id    : str,
    speciality_id: int,
    service_id   : str,
    diagnosis_id : int,
    csv_path     : str = None
) -> dict:
    """
    Converts IDs back to human-readable names for fraud reports.

    Use this when you want to display the result to the user
    or save it to the database with readable labels.

    Returns dict with:
        doctor_id, speciality_name, service_description, diagnosis
    """
    maps = get_maps(csv_path)

    return {
        'doctor_id'          : doctor_id,
        'speciality_name'    : maps['spec_id_to_name'].get(int(speciality_id), f'ID:{speciality_id}'),
        'service_description': maps['svc_id_to_desc'].get(str(service_id),   f'ID:{service_id}'),
        'diagnosis'          : maps['diag_id_to_name'].get(int(diagnosis_id), f'ID:{diagnosis_id}'),
    }


def get_all_specialties(csv_path: str = None) -> list:
    """Returns all known specialty names (for frontend dropdowns)."""
    maps = get_maps(csv_path)
    return sorted(maps['spec_name_to_id'].keys())


def get_all_services(csv_path: str = None) -> list:
    """Returns all known service descriptions (for frontend dropdowns)."""
    maps = get_maps(csv_path)
    return sorted(maps['svc_desc_to_id'].keys())


def get_all_diagnoses(csv_path: str = None) -> list:
    """Returns all known diagnosis names (for frontend dropdowns)."""
    maps = get_maps(csv_path)
    known = [k for k in maps['diag_name_to_id'].keys()
             if k not in ('Unknown', 'No Diagnosis')]
    return sorted(known)


def get_services_by_specialty(specialty_name: str, csv_path: str = None) -> list:
    """Returns services filtered by specialty name."""
    maps = get_maps(csv_path)
    
    # CSV se specialty-service mapping build karo
    import pandas as pd
    csv_path = csv_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'FYPDATA.csv'
    )
    df = pd.read_csv(csv_path)
    
    filtered = df[df['SPECIALITY_NAME'] == specialty_name]['SERVICE_DESCRIPTION']
    services = sorted(filtered.dropna().unique().tolist())
    return services

# ── Quick self-test ────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else 'FYPDATA.csv'

    print("=" * 55)
    print("ID MAPPINGS — SELF TEST")
    print("=" * 55)

    # Test: names → IDs
    result = names_to_ids(
        doctor_id       = 'doc_03',
        speciality_name = 'Emergency',
        service_desc    = 'HBA1C (HS16)',
        diagnosis       = None,
        patient_id      = 999,
        csv_path        = csv
    )
    print("\nnames_to_ids test:")
    print(f"  DOCTOR_ID     : {result['DOCTOR_ID']}")
    print(f"  SPECIALITY_ID : {result['SPECIALITY_ID']}")
    print(f"  SERVICE_ID    : {result['SERVICE_ID']}")
    print(f"  DIAGNOSIS_ID  : {result['DIAGNOSIS_ID']}")
    print(f"  Warnings      : {result['_warnings']}")

    # Test: IDs → names
    names = ids_to_names(
        doctor_id    = 'doc_03',
        speciality_id= result['SPECIALITY_ID'],
        service_id   = result['SERVICE_ID'],
        diagnosis_id = result['DIAGNOSIS_ID'],
        csv_path     = csv
    )
    print("\nids_to_names test:")
    print(f"  Specialty    : {names['speciality_name']}")
    print(f"  Service      : {names['service_description']}")
    print(f"  Diagnosis    : {names['diagnosis']}")

    # Test unknown
    print("\nUnknown value test:")
    result2 = names_to_ids(
        doctor_id       = 'doc_999',
        speciality_name = 'NonExistentSpecialty',
        service_desc    = 'Fake Service',
        diagnosis       = 'FEVER',
        patient_id      = 1,
        csv_path        = csv
    )
    print(f"  Warnings: {result2['_warnings']}")
    print(f"  DIAGNOSIS_ID for FEVER: {result2['DIAGNOSIS_ID']}")

    print("\n✅ Self-test complete!")