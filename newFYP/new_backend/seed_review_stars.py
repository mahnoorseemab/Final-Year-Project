# ============================================================
# seed_review_stars.py — ONE-TIME SETUP SCRIPT
#
# PURPOSE:
#   Reads review star ratings for FYP doctors (doc_01 to doc_118)
#   from Reviews_new.csv and inserts them into the doctor_scores
#   table in MySQL.
#
# WHEN TO RUN:
#   Run ONCE after creating your MySQL tables with models_db.py.
#   Safe to re-run — uses upsert logic (insert or update).
#   If a doctor row already exists, only review_stars is updated.
#   If a doctor row does not exist yet, a fresh row is created
#   with current_score=100, fraud_count=0.
#
# HOW IT WORKS:
#   1. Reads Reviews_new.csv
#   2. Filters rows where Doctor_Name matches doc_XX pattern
#   3. For each such doctor, reads their Reviews star value
#   4. Upserts into doctor_scores table
#
# USAGE:
#   python seed_review_stars.py
#   python seed_review_stars.py --csv path/to/Reviews_new.csv
# ============================================================

import os
import sys
import argparse
import re
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database  import SessionLocal, engine
from models_db import Base, DoctorScore


def seed_review_stars(csv_path: str, db: Session) -> dict:
    """
    Reads FYP doctor star ratings from Reviews_new.csv and
    upserts them into the doctor_scores table.

    Args:
        csv_path : path to Reviews_new.csv
        db       : SQLAlchemy session

    Returns:
        dict with counts: inserted, updated, skipped, errors
    """
    print(f"\nReading: {csv_path}")
    reviews = pd.read_csv(csv_path)

    # ── Filter to FYP doctor rows only ───────────────────────
    # These are rows where Doctor_Name is exactly "doc_XX"
    fyp_mask    = reviews['Doctor_Name'].str.match(r'^doc_\d+$', na=False)
    fyp_reviews = reviews[fyp_mask].copy()

    print(f"Total rows in CSV       : {len(reviews)}")
    print(f"FYP doctor rows found   : {len(fyp_reviews)}")

    if len(fyp_reviews) == 0:
        print("❌ No FYP doctor rows found. Check CSV path and format.")
        return {'inserted': 0, 'updated': 0, 'skipped': 0, 'errors': 0}

    # ── For doctors with multiple review rows, take the mean ─
    # (in our current CSV each doctor has exactly one row,
    #  but this handles future data safely)
    star_map = (
        fyp_reviews
        .groupby('Doctor_Name')['Reviews']
        .mean()
        .round(2)
        .to_dict()
    )

    print(f"Unique doctors to seed  : {len(star_map)}")
    print(f"\nStar distribution:")
    from collections import Counter
    dist = Counter(round(v) for v in star_map.values())
    for stars in sorted(dist):
        print(f"  {stars:.0f} stars: {dist[stars]} doctors")

    # ── Upsert each doctor into doctor_scores ────────────────
    inserted = 0
    updated  = 0
    errors   = 0

    for doctor_id, stars in sorted(star_map.items()):
        try:
            stars_float = float(stars)

            # Clamp to valid range 1.0–5.0
            stars_float = max(1.0, min(5.0, stars_float))

            existing = db.query(DoctorScore).filter(
                DoctorScore.doctor_id == doctor_id
            ).first()

            if existing:
                # Row exists — update only review_stars
                old_stars = existing.review_stars
                existing.review_stars = stars_float
                existing.last_updated = datetime.utcnow()
                updated += 1
                print(f"  ✏️  {doctor_id:12} → stars updated: {old_stars} → {stars_float}")
            else:
                # New row — insert with default score=100, fraud_count=0
                new_row = DoctorScore(
                    doctor_id       = doctor_id,
                    current_score   = 100.0,
                    total_penalties = 0.0,
                    fraud_count     = 0,
                    review_stars    = stars_float,
                    last_updated    = datetime.utcnow(),
                )
                db.add(new_row)
                inserted += 1
                print(f"  ➕ {doctor_id:12} → inserted with stars={stars_float}")

        except Exception as e:
            print(f"  ❌ {doctor_id}: {e}")
            errors += 1

    db.commit()

    summary = {
        'inserted': inserted,
        'updated' : updated,
        'skipped' : 0,
        'errors'  : errors,
    }

    print(f"\n{'='*45}")
    print(f"SEEDING COMPLETE")
    print(f"{'='*45}")
    print(f"  Inserted (new rows) : {inserted}")
    print(f"  Updated (stars only): {updated}")
    print(f"  Errors              : {errors}")
    print(f"\nDoctor review stars are now in the database.")
    print(f"scoring_node will use them for display_score calculation.")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Seed doctor review stars from Reviews_new.csv into doctor_scores table.'
    )
    parser.add_argument(
        '--csv',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Reviews_new.csv'),
        help='Path to Reviews_new.csv (default: same folder as this script)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ CSV not found: {args.csv}")
        print(f"   Pass the correct path with --csv path/to/Reviews_new.csv")
        sys.exit(1)

    # Ensure tables exist
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        seed_review_stars(args.csv, db)
    finally:
        db.close()