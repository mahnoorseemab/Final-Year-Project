"""
=============================================================
MEDETECTIVE — Agent 2: Hybrid RL Recommendation System
=============================================================
Techniques used (from 2025-2026 research papers):
  1. Bi-Clustering       → doctors + features dono cluster (Paper: UET Taxila 2025)
  2. Content-Based       → doctor ki apni features (specialty, fee, avg_rating)
  3. Collaborative       → similar patients ne kisse visit kiya
  4. Epsilon-Greedy RL   → recommendations improve over time via feedback
=============================================================
Dataset: Reviews_new.csv
  - Employee_ID  : patient ID (E1-E100 = real patients, P1-P118 = FYP doctors placeholder)
  - Doctor_Name  : doctor name OR doc_XX id
  - Dept_visited : specialty / department
  - Reviews      : rating 1-5 (blank/NaN for new FYP doctors — filled by Agent 1 later)
  - Fee          : consultation fee
  - Hospital_Name: hospital
  - source       : 'reviews' (real doctors) or 'agent1' (FYP doc_XX doctors)
=============================================================

FIX SUMMARY:
  FIX 1 — load_and_clean: do NOT dropna on Reviews for the whole df.
           Split into reviewed_df (has Reviews) and unreviewed_df (blank Reviews).
           Biclustering and RL pre-training use only reviewed_df.
           Doctor profiles use ALL rows so new FYP doctors appear in recommendations.
  FIX 2 — load_and_clean: add 'source' column so agent2_graph.py can read it.
  FIX 3 — dept_fixes: removed renames that conflict with new FYP doctor dept names
           (Pediatrician, Pulmonologist, Dermatologist, Rheumatologist,
            Opthalmologist, Endocrinologist) — these are valid dept names in our CSV.
  FIX 4 — build_doctor_profiles: for doctors with no Reviews (new FYP doctors),
           avg_rating defaults to 3.0 (neutral), rating_std defaults to 0.
  FIX 5 — apply_biclustering: only uses reviewed_df (rows with real Reviews)
           so NaN Reviews don't break the pivot table.
  FIX 6 — New FYP doctors still appear in recommendations via content score
           (avg_rating=3.0 neutral) even before Agent 1 fills their Reviews.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import SpectralBiclustering
import warnings
warnings.filterwarnings("ignore")


# ==============================================================
# STEP 1 — Load & Clean Data
# FIX 1: Do NOT drop rows with blank Reviews.
# FIX 2: Add 'source' column (reviews vs agent1).
# FIX 3: Removed dept_fixes that conflict with FYP doctor dept names.
# ==============================================================

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # FIX 3 — Only fix genuine typos in original Reviews doctors.
    # Do NOT rename Pediatrician, Pulmonologist, Dermatologist,
    # Rheumatologist, Opthalmologist, Endocrinologist — these are
    # valid dept names used by our new FYP doctor rows.
    dept_fixes = {
        "Caardiology"  : "Cardiology",
        "Orthopaedicss": "Orthopaedics",
        "Child Specialist": "Paediatrics",
        "Gastroentrology": "Gastroenterology",
    }
    df["Dept_visited"] = df["Dept_visited"].replace(dept_fixes)

    # Convert Reviews to numeric — blank/NaN stays as NaN
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    # FIX 2 — Add source column:
    # Detect FYP doctor rows by: Doctor_Name starts with 'doc_'
    # This works regardless of whether Employee_ID is NaN or P-prefix
    df["source"] = df["Doctor_Name"].apply(
        lambda x: "agent1" if str(x).startswith("doc_") else "reviews"
    )

    # FIX 1 — Drop only rows where Doctor_Name is missing.
    # Keep rows with blank Reviews (new FYP doctors).
    df.dropna(subset=["Doctor_Name"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fill NaN Employee_ID for FYP rows so groupby count works
    df["Employee_ID"] = df["Employee_ID"].astype(str)
    mask = df["source"] == "agent1"
    df.loc[mask, "Employee_ID"] = [
        f"P{i+1}" for i in range(mask.sum())
    ]

    # Fill NaN Hospital_Name for FYP rows
    hospitals = ["AFIC", "CMH", "MH", "Shifa International Hospital"]
    import random; random.seed(42)
    nan_hosp = df["Hospital_Name"].isna()
    df.loc[nan_hosp, "Hospital_Name"] = [
        random.choice(hospitals) for _ in range(nan_hosp.sum())
    ]

    return df


# ==============================================================
# STEP 2 — Doctor Feature Profile (for Content-Based Filtering)
# FIX 4 — New FYP doctors have no Reviews → default avg_rating=3.0
# ==============================================================

def build_doctor_profiles(df: pd.DataFrame):
    """
    Har doctor ke liye ek feature vector banao.
    FYP doctors (doc_XX) have NaN Reviews → avg_rating defaults to 3.0 (neutral).
    """
    # Separate reviewed vs unreviewed
    reviewed   = df[df["Reviews"].notna()].copy()
    unreviewed = df[df["Reviews"].isna()].copy()

    # Profiles from reviewed doctors
    profiles_reviewed = reviewed.groupby("Doctor_Name").agg(
        avg_rating   =("Reviews",      "mean"),
        total_visits =("Employee_ID",  "count"),
        avg_fee      =("Fee",          "mean"),
        rating_std   =("Reviews",      "std"),
        department   =("Dept_visited", lambda x: x.mode()[0]),
        hospital     =("Hospital_Name",lambda x: x.mode()[0]),
    ).reset_index()

    # Profiles from unreviewed doctors (new FYP doc_XX)
    if not unreviewed.empty:
        profiles_unreviewed = unreviewed.groupby("Doctor_Name").agg(
            avg_fee      =("Fee",          "mean"),
            total_visits =("Employee_ID",  "count"),
            department   =("Dept_visited", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else "General Medicine"),
            hospital     =("Hospital_Name",lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else "CMH"),
        ).reset_index()
        # Default neutral rating for unreviewed doctors
        profiles_unreviewed["avg_rating"] = 3.0
        profiles_unreviewed["rating_std"] = 0.0
        profiles_reviewed = pd.concat(
            [profiles_reviewed, profiles_unreviewed], ignore_index=True
        )

    profiles = profiles_reviewed.copy()
    profiles["rating_std"] = profiles["rating_std"].fillna(0)
    profiles["avg_fee"]    = profiles["avg_fee"].fillna(0)

    # Encode categorical features
    le_dept = LabelEncoder()
    le_hosp = LabelEncoder()
    profiles["dept_encoded"]    = le_dept.fit_transform(profiles["department"])
    profiles["hospital_encoded"]= le_hosp.fit_transform(profiles["hospital"])

    return profiles, le_dept, le_hosp


# ==============================================================
# STEP 3 — Bi-Clustering (Paper 1: UET Taxila 2025)
# FIX 5 — Only use rows that have real Reviews for pivot/biclustering.
#          New FYP doctor rows (blank Reviews) are excluded here.
# ==============================================================

def apply_biclustering(df: pd.DataFrame, n_clusters: int = 5):
    """
    Patient-Doctor matrix banao, phir Spectral Bi-Clustering apply karo.
    Only uses rows with real Reviews — NaN rows excluded.

    Returns:
      - patient_clusters dict
      - doctor_clusters dict
      - pivot matrix
    """
    # FIX 5 — filter to only reviewed rows before pivot
    reviewed = df[df["Reviews"].notna()].copy()

    pivot = reviewed.pivot_table(
        index="Employee_ID",
        columns="Doctor_Name",
        values="Reviews",
        aggfunc="mean"
    ).fillna(0)

    n_cl = min(n_clusters, min(pivot.shape) - 1)
    model = SpectralBiclustering(n_clusters=n_cl, method="log", random_state=42)
    model.fit(pivot.values)

    patient_clusters = {
        patient: int(model.row_labels_[i])
        for i, patient in enumerate(pivot.index)
    }
    doctor_clusters = {
        doctor: int(model.column_labels_[i])
        for i, doctor in enumerate(pivot.columns)
    }

    return patient_clusters, doctor_clusters, pivot


# ==============================================================
# STEP 4 — Content-Based Score
# ==============================================================

def content_based_score(
    doctor_name: str,
    profiles: pd.DataFrame,
    preferred_dept: str = None,
    max_fee: int = None
) -> float:
    row = profiles[profiles["Doctor_Name"] == doctor_name]
    if row.empty:
        return 0.0

    row = row.iloc[0]

    if preferred_dept and row["department"] != preferred_dept:
        return 0.0

    # Skip fee filter if fee is 0 (unknown for FYP doctors)
    if max_fee and row["avg_fee"] > 0 and row["avg_fee"] > max_fee:
        return 0.0

    score = (
        (row["avg_rating"] / 5.0) * 0.5 +
        min(row["total_visits"] / 50.0, 1.0) * 0.3 +
        (1 - min(row["rating_std"] / 2.0, 1.0)) * 0.2
    )
    return round(float(score), 4)


# ==============================================================
# STEP 5 — Collaborative Filtering Score
# ==============================================================

def collaborative_score(
    patient_id: str,
    doctor_name: str,
    pivot: pd.DataFrame,
    patient_clusters: dict,
    doctor_clusters: dict,
    profiles: pd.DataFrame
) -> float:
    if patient_id not in patient_clusters:
        return 0.0
    if doctor_name not in doctor_clusters:
        return 0.0

    patient_cluster = patient_clusters[patient_id]

    similar_patients = [
        p for p, c in patient_clusters.items()
        if c == patient_cluster and p != patient_id
    ]

    if not similar_patients or doctor_name not in pivot.columns:
        return 0.0

    ratings = pivot.loc[
        [p for p in similar_patients if p in pivot.index],
        doctor_name
    ]
    ratings = ratings[ratings > 0]

    if ratings.empty:
        return 0.0

    avg = ratings.mean()
    return round(float(avg / 5.0), 4)


# ==============================================================
# STEP 6 — Epsilon-Greedy RL Agent
# ==============================================================

class EpsilonGreedyRecommender:
    """
    Multi-Armed Bandit (Epsilon-Greedy) RL Agent.
    Epsilon = 0.1 → 10% explore, 90% exploit.
    New FYP doctors start at Q=0.6 (neutral, matching 3.0/5.0 rating).
    """

    def __init__(self, doctor_names: list, epsilon: float = 0.1):
        self.epsilon     = epsilon
        self.doctor_names= doctor_names
        # FIX 6 — Initialize new FYP doctors (doc_XX) at 0.6 (neutral 3/5)
        #          instead of 0.0 so they are not immediately disadvantaged
        self.q_values    = {
            doc: (0.6 if str(doc).startswith("doc_") else 0.0)
            for doc in doctor_names
        }
        self.visit_counts= {doc: 0 for doc in doctor_names}

    def recommend(self, candidates: list) -> str:
        if not candidates:
            return None
        if np.random.rand() < self.epsilon:
            return np.random.choice(candidates)
        else:
            return max(candidates, key=lambda d: self.q_values.get(d, 0))

    def update(self, doctor_name: str, reward: float):
        if doctor_name not in self.q_values:
            self.q_values[doctor_name]    = 0.6 if str(doctor_name).startswith("doc_") else 0.0
            self.visit_counts[doctor_name]= 0

        self.visit_counts[doctor_name] += 1
        n     = self.visit_counts[doctor_name]
        old_q = self.q_values[doctor_name]
        self.q_values[doctor_name] = old_q + (1 / n) * (reward - old_q)

    def get_q_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Doctor"            : list(self.q_values.keys()),
            "Q_Value"           : list(self.q_values.values()),
            "Times_Recommended" : list(self.visit_counts.values())
        }).sort_values("Q_Value", ascending=False)


# ==============================================================
# STEP 7 — Hybrid Score (Content + Collaborative + RL)
# ==============================================================

def hybrid_score(
    patient_id: str,
    doctor_name: str,
    profiles: pd.DataFrame,
    pivot: pd.DataFrame,
    patient_clusters: dict,
    doctor_clusters: dict,
    rl_agent: EpsilonGreedyRecommender,
    preferred_dept: str = None,
    max_fee: int = None,
    weights: dict = None
) -> float:
    if weights is None:
        weights = {"content": 0.4, "collab": 0.4, "rl": 0.2}

    cb = content_based_score(doctor_name, profiles, preferred_dept, max_fee)
    cf = collaborative_score(
        patient_id, doctor_name, pivot,
        patient_clusters, doctor_clusters, profiles
    )
    rl = rl_agent.q_values.get(doctor_name, 0.0)

    score = (
        weights["content"] * cb +
        weights["collab"]  * cf +
        weights["rl"]      * rl
    )
    return round(score, 4)


# ==============================================================
# STEP 8 — Main Recommendation Function
# ==============================================================

def recommend_doctors(
    patient_id: str,
    preferred_dept: str = None,
    max_fee: int = None,
    top_n: int = 5,
    df: pd.DataFrame = None,
    profiles: pd.DataFrame = None,
    pivot: pd.DataFrame = None,
    patient_clusters: dict = None,
    doctor_clusters: dict = None,
    rl_agent: EpsilonGreedyRecommender = None
) -> pd.DataFrame:

    if preferred_dept:
        candidates = profiles[
            profiles["department"] == preferred_dept
        ]["Doctor_Name"].tolist()
    else:
        candidates = profiles["Doctor_Name"].tolist()

    if not candidates:
        print(f"No doctors found for dept: {preferred_dept}")
        return pd.DataFrame()

    scored = []
    for doc in candidates:
        score = hybrid_score(
            patient_id, doc, profiles, pivot,
            patient_clusters, doctor_clusters, rl_agent,
            preferred_dept, max_fee
        )
        if score > 0:
            row = profiles[profiles["Doctor_Name"] == doc].iloc[0]
            fee_val = int(row["avg_fee"]) if row["avg_fee"] > 0 else None
            scored.append({
                "Doctor_Name"  : doc,
                "Department"   : row["department"],
                "Hospital"     : row["hospital"],
                "Avg_Fee"      : fee_val,
                "Avg_Rating"   : round(row["avg_rating"], 2),
                "Total_Visits" : int(row["total_visits"]),
                "Hybrid_Score" : score
            })

    if not scored:
        return pd.DataFrame()

    result = pd.DataFrame(scored).sort_values(
        "Hybrid_Score", ascending=False
    ).head(top_n)
    result.reset_index(drop=True, inplace=True)
    result.index += 1

    top_docs = result["Doctor_Name"].tolist()
    rl_pick  = rl_agent.recommend(top_docs)
    result["RL_Top_Pick"] = result["Doctor_Name"].apply(
        lambda x: "⭐ RL Pick" if x == rl_pick else ""
    )

    return result


# ==============================================================
# STEP 9 — Feedback Loop (RL Update)
# ==============================================================

def submit_feedback(
    doctor_name: str,
    actual_rating: float,
    rl_agent: EpsilonGreedyRecommender
):
    reward = actual_rating / 5.0
    rl_agent.update(doctor_name, reward)
    print(f"[RL Update] {doctor_name} → new Q = {rl_agent.q_values[doctor_name]:.4f}")


# ==============================================================
# STEP 10 — Initialize Everything (call once at startup)
# ==============================================================

def initialize_agent2(filepath: str):
    print("Loading data...")
    df = load_and_clean(filepath)

    print("Building doctor profiles...")
    profiles, le_dept, le_hosp = build_doctor_profiles(df)

    print("Applying Bi-Clustering (SpectralBiclustering)...")
    patient_clusters, doctor_clusters, pivot = apply_biclustering(df, n_clusters=5)

    print(f"Doctors clustered: {len(doctor_clusters)}")
    print(f"Patients clustered: {len(patient_clusters)}")

    print("Initializing RL Agent (Epsilon-Greedy)...")
    rl_agent = EpsilonGreedyRecommender(
        doctor_names=profiles["Doctor_Name"].tolist(),
        epsilon=0.1
    )

    # Pre-train RL only on rows that have real Reviews
    print("Pre-training RL on existing review data...")
    reviewed = df[df["Reviews"].notna()]
    for _, row in reviewed.iterrows():
        rl_agent.update(row["Doctor_Name"], row["Reviews"] / 5.0)

    print("Agent 2 ready!\n")
    return df, profiles, pivot, patient_clusters, doctor_clusters, rl_agent