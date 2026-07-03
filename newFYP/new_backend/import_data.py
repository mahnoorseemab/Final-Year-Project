import pandas as pd
from sqlalchemy import create_engine

# ─────────────────────────────────────────────
# CHANGE THESE TO YOUR MYSQL DETAILS
# ─────────────────────────────────────────────
MYSQL_USER     = "root"
MYSQL_PASSWORD = "1234"   # ← change this!
MYSQL_HOST     = "localhost"
MYSQL_PORT     = "3306"
MYSQL_DATABASE = "fraud_detection"

# ─────────────────────────────────────────────
# CSV FILE PATH
# ─────────────────────────────────────────────
CSV_PATH = r"C:\Users\Itcomplex\OneDrive\Documents\FYP WORK\FYPDATA.csv"

# ─────────────────────────────────────────────
# CONNECT TO MYSQL
# ─────────────────────────────────────────────
print("Connecting to MySQL...")
engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)
print("✅ Connected!")

# ─────────────────────────────────────────────
# READ AND CLEAN CSV
# ─────────────────────────────────────────────
print("Reading CSV...")
df = pd.read_csv(CSV_PATH)
print(f"   CSV loaded: {len(df)} rows")
print(f"   Columns: {df.columns.tolist()}")

# Drop corrupted columns
df = df.drop(columns=['SPECIALITY_ID', 'DIAGNOSIS_ID'])
print("   Dropped SPECIALITY_ID and DIAGNOSIS_ID")

# Fill missing diagnosis
df['DIAGNOSIS'] = df['DIAGNOSIS'].fillna('Unknown')
print("   Filled missing DIAGNOSIS with Unknown")

# Rename columns to match table
df = df.rename(columns={
    'PATIENT_ID'          : 'patient_id',
    'SPECIALITY_NAME'     : 'speciality_name',
    'DIAGNOSIS'           : 'diagnosis',
    'SERVICE_DESCRIPTION' : 'service_description',
    'SERVICE_ID'          : 'service_id',
    'DOCTOR_ID'           : 'doctor_id'
})
print(f"   Columns renamed: {df.columns.tolist()}")

# ─────────────────────────────────────────────
# UPLOAD TO MYSQL
# ─────────────────────────────────────────────
print("\nUploading to MySQL...")
df.to_sql(
    'transactions',
    engine,
    if_exists = 'append',   # add to existing table
    index     = False        # don't save dataframe index
)
print(f"✅ {len(df)} records imported successfully!")

# ─────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────
with engine.connect() as conn:
    result = conn.execute(
        __import__('sqlalchemy').text("SELECT COUNT(*) FROM transactions")
    )
    count = result.fetchone()[0]
    print(f"✅ Total rows in database: {count}")

    result2 = conn.execute(
        __import__('sqlalchemy').text(
            "SELECT COUNT(*) FROM transactions WHERE doctor_id = 'doc_64'"
        )
    )
    doc64_count = result2.fetchone()[0]
    print(f"✅ doc_64 transactions: {doc64_count}")

print("\n🎯 Database ready for fraud detection!")