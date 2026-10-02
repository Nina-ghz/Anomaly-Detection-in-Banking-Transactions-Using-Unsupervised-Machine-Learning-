import random
from datetime import timedelta
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

INPUT_CSV = "../TEST3/transactions_client_final.csv"
OUTPUT_CSV = "transactions_client_final_injected325.csv"

N_PER_TYPE = {
    "NIGHT_RAPID_BURST": 60,
    "NIGHT_LARGE_AMOUNT": 60,
    "SMURFING_RAPID_BURST": 60,
    "UNUSUAL_SPENDING": 60,
    "FOREIGN_SPIKE": 45,
    "FOREIGN_NIGHT": 40,
}
assert sum(N_PER_TYPE.values()) == 325

BRANCH_CODE = 10000
CIF_PREFIX = "7"
CIF_LENGTH = 9       
TRS_NO_PREFIX = "9999"

LOCAL_CURRENCY = 12

BASELINE_TXNS_PER_CLIENT = (6, 10)     
BASELINE_WINDOW_DAYS = (30, 180)      
INJECTION_START_BUFFER_DAYS = 15      
INJECTION_SPAN_DAYS = 500          

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
print("Original dataset:", df.shape)

df["TRS_AC_CIF"] = df["TRS_AC_CIF"].astype(str)
df["TRS_CY"] = pd.to_numeric(df["TRS_CY"], errors="coerce")
df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")

df["INJECTED"] = 0
df["ANOMALY_TYPE"] = "ORIGINAL"

# ============================================================
# REAL GL CODES / CURRENCY CODES / DATE RANGE / CIF COLLISION SET
# ============================================================

real_gl_codes = (
    pd.to_numeric(df["TRS_AC_GL"], errors="coerce")
    .dropna()
    .astype(int)
    .unique()
)

foreign_codes = [
    c for c in df["TRS_CY"].dropna().astype(int).unique()
    if c != LOCAL_CURRENCY
]

real_cif_set = set(df["TRS_AC_CIF"].unique())

max_real_date = df["DATETIME"].max()

# ID GENERATORS

_trs_counter = 1
_used_cifs = set()


def next_trs_no() -> str:
    global _trs_counter
    val = f"{TRS_NO_PREFIX}{_trs_counter:06d}"
    _trs_counter += 1
    return val


def next_synthetic_cif() -> str:
    """Generate a CIF starting with 7 that collides with nothing real or
    already-used in this run."""
    while True:
        digits = "".join(random.choices("0123456789", k=CIF_LENGTH - len(CIF_PREFIX)))
        cif = f"{CIF_PREFIX}{digits}"
        if cif not in real_cif_set and cif not in _used_cifs:
            _used_cifs.add(cif)
            return cif


# DATE HELPERS

injection_horizon_start = max_real_date + timedelta(days=INJECTION_START_BUFFER_DAYS)


def random_anomaly_anchor():
    """A random point in time, after the real data ends, used as the anchor
    for one synthetic client's whole timeline (baseline + anomaly)."""
    offset_days = random.randint(0, INJECTION_SPAN_DAYS)
    return injection_horizon_start + timedelta(days=offset_days)


def to_night_hour(dt):
    if random.random() < 0.5:
        hour = random.randint(20, 23)
    else:
        hour = random.randint(0, 8)
    return dt.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))


def to_day_hour(dt):
    hour = random.randint(9, 18)
    return dt.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))


# ============================================================
# ROW BUILDER
# ============================================================

def make_row(cif, trs_no, dt, currency, amount, gl, exchange_rate, injected,
             anomaly_type, to_gl="", to_cif=""):
    if currency == LOCAL_CURRENCY:
        cv_amount = amount
        exchange_rate = 1.0
    else:
        cv_amount = amount * exchange_rate

    return {
        "BRANCH_CODE": BRANCH_CODE,
        "TRS_NO": trs_no,
        "LONG_DESC_ENG": "INJECTED_ANOMALY" if anomaly_type != "BASELINE_NORMAL" else "INJECTED_BASELINE",
        "DATETIME": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "TRS_CY": currency,
        "TRS_AC_GL": gl,
        "TRS_AC_CIF": cif,
        "AMOUNT": round(amount, 2),
        "CV_AMOUNT": round(cv_amount, 2),
        "EXCHANGE_RATE": round(exchange_rate, 4),
        "TO_TRS_AC_GL": to_gl,
        "TO_TRS_AC_CIF": to_cif,
        "CV_AMOUNT_MISSING": 0,
        "EXCH_RATE_MISSING": 0,
        "EXCH_RATE_NEGATIVE_FLAG": 0,
        "INJECTED": injected,
        "ANOMALY_TYPE": anomaly_type,
    }


# ============================================================
# BASELINE GENERATOR
#
# Gives a fresh synthetic client a plausible "normal" history in local
# currency before the anomaly, so CLIENT_AVG_AMOUNT / RATIO_TO_AVG are
# computed against something real rather than nothing.
# Returns (rows, baseline_avg_cv_amount, last_baseline_dt)
# ============================================================

def generate_baseline(cif, anchor_dt):
    n = random.randint(*BASELINE_TXNS_PER_CLIENT)
    client_mean = random.uniform(3000, 20000)  # this client's "normal" amount, local currency

    rows = []
    dts = []
    for _ in range(n):
        days_before = random.randint(*BASELINE_WINDOW_DAYS)
        dt = to_day_hour(anchor_dt - timedelta(days=days_before))
        dts.append(dt)
        amount = client_mean * random.uniform(0.7, 1.3)
        gl = int(random.choice(real_gl_codes))
        rows.append(make_row(
            cif=cif, trs_no=next_trs_no(), dt=dt, currency=LOCAL_CURRENCY,
            amount=amount, gl=gl, exchange_rate=1.0, injected=1,
            anomaly_type="BASELINE_NORMAL",
        ))

    dts.sort()
    baseline_avg_cv = float(np.mean([r["CV_AMOUNT"] for r in rows]))
    return rows, baseline_avg_cv, dts[-1]


rows = []

# ============================================================
# 1. NIGHT_RAPID_BURST
# ============================================================
remaining = N_PER_TYPE["NIGHT_RAPID_BURST"]
while remaining > 0:
    cif = next_synthetic_cif()
    anchor = random_anomaly_anchor()
    baseline_rows, baseline_avg, last_baseline_dt = generate_baseline(cif, anchor)
    rows.extend(baseline_rows)

    burst_size = min(random.randint(5, 8), remaining)
    start_dt = to_night_hour(anchor)
    for i in range(burst_size):
        dt = start_dt + timedelta(minutes=random.randint(1, 3) * i)
        amount = baseline_avg * random.uniform(8, 15)
        gl = int(random.choice(real_gl_codes))
        rows.append(make_row(cif, next_trs_no(), dt, LOCAL_CURRENCY, amount, gl,
                              1.0, injected=1, anomaly_type="NIGHT_RAPID_BURST"))
    remaining -= burst_size

# ============================================================
# 2. NIGHT_LARGE_AMOUNT
# ============================================================
for _ in range(N_PER_TYPE["NIGHT_LARGE_AMOUNT"]):
    cif = next_synthetic_cif()
    anchor = random_anomaly_anchor()
    baseline_rows, baseline_avg, _ = generate_baseline(cif, anchor)
    rows.extend(baseline_rows)

    dt = to_night_hour(anchor)
    amount = baseline_avg * random.uniform(20, 50)
    gl = int(random.choice(real_gl_codes))
    rows.append(make_row(cif, next_trs_no(), dt, LOCAL_CURRENCY, amount, gl,
                          1.0, injected=1, anomaly_type="NIGHT_LARGE_AMOUNT"))

# ============================================================
# 3. SMURFING_RAPID_BURST
# ============================================================
remaining = N_PER_TYPE["SMURFING_RAPID_BURST"]
while remaining > 0:
    cif = next_synthetic_cif()
    anchor = random_anomaly_anchor()
    baseline_rows, baseline_avg, _ = generate_baseline(cif, anchor)
    rows.extend(baseline_rows)

    burst_size = min(random.randint(6, 10), remaining)
    start_dt = to_day_hour(anchor)
    for i in range(burst_size):
        dt = start_dt + timedelta(minutes=random.randint(1, 3) * i)
        amount = baseline_avg * random.uniform(0.8, 1.2)  # amount stays normal, count is the signal
        gl = int(random.choice(real_gl_codes))
        to_cif = next_synthetic_cif() if random.random() < 0.5 else cif
        rows.append(make_row(cif, next_trs_no(), dt, LOCAL_CURRENCY, amount, gl,
                              1.0, injected=1, anomaly_type="SMURFING_RAPID_BURST",
                              to_cif=to_cif))
    remaining -= burst_size

# ============================================================
# 4. UNUSUAL_SPENDING
# ============================================================
for _ in range(N_PER_TYPE["UNUSUAL_SPENDING"]):
    cif = next_synthetic_cif()
    anchor = random_anomaly_anchor()
    baseline_rows, baseline_avg, _ = generate_baseline(cif, anchor)
    rows.extend(baseline_rows)

    dt = to_day_hour(anchor)
    amount = baseline_avg * random.uniform(10, 30)
    gl = int(random.choice(real_gl_codes))
    rows.append(make_row(cif, next_trs_no(), dt, LOCAL_CURRENCY, amount, gl,
                          1.0, injected=1, anomaly_type="UNUSUAL_SPENDING"))

# ============================================================
# 5. FOREIGN_SPIKE  (daytime, foreign currency, amount spike vs local baseline)
# ============================================================
for _ in range(N_PER_TYPE["FOREIGN_SPIKE"]):
    cif = next_synthetic_cif()
    anchor = random_anomaly_anchor()
    baseline_rows, baseline_avg, _ = generate_baseline(cif, anchor)
    rows.extend(baseline_rows)

    dt = to_day_hour(anchor)
    currency = int(random.choice(foreign_codes))
    exchange_rate = random.uniform(100, 250)
    target_cv = baseline_avg * random.uniform(10, 30)  # spike measured in converted amount
    amount = target_cv / exchange_rate
    gl = int(random.choice(real_gl_codes))
    rows.append(make_row(cif, next_trs_no(), dt, currency, amount, gl,
                          exchange_rate, injected=1, anomaly_type="FOREIGN_SPIKE"))

# ============================================================
# 6. FOREIGN_NIGHT  (night, foreign currency, amount spike vs local baseline)
# ============================================================
for _ in range(N_PER_TYPE["FOREIGN_NIGHT"]):
    cif = next_synthetic_cif()
    anchor = random_anomaly_anchor()
    baseline_rows, baseline_avg, _ = generate_baseline(cif, anchor)
    rows.extend(baseline_rows)

    dt = to_night_hour(anchor)
    currency = int(random.choice(foreign_codes))
    exchange_rate = random.uniform(100, 250)
    target_cv = baseline_avg * random.uniform(10, 30)
    amount = target_cv / exchange_rate
    gl = int(random.choice(real_gl_codes))
    rows.append(make_row(cif, next_trs_no(), dt, currency, amount, gl,
                          exchange_rate, injected=1, anomaly_type="FOREIGN_NIGHT"))

# ============================================================
# ASSEMBLE
# ============================================================

injected_df = pd.DataFrame(rows)

n_anomalies = (injected_df["ANOMALY_TYPE"] != "BASELINE_NORMAL").sum()
assert n_anomalies == 325, f"Expected 325 anomaly rows, got {n_anomalies}"

column_order = [
    "BRANCH_CODE", "TRS_NO", "LONG_DESC_ENG", "DATETIME", "TRS_CY", "TRS_AC_GL",
    "TRS_AC_CIF", "AMOUNT", "CV_AMOUNT", "EXCHANGE_RATE", "TO_TRS_AC_GL",
    "TO_TRS_AC_CIF", "CV_AMOUNT_MISSING", "EXCH_RATE_MISSING",
    "EXCH_RATE_NEGATIVE_FLAG", "INJECTED", "ANOMALY_TYPE",
]
injected_df = injected_df[column_order]

# Sort each synthetic client's own rows chronologically (baseline before anomaly)
injected_df = injected_df.sort_values(["TRS_AC_CIF", "DATETIME"]).reset_index(drop=True)

final_df = pd.concat([df[column_order], injected_df], ignore_index=True)
final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# ============================================================
# VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("INJECTION COMPLETE")
print("=" * 70)
print(f"Original rows        : {len(df):,}")
print(f"Injected rows (total): {len(injected_df):,}  (325 anomalies + baseline support rows)")
print(f"Final rows           : {len(final_df):,}")

print("\nAnomaly distribution:")
print(injected_df.loc[injected_df["ANOMALY_TYPE"] != "BASELINE_NORMAL", "ANOMALY_TYPE"].value_counts())
print("Baseline support rows:", (injected_df["ANOMALY_TYPE"] == "BASELINE_NORMAL").sum())

print("\nChecks:")
print("TRS_NO starts with 9999:", injected_df["TRS_NO"].astype(str).str.startswith("9999").all())
print("TRS_AC_CIF starts with 7:", injected_df["TRS_AC_CIF"].astype(str).str.startswith("7").all())
print("No CIF collides with a real client:", len(set(injected_df["TRS_AC_CIF"]) & real_cif_set) == 0)
print("BRANCH_CODE = 10000:", (injected_df["BRANCH_CODE"] == 10000).all())
print("All injected dates after real max date:",
      (pd.to_datetime(injected_df["DATETIME"]) > max_real_date).all())
print("INJECTED flag correct on original rows:", (final_df.iloc[:len(df)]["INJECTED"] == 0).all())

print("\nSaved to:", OUTPUT_CSV)
print("=" * 70)