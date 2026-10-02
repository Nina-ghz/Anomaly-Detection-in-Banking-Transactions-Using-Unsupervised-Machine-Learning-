import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

FEATURES_FILE = "../TEST7/trs_features_final.csv"
IDS_FILE = "../TEST7/trs_ids_final.csv"

df = pd.read_csv(FEATURES_FILE)
ids = pd.read_csv(IDS_FILE)

model_feature_cols = [
    'AMOUNT',
    'CLIENT_AVG_AMOUNT',
    'IS_NIGHT',
    'TRS_COUNT_24H',
    'TRS_COUNT_7D',
    'IS_FOREIGN_CURRENCY',
    'RATIO',
    'AMOUNT_ZSCORE',
    'COUNT_RATIO',
    'AMOUNT_SUM_24H',
    'MEDIAN_RATIO',
    'UNUSUAL_SPENDING_FLAG',
]

X = df[model_feature_cols].copy()

CONTAMINATION = 0.01
N_ESTIMATORS = 200
RANDOM_STATE = 42

iso_forest = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

iso_forest.fit(X)

df['IF_SCORE'] = iso_forest.decision_function(X)

# Isolation Forest: LOWER score = MORE anomalous.
# We flag the bottom 5% of raw scores as anomalies (this is what was
# labeled "p95" during threshold tuning: bottom-5% cut, chosen because
# it dominated stricter cuts on NIGHT_RAPID_BURST/UNUSUAL_SPENDING
# recall without losing the categories already at 100%).
IF_THRESHOLD = np.percentile(df['IF_SCORE'], 5)

df['IF_ANOMALY'] = (
    df['IF_SCORE'] <= IF_THRESHOLD
).astype(int)

output = pd.concat(
    [
        ids[['BRANCH_CODE', 'TRS_NO', 'DATETIME', 'TRS_AC_CIF']],
        df[['IF_SCORE', 'IF_ANOMALY']]
    ],
    axis=1,
)

output.to_csv(
    "iso_results.csv",
    index=False,
    encoding="utf-8"
)

joblib.dump(
    iso_forest,
    "isolation_forest_model.joblib"
)

joblib.dump(
    IF_THRESHOLD,
    "isolation_forest_threshold.joblib"
)

print(
    f"Isolation Forest trained on {len(X)} transactions, "
    f"{len(model_feature_cols)} features."
)

print(f"Bottom-5% threshold (raw IF_SCORE p5): {IF_THRESHOLD:.6f}")

print(
    f"Flagged {df['IF_ANOMALY'].sum()} anomalies "
    f"({df['IF_ANOMALY'].mean() * 100:.2f}%)."
)
