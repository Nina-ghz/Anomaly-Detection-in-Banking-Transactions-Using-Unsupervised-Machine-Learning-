import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import joblib

FEATURES_FILE = "../TEST7/trs_features_final.csv"
IDS_FILE = "../TEST7/trs_ids_final.csv"
 
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
    'MEDIAN_RATIO',   # <-- new
    'UNUSUAL_SPENDING_FLAG',  
]

df = pd.read_csv(FEATURES_FILE)
ids = pd.read_csv(IDS_FILE)
X_raw = df[model_feature_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

 
N_NEIGHBORS = 20         
CONTAMINATION = 0.01 
 
lof = LocalOutlierFactor(
    n_neighbors=N_NEIGHBORS,
    contamination=CONTAMINATION,
    novelty=True,
    n_jobs=-1,
)
 
lof.fit(X_scaled)
df['LOF_SCORE'] = -lof.decision_function(X_scaled)

LOF_THRESHOLD = np.percentile(df['LOF_SCORE'], 97)

df['LOF_ANOMALY'] = (
    df['LOF_SCORE'] >= LOF_THRESHOLD
).astype(int)

output = pd.concat(
    [ids[['BRANCH_CODE', 'TRS_NO', 'DATETIME', 'TRS_AC_CIF']], df[['LOF_SCORE', 'LOF_ANOMALY']]],
    axis=1,
)
output.to_csv("lof_results.csv", index=False, encoding="utf-8")

joblib.dump(lof, "lof_model.joblib")
joblib.dump(scaler, "lof_scaler.joblib")
joblib.dump(LOF_THRESHOLD, "lof_threshold.joblib")

print(
    f"LOF fit on {len(X_raw)} transactions, "
    f"{len(model_feature_cols)} features, "
    f"n_neighbors={N_NEIGHBORS}."
)

print(f"p97 LOF threshold: {LOF_THRESHOLD:.6f}")

print(
    f"Flagged {df['LOF_ANOMALY'].sum()} anomalies "
    f"({df['LOF_ANOMALY'].mean() * 100:.2f}%)."
)