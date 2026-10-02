import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

INJECTED_FEATURES_FILE = "../feature.af.inject/trs_features_inject_7.csv"
INJECTED_IDS_FILE = "../feature.af.inject/trs_ids_inject_7.csv"

ISO_MODEL_FILE = "../training-py8/isolation_forest_model.joblib"

LOF_MODEL_FILE = "../training-py8/lof_model.joblib"
LOF_SCALER_FILE = "../training-py8/lof_scaler.joblib"

AE_MODEL_FILE = "../training-py8/autoencoder_model.pt"
AE_SCALER_FILE = "../training-py8/autoencoder_scaler.joblib"


# =========================================================
# MODEL FEATURES
# =========================================================

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


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INJECTED_FEATURES_FILE)
ids = pd.read_csv(INJECTED_IDS_FILE)

X_raw = df[model_feature_cols].copy()


# =========================================================
# 1) ISOLATION FOREST — p99
# =========================================================

iso_forest = joblib.load(ISO_MODEL_FILE)

# Lower score = more anomalous
df['IF_SCORE'] = iso_forest.decision_function(X_raw)

# p99 threshold from original clean training data
IF_THRESHOLD =  0.104101

df['IF_ANOMALY'] = (
    df['IF_SCORE'] <= IF_THRESHOLD
).astype(int)


# =========================================================
# 2) LOF — p97
# =========================================================

lof = joblib.load(LOF_MODEL_FILE)
lof_scaler = joblib.load(LOF_SCALER_FILE)

# Apply the SAME scaler used during LOF training
X_lof_scaled = lof_scaler.transform(X_raw)

# Higher score = more anomalous
df['LOF_SCORE'] = -lof.decision_function(X_lof_scaled)

# p97 threshold from original clean training data
LOF_THRESHOLD = -1.432020

df['LOF_ANOMALY'] = (
    df['LOF_SCORE'] >= LOF_THRESHOLD
).astype(int)


# =========================================================
# 3) AUTOENCODER — p95
# =========================================================

class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Load the SAME scaler used during AE training
ae_scaler = joblib.load(AE_SCALER_FILE)

X_ae_scaled = ae_scaler.transform(X_raw).astype(np.float32)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

n_features = X_ae_scaled.shape[1]

# Create model with the same architecture
ae_model = Autoencoder(n_features).to(device)

# Load trained weights
ae_model.load_state_dict(
    torch.load(
        AE_MODEL_FILE,
        map_location=device
    )
)

ae_model.eval()

X_tensor = torch.from_numpy(X_ae_scaled).to(device)

with torch.no_grad():
    X_reconstructed = ae_model(X_tensor).cpu().numpy()


# Reconstruction error
reconstruction_error = np.mean(
    (X_ae_scaled - X_reconstructed) ** 2,
    axis=1
)

df['AE_SCORE'] = reconstruction_error

# p95 threshold from original clean training data
AE_THRESHOLD = 0.9246

df['AE_ANOMALY'] = (
    df['AE_SCORE'] >= AE_THRESHOLD
).astype(int)


# =========================================================
# EXPORT RESULTS
# =========================================================

result_cols = [
    'IF_SCORE',
    'IF_ANOMALY',
    'LOF_SCORE',
    'LOF_ANOMALY',
    'AE_SCORE',
    'AE_ANOMALY'
]

output = pd.concat(
    [
        ids[
            [
                'BRANCH_CODE',
                'TRS_NO',
                'DATETIME',
                'TRS_AC_CIF'
            ]
        ],
        df[result_cols]
    ],
    axis=1
)

output.to_csv(
    "injected_predictions_3models.csv",
    index=False,
    encoding="utf-8"
)


# =========================================================
# SUMMARY
# =========================================================

print("=" * 70)
print("3-MODEL PREDICTION")
print("=" * 70)

print(f"Tested transactions : {len(X_raw)}")
print(f"Number of features  : {len(model_feature_cols)}")

print()
print("THRESHOLDS")
print(f"Isolation Forest p99 : {IF_THRESHOLD:.6f}")
print(f"LOF p97              : {LOF_THRESHOLD:.6f}")
print(f"Autoencoder p95      : {AE_THRESHOLD:.6f}")

print()
print("ANOMALIES FLAGGED")
print(
    f"Isolation Forest : {df['IF_ANOMALY'].sum()} "
    f"({df['IF_ANOMALY'].mean() * 100:.2f}%)"
)

print(
    f"LOF              : {df['LOF_ANOMALY'].sum()} "
    f"({df['LOF_ANOMALY'].mean() * 100:.2f}%)"
)

print(
    f"Autoencoder      : {df['AE_ANOMALY'].sum()} "
    f"({df['AE_ANOMALY'].mean() * 100:.2f}%)"
)


print()
print("Output: injected_predictions_3models.csv")
print("=" * 70)