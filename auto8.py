import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
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
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
n_features = X_scaled.shape[1]
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.from_numpy(X_scaled).to(device)

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
 
 
model = Autoencoder(n_features).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


n_rows = X_tensor.shape[0]
n_val = int(n_rows * 0.1)
perm = torch.randperm(n_rows)
val_idx, train_idx = perm[:n_val], perm[n_val:]
 
X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
 
EPOCHS = 100
BATCH_SIZE = 256
PATIENCE = 5
 
best_val_loss = float('inf')
epochs_no_improve = 0
best_state = None
 
for epoch in range(EPOCHS):
    model.train()
    shuffle = torch.randperm(X_train.shape[0])
    for start in range(0, X_train.shape[0], BATCH_SIZE):
        batch_idx = shuffle[start:start + BATCH_SIZE]
        batch = X_train[batch_idx]
 
        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
 
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), X_val).item()
 
    print(f"Epoch {epoch + 1}/{EPOCHS} — val_loss: {val_loss:.6f}")
 
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1} (best val_loss: {best_val_loss:.6f})")
            break
 
if best_state is not None:
    model.load_state_dict(best_state)


 
model.eval()
with torch.no_grad():
    X_reconstructed = model(X_tensor).cpu().numpy()
 
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
df['AE_SCORE'] = reconstruction_error
 
THRESHOLD_PERCENTILE = 95
threshold = np.percentile(reconstruction_error, THRESHOLD_PERCENTILE)
df['AE_ANOMALY'] = (df['AE_SCORE'] >= threshold).astype(int)

output = pd.concat(
    [ids[['BRANCH_CODE', 'TRS_NO', 'DATETIME', 'TRS_AC_CIF']], df[['AE_SCORE', 'AE_ANOMALY']]],
    axis=1,
)
output.to_csv("auto_results.csv", index=False, encoding="utf-8")
 
torch.save(model.state_dict(), "autoencoder_model.pt")
joblib.dump(scaler, "autoencoder_scaler.joblib")
 
print(f"Autoencoder trained on {len(X_raw)} transactions, {n_features} features.")
print(f"Reconstruction error threshold (p{THRESHOLD_PERCENTILE}): {threshold:.4f}")
print(f"Flagged {df['AE_ANOMALY'].sum()} anomalies ({df['AE_ANOMALY'].mean() * 100:.2f}%).")