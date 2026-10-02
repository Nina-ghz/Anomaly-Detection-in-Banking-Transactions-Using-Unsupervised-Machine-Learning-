import pandas as pd

INPUT_FILE = "injected_predictions_3models.csv"

df = pd.read_csv(INPUT_FILE)


# --- Score d'ensemble : nombre de modeles qui flaguent la transaction ---
df['N_MODELS_FLAGGED'] = df[['IF_ANOMALY', 'LOF_ANOMALY', 'AE_ANOMALY']].sum(axis=1)

# Flag final : anomalie si au moins 2 modeles sur 3 sont d'accord
df['ENSEMBLE_ANOMALY'] = (df['N_MODELS_FLAGGED'] >= 2).astype(int)

# --- Overlap entre les modeles ---
if_set = set(df.index[df['IF_ANOMALY'] == 1])
lof_set = set(df.index[df['LOF_ANOMALY'] == 1])
ae_set = set(df.index[df['AE_ANOMALY'] == 1])

print(f"Total transactions: {len(df)}")
print(f"IF only flagged:  {len(if_set)}")
print(f"LOF only flagged: {len(lof_set)}")
print(f"AE only flagged:  {len(ae_set)}")
print()
print(f"IF & LOF overlap:      {len(if_set & lof_set)}")
print(f"IF & AE overlap:       {len(if_set & ae_set)}")
print(f"LOF & AE overlap:      {len(lof_set & ae_set)}")
print(f"All 3 agree:           {len(if_set & lof_set & ae_set)}")
print()
print(f"Ensemble (>=2 models): {df['ENSEMBLE_ANOMALY'].sum()}")

df.to_csv("injected_predictions_with_ensemble.csv", index=False, encoding="utf-8")
print("\nSaved: injected_predictions_with_ensemble.csv")