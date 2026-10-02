import pandas as pd
import numpy as np

trs = pd.read_csv("../injections/transactions_client_final_injected325.csv", encoding="utf-8-sig", low_memory=False)
print(trs.columns.tolist())

trs['DATETIME'] = pd.to_datetime(trs['DATETIME'], errors='coerce')

trs = trs.sort_values(['TRS_AC_CIF', 'DATETIME']).reset_index(drop=True)

trs['TRS_HOUR'] = trs['DATETIME'].dt.hour
trs['IS_NIGHT'] = (
    (trs['TRS_HOUR'] >= 20) | 
    (trs['TRS_HOUR'] < 8)
).astype('Int64')

LOCAL_CURRENCY_CODE = 12
trs['CV_AMOUNT'] = np.where(
    trs['TRS_CY'] == LOCAL_CURRENCY_CODE,
    trs['AMOUNT'],
   trs['AMOUNT'] * trs['EXCHANGE_RATE']
)
grp = trs.groupby('TRS_AC_CIF')['CV_AMOUNT']

trs['CLIENT_AVG_AMOUNT'] = grp.transform(lambda s: s.expanding().mean().shift(1))
 
trs['CLIENT_AVG_AMOUNT'] = trs['CLIENT_AVG_AMOUNT'].fillna(trs['CV_AMOUNT'])
 
trs['_CLIENT_STD_AMOUNT'] = grp.transform(lambda s: s.expanding().std().shift(1)).fillna(0)
 
BASELINE_FLOOR = 1.0   # avoids a near-zero average blowing up the ratio
RATIO_CAP = 100.0      # avoids one extreme outlier dominating the feature's scale
 
client_avg_floored = trs['CLIENT_AVG_AMOUNT'].clip(lower=BASELINE_FLOOR)
trs['RATIO'] = (trs['CV_AMOUNT'] / client_avg_floored).clip(upper=RATIO_CAP)

# --- New feature: AMOUNT_ZSCORE ---
ZSCORE_STD_FLOOR = 1.0
ZSCORE_CAP = 10.0

client_std_floored = trs['_CLIENT_STD_AMOUNT'].clip(lower=ZSCORE_STD_FLOOR)
trs['AMOUNT_ZSCORE'] = (
    (trs['CV_AMOUNT'] - trs['CLIENT_AVG_AMOUNT']) / client_std_floored
).clip(lower=-ZSCORE_CAP, upper=ZSCORE_CAP)

trs['_ONE'] = 1.0
 
trs['TRS_COUNT_24H'] = (
    trs.groupby('TRS_AC_CIF')
       .rolling('24h', on='DATETIME')['_ONE']
       .sum()
       .values
)
trs['TRS_COUNT_7D'] = (
    trs.groupby('TRS_AC_CIF')
       .rolling('7D', on='DATETIME')['_ONE']
       .sum()
       .values
)
trs.drop(columns=['_ONE'], inplace=True)

# --- New feature: COUNT_RATIO ---
COUNT_BASELINE_FLOOR = 1.0
COUNT_RATIO_CAP = 20.0

grp_count = trs.groupby('TRS_AC_CIF')['TRS_COUNT_24H']
trs['CLIENT_AVG_COUNT_24H'] = grp_count.transform(
    lambda s: s.expanding().mean().shift(1)
)
trs['CLIENT_AVG_COUNT_24H'] = trs['CLIENT_AVG_COUNT_24H'].fillna(trs['TRS_COUNT_24H'])

client_count_floored = trs['CLIENT_AVG_COUNT_24H'].clip(lower=COUNT_BASELINE_FLOOR)
trs['COUNT_RATIO'] = (
    trs['TRS_COUNT_24H'] / client_count_floored
).clip(upper=COUNT_RATIO_CAP)

# --- New feature: MEDIAN_RATIO ---
# Compare the current amount with the client's historical median amount.
# shift(1) ensures only previous transactions are used.

trs['CLIENT_MEDIAN_AMOUNT'] = grp.transform(
    lambda s: s.expanding().median().shift(1)
)

trs['CLIENT_MEDIAN_AMOUNT'] = trs['CLIENT_MEDIAN_AMOUNT'].fillna(
    trs['CV_AMOUNT']
)

MEDIAN_BASELINE_FLOOR = 1.0
MEDIAN_RATIO_CAP = 100.0

client_median_floored = trs['CLIENT_MEDIAN_AMOUNT'].clip(
    lower=MEDIAN_BASELINE_FLOOR
)

trs['MEDIAN_RATIO'] = (
    trs['CV_AMOUNT'] / client_median_floored
).clip(upper=MEDIAN_RATIO_CAP)

trs['UNUSUAL_SPENDING_FLAG'] = (
    (trs['RATIO'] >= 10) | (trs['MEDIAN_RATIO'] >= 10)
).astype(int)

# --- New feature: AMOUNT_SUM_24H ---
trs['AMOUNT_SUM_24H'] = (
    trs.groupby('TRS_AC_CIF')
       .rolling('24h', on='DATETIME')['CV_AMOUNT']
       .sum()
       .values
)

trs['IS_FOREIGN_CURRENCY'] = (trs['TRS_CY'] != LOCAL_CURRENCY_CODE).astype('Int64')

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

reference_cols = ['TRS_NO', 'TRS_AC_CIF']  # not fed to the model
feature_export_cols = reference_cols + model_feature_cols

id_cols = ['BRANCH_CODE', 'TRS_NO', 'DATETIME', 'TRS_AC_CIF']

final = trs.copy()
final[model_feature_cols] = final[model_feature_cols].astype('float64')

final[id_cols].to_csv("trs_ids_inject_7.csv", index=False, encoding="utf-8")
final[feature_export_cols].to_csv("trs_features_inject_7.csv", index=False, encoding="utf-8")