import pandas as pd

trs = pd.read_csv("Sirine_Transactions.csv", encoding="latin-1", low_memory=False)


trs = trs.drop(columns=['TRS_AC_BR', 'ACC_ADDITIONAL_REFERENCE', 'TO_ACC_ADD_REF'])
for col in ['BRANCH_CODE', 'TRS_CY', 'TRS_AC_GL', 'TO_TRS_AC_GL']:
    trs[col] = trs[col].astype('category')

trs = trs.rename(columns={
    "TO_CHAR(SERVER_APPROVED_DATE,'DD-MM-YYYYHH24:MI:SS')": "DATETIME"
})
trs['DATETIME'] = pd.to_datetime(
    trs['DATETIME'],
    format='%d-%m-%Y %H:%M:%S',
    errors='coerce'
)

trs['LONG_DESC_ENG'] = trs['LONG_DESC_ENG'].astype(str).str.strip()

# --- Missing value flags (before touching the values) ---
trs['CV_AMOUNT_MISSING'] = trs['CV_AMOUNT'].isna().astype(int)
trs['EXCH_RATE_MISSING'] = trs['EXCHANGE_RATE'].isna().astype(int)

# --- Fill: missing CV_AMOUNT/EXCHANGE_RATE only happens on DZD (TRS_CY == 12) txns,
#     where CV_AMOUNT should equal AMOUNT and EXCHANGE_RATE should be 1 ---
trs.loc[trs['CV_AMOUNT'].isna(), 'CV_AMOUNT'] = trs.loc[trs['CV_AMOUNT'].isna(), 'AMOUNT']
trs['EXCHANGE_RATE'] = trs['EXCHANGE_RATE'].fillna(1)

# --- EXCHANGE_RATE can't be negative -> flag and correct ---
trs['EXCH_RATE_NEGATIVE_FLAG'] = (trs['EXCHANGE_RATE'] < 0).astype(int)
trs.loc[trs['EXCHANGE_RATE'] < 0, 'EXCHANGE_RATE'] = 0

print("rows before dedup:", len(trs))
trs = trs.drop_duplicates()
print("rows after dedup:", len(trs))

trs.to_csv("transactions_cleaned_Final.csv", index=False, encoding="utf-8")

print("Transactions cleaned shape:", trs.shape)