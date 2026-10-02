import pandas as pd 

accounts = pd.read_csv("Sirine_Accounts.csv", encoding="latin-1")

accounts= accounts.drop(columns=["ECO_SECTOR"])

accounts['LAST_TRANS_DATE'] = pd.to_datetime(
    accounts['LAST_TRANS_DATE'],
    format='%d-%b-%y', 
   errors= 'coerce'
     )

accounts['DATE_OPND'] = pd.to_datetime(
    accounts['DATE_OPND'],
        format='%d-%b-%y',  
        errors= 'coerce'
          )

accounts['DATE_CLOSD'] = pd.to_datetime(
    accounts['DATE_CLOSD'],
     format='%d-%b-%y',
      errors= 'coerce'  
        )

accounts['CV_BAL_MISSING'] = accounts['CV_AVAIL_BAL'].isna().astype(int)
accounts['FC_BAL_MISSING'] = accounts['FC_AVAIL_BAL'].isna().astype(int)

accounts['CV_AVAIL_BAL'] = accounts['CV_AVAIL_BAL'].fillna(0)
accounts['FC_AVAIL_BAL'] = accounts['FC_AVAIL_BAL'].fillna(0)

accounts['IS_OPEN'] = accounts['DATE_CLOSD'].isna().astype(int)

accounts['STATUS_CLOSED_NO_DATE_FLAG'] = ((accounts['STATUS'] == 'C') & accounts['DATE_CLOSD'].isna()).astype(int)

accounts = accounts.drop_duplicates()

assert accounts['ACCOUNT_NUMBER'].is_unique, "ACCOUNT_NUMBER should be unique per account"

accounts.to_csv("accounts_cleaned_Final.csv", index=False, encoding="utf-8")

print("AMF cleaned shape:", accounts.shape)
