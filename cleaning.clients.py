import pandas as pd

client = pd.read_csv("MERGED_CLIENTS.csv", encoding="latin-1")

client['DATE_CREATED'] = pd.to_datetime(
    client['DATE_CREATED'],
    format='%d-%b-%y' ,
     errors='coerce')

client['BIRTH_DATE'] = pd.to_datetime(
    client['BIRTH_DATE'], 
    format='%d-%b-%y',
      errors='coerce')

placeholder = pd.to_datetime('01-JAN-00', format='%d-%b-%y')
client.loc[client['BIRTH_DATE'] == placeholder, 'BIRTH_DATE'] = pd.NaT

today = pd.Timestamp.today()
future_mask = client['BIRTH_DATE'] > today
client.loc[future_mask, 'BIRTH_DATE'] = client.loc[future_mask, 'BIRTH_DATE'] - pd.DateOffset(years=100)


salary_cap = client['MONTHLY_SALARY'].quantile(0.99)
 
client['SALARY_MISSING'] = client['MONTHLY_SALARY'].isna().astype(int)
client['SALARY_OUTLIER_FLAG'] = (
    (client['MONTHLY_SALARY'] > salary_cap) & client['MONTHLY_SALARY'].notna()
).astype(int)

too_old_mask = client['BIRTH_DATE'] < pd.Timestamp.today() - pd.DateOffset(years=100)
print("suspiciously old birth dates:", too_old_mask.sum())

client = client.drop_duplicates(subset=['CIF_NO'])
 
client.to_csv("clients_cleaned_Final.csv", index=False, encoding="utf-8")

print("Clients cleaned shape:", client.shape)