import pandas as pd

cif_type = pd.read_csv("Cif_Type.csv", encoding="cp1252")

gen_ledger = pd.read_csv("Stage_Gen_Ledger.csv", encoding="cp1252")

cif_type['LONG_DESC_ENG'] = cif_type['LONG_DESC_ENG'].str.strip()

cif_type = cif_type.drop_duplicates(subset=['TYPE_CODE'])

cif_type.to_csv("cif_type_cleaned.csv", index=False, encoding="utf-8")


print("Cif_Type cleaned shape:", cif_type.shape)


gen_ledger = pd.read_csv("Stage_Gen_Ledger.csv", encoding="cp1252")

gen_ledger['LONG_DESC_ENG'] = gen_ledger['LONG_DESC_ENG'].str.strip()

gen_ledger = gen_ledger.drop_duplicates(subset=['GL_CODE'])


gen_ledger.to_csv("stage_gen_ledger_cleaned.csv", index=False, encoding="utf-8")
print("Stage_Gen_Ledger cleaned shape:", gen_ledger.shape)