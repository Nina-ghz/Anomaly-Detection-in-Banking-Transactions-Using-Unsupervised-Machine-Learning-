import pandas as pd
import re

INPUT_FILE = "../clean/ledger_cleaned.csv"
OUTPUT_ALL = "ledger_categorized.csv"
OUTPUT_CLIENT = "gl_client.csv"
OUTPUT_AUTRE = "gl_autre.csv"

# Mots-clés à adapter selon vos besoins
CLIENT_KEYWORDS = [
    "client",       # couvre client, clients, clientèle, clientèles (via regex)
]

# Construit un pattern regex insensible à la casse et aux accents simples
pattern = re.compile("|".join(CLIENT_KEYWORDS), flags=re.IGNORECASE)

def categorize(description: str) -> str:
    if pd.isna(description):
        return "Autre"
    return "Client" if pattern.search(str(description)) else "Autre"

def main():
    df = pd.read_csv(INPUT_FILE, dtype={"GL_CODE": str})

    df["CATEGORY"] = df["LONG_DESC_ENG"].apply(categorize)

    # Sauvegarde du fichier complet avec la colonne CATEGORY
    df.to_csv(OUTPUT_ALL, index=False)

    # Sauvegarde des deux fichiers séparés
    df[df["CATEGORY"] == "Client"].to_csv(OUTPUT_CLIENT, index=False)
    df[df["CATEGORY"] == "Autre"].to_csv(OUTPUT_AUTRE, index=False)

    print(f"Total lignes        : {len(df)}")
    print(f"Codes 'Client'      : {(df['CATEGORY'] == 'Client').sum()}")
    print(f"Codes 'Autre'       : {(df['CATEGORY'] == 'Autre').sum()}")
    print(f"\nFichiers générés :")
    print(f" - {OUTPUT_ALL}")
    print(f" - {OUTPUT_CLIENT}")
    print(f" - {OUTPUT_AUTRE}")

if __name__ == "__main__":
    main()