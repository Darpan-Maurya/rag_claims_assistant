import pandas as pd
df = pd.read_parquet("data/processed/claims_processed.parquet")
print(df[["claim_id", "claim_status", "claim_text"]].head())
