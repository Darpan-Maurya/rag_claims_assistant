def approval_percentage(df):
    total = len(df)
    approved = len(df[df["claim_status"] == "APPROVED"])
    return round((approved / total) * 100, 2)
