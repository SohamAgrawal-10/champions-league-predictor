import pandas as pd

# Load raw data
matches = pd.read_csv(
    "data/raw/Matches.csv",
    low_memory=False
)

# Keep only the Big Five European leagues
top_leagues = ["E0", "SP1", "D1", "I1", "F1"]

df = matches[matches["Division"].isin(top_leagues)].copy()

# Keep only information that is available BEFORE the match
# Match statistics such as shots/corners/cards are deliberately excluded
# because they are only known after the match.
columns = [
    "Division",
    "MatchDate",
    "HomeTeam",
    "AwayTeam",
    "HomeElo",
    "AwayElo",
    "Form3Home",
    "Form5Home",
    "Form3Away",
    "Form5Away",
    "OddHome",
    "OddDraw",
    "OddAway",
    "FTHome",
    "FTAway",
    "FTResult"
]

df = df[columns]

# Remove matches without a result
df = df.dropna(subset=["FTResult"])

# Convert date to datetime
df["MatchDate"] = pd.to_datetime(df["MatchDate"])

# Sort chronologically
df = df.sort_values("MatchDate").reset_index(drop=True)

# Save processed dataset
output_path = "data/processed/big5_matches.csv"
df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Date range: {df['MatchDate'].min().date()} → {df['MatchDate'].max().date()}")

print("\nLeague distribution:")
print(df["Division"].value_counts())

print("\nSaved to:")
print(output_path)