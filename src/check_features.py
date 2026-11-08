import pandas as pd

# Load processed dataset
df = pd.read_csv("data/processed/big5_matches.csv")

# Remove matches without Elo ratings
df = df.dropna(subset=["HomeElo", "AwayElo"]).copy()

# Convert date
df["MatchDate"] = pd.to_datetime(df["MatchDate"])

# Relative strength
df["EloDiff"] = df["HomeElo"] - df["AwayElo"]

# Recent form differences
df["Form3Diff"] = df["Form3Home"] - df["Form3Away"]
df["Form5Diff"] = df["Form5Home"] - df["Form5Away"]

# Features for the model
features = [
    "MatchDate",
    "Division",
    "HomeTeam",
    "AwayTeam",

    # Absolute team strength
    "HomeElo",
    "AwayElo",

    # Relative team strength
    "EloDiff",

    # Recent form
    "Form3Home",
    "Form3Away",
    "Form3Diff",
    "Form5Home",
    "Form5Away",
    "Form5Diff",

    # Target
    "FTResult"
]

model_df = df[features].copy()

# Sort chronologically
model_df = model_df.sort_values("MatchDate").reset_index(drop=True)

# Save
output_path = "data/processed/model_data.csv"
model_df.to_csv(output_path, index=False)

print("Feature dataset created successfully!")
print(f"Rows: {len(model_df):,}")
print(f"Columns: {len(model_df.columns)}")

print("\nFeatures:")
print(model_df.columns.tolist())

print("\nFirst 5 rows:")
print(model_df.head())

print("\nResult distribution:")
print(model_df["FTResult"].value_counts())