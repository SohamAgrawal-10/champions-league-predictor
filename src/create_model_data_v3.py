import pandas as pd


# --------------------------------------------------
# LOAD ELO FEATURES
# --------------------------------------------------

df = pd.read_csv("data/processed/elo_features.csv")

df["MatchDate"] = pd.to_datetime(df["MatchDate"])


# --------------------------------------------------
# REMOVE MISSING ELO
# --------------------------------------------------

df = df.dropna(
    subset=["OurHomeElo", "OurAwayElo"]
).copy()


# --------------------------------------------------
# SELECT FEATURES
# --------------------------------------------------

features = [
    "MatchDate",
    "Division",
    "HomeTeam",
    "AwayTeam",

    # OUR ELO
    "OurHomeElo",
    "OurAwayElo",
    "OurEloDiff",

    # ROLLING FORM
    "HomeForm5Points",
    "AwayForm5Points",
    "Form5PointsDiff",

    "HomeForm5GoalsScored",
    "AwayForm5GoalsScored",
    "Form5GoalsScoredDiff",

    "HomeForm5GoalsConceded",
    "AwayForm5GoalsConceded",
    "Form5GoalsConcededDiff",

    "HomeForm5GoalDifference",
    "AwayForm5GoalDifference",
    "Form5GoalDifferenceDiff",

    "HomeForm5WinRate",
    "AwayForm5WinRate",
    "Form5WinRateDiff",

    # TARGET
    "FTResult",
]


model_df = df[features].copy()

model_df = (
    model_df
    .sort_values("MatchDate")
    .reset_index(drop=True)
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_path = "data/processed/model_data_v3.csv"

model_df.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("=== MODEL DATA V3 ===")

print(f"Rows: {len(model_df):,}")
print(f"Columns: {len(model_df.columns)}")

print("\nFeatures:")

for column in model_df.columns:
    print(column)

print("\nDate range:")

print(
    f"{model_df['MatchDate'].min().date()} "
    f"→ "
    f"{model_df['MatchDate'].max().date()}"
)

print("\nResult distribution:")

print(
    model_df["FTResult"]
    .value_counts(normalize=True)
    .round(3)
)

print("\nSaved to:")
print(output_path)