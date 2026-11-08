import pandas as pd

df = pd.read_csv("data/raw/Matches.csv", low_memory=False)

top_leagues = ["E0", "SP1", "D1", "I1", "F1"]

df = df[df["Division"].isin(top_leagues)].copy()

df["MatchDate"] = pd.to_datetime(df["MatchDate"])

# Season definition:
# August–December -> current year
# January–July -> previous year
df["Season"] = df["MatchDate"].dt.year
df.loc[df["MatchDate"].dt.month < 8, "Season"] -= 1

df["HomeGoals"] = pd.to_numeric(df["FTHome"], errors="coerce")
df["AwayGoals"] = pd.to_numeric(df["FTAway"], errors="coerce")

# Only use completed seasons.
# 2026 is currently an incomplete season.
df = df[df["Season"] <= 2025]

league_strength = (
    df.groupby(["Season", "Division"])
    .agg(
        Matches=("Division", "count"),
        HomeGoals=("HomeGoals", "mean"),
        AwayGoals=("AwayGoals", "mean")
    )
    .reset_index()
)

league_strength["TotalGoals"] = (
    league_strength["HomeGoals"] +
    league_strength["AwayGoals"]
)

# Use the most recent 3 completed seasons.
recent = league_strength[
    league_strength["Season"].between(2023, 2025)
].copy()

summary = (
    recent.groupby("Division")
    .agg(
        AvgHomeGoals=("HomeGoals", "mean"),
        AvgAwayGoals=("AwayGoals", "mean"),
        AvgTotalGoals=("TotalGoals", "mean")
    )
    .reset_index()
)

print("=== RECENT LEAGUE STRENGTH ===")
print(summary.round(3).to_string(index=False))

output_path = "data/processed/league_strength.csv"
summary.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")