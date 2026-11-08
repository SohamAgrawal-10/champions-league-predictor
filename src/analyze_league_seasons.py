import pandas as pd

df = pd.read_csv("data/raw/Matches.csv", low_memory=False)

top_leagues = ["E0", "SP1", "D1", "I1", "F1"]

df = df[df["Division"].isin(top_leagues)].copy()

df["MatchDate"] = pd.to_datetime(df["MatchDate"])

df["Season"] = df["MatchDate"].dt.year

# Approximate football season:
# August–December belongs to that year's season.
# January–July belongs to the previous year's season.
df.loc[df["MatchDate"].dt.month < 8, "Season"] -= 1

df["HomeGoals"] = pd.to_numeric(df["FTHome"], errors="coerce")
df["AwayGoals"] = pd.to_numeric(df["FTAway"], errors="coerce")

season_table = (
    df.groupby(["Season", "Division"])
    .agg(
        Matches=("Division", "count"),
        HomeGoals=("HomeGoals", "mean"),
        AwayGoals=("AwayGoals", "mean")
    )
    .reset_index()
)

season_table["TotalGoals"] = (
    season_table["HomeGoals"] +
    season_table["AwayGoals"]
)

print("=== RECENT LEAGUE SCORING ENVIRONMENT ===")

recent = season_table[
    season_table["Season"] >= 2020
]

print(
    recent[
        [
            "Season",
            "Division",
            "Matches",
            "HomeGoals",
            "AwayGoals",
            "TotalGoals"
        ]
    ]
    .round(3)
    .to_string(index=False)
)