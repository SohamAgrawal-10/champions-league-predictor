import pandas as pd

df = pd.read_csv("data/processed/elo_features.csv")

df["MatchDate"] = pd.to_datetime(df["MatchDate"])

league_strength = (
    df.groupby("Division")
    .agg(
        AverageHomeElo=("OurHomeElo", "mean"),
        AverageAwayElo=("OurAwayElo", "mean"),
        Matches=("Division", "count")
    )
    .reset_index()
)

league_strength["AverageElo"] = (
    league_strength["AverageHomeElo"]
    + league_strength["AverageAwayElo"]
) / 2

league_strength = league_strength.sort_values(
    "AverageElo",
    ascending=False
)

print("=== LEAGUE STRENGTH ===")
print(
    league_strength[
        ["Division", "AverageElo", "Matches"]
    ].to_string(index=False)
)