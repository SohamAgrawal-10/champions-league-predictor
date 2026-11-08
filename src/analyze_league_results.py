import pandas as pd

df = pd.read_csv("data/processed/big5_matches.csv")

print("=== RESULT DISTRIBUTION BY LEAGUE ===")

result_table = pd.crosstab(
    df["Division"],
    df["FTResult"],
    normalize="index"
) * 100

result_table.columns = ["Away", "Draw", "Home"]

print(result_table.round(2))

print("\n=== AVERAGE GOALS BY LEAGUE ===")

goal_table = (
    df.groupby("Division")
    .agg(
        HomeGoals=("FTHome", "mean"),
        AwayGoals=("FTAway", "mean")
    )
)

goal_table["TotalGoals"] = (
    goal_table["HomeGoals"] +
    goal_table["AwayGoals"]
)

print(goal_table.round(3))