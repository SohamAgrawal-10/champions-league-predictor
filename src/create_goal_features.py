import pandas as pd
from collections import defaultdict, deque

df = pd.read_csv("data/processed/elo_features.csv")
df["MatchDate"] = pd.to_datetime(df["MatchDate"])

df = df.sort_values("MatchDate").reset_index(drop=True)

# Last 5 matches for every team
team_history = defaultdict(lambda: deque(maxlen=5))

rows = []

for _, match in df.iterrows():

    home = match["HomeTeam"]
    away = match["AwayTeam"]

    # -------------------------
    # PRE-MATCH FEATURES
    # -------------------------

    def get_stats(team):
        history = team_history[team]

        if len(history) == 0:
            return {
                "attack": 0,
                "defense": 0
            }

        goals_scored = sum(x[0] for x in history)
        goals_conceded = sum(x[1] for x in history)

        return {
            "attack": goals_scored / len(history),
            "defense": goals_conceded / len(history)
        }

    home_stats = get_stats(home)
    away_stats = get_stats(away)

    row = match.copy()

    row["HomeAttack5"] = home_stats["attack"]
    row["HomeDefense5"] = home_stats["defense"]

    row["AwayAttack5"] = away_stats["attack"]
    row["AwayDefense5"] = away_stats["defense"]

    row["AttackDiff"] = (
        home_stats["attack"] -
        away_stats["attack"]
    )

    row["DefenseDiff"] = (
        home_stats["defense"] -
        away_stats["defense"]
    )

    rows.append(row)

    # -------------------------
    # UPDATE HISTORY
    # -------------------------

    home_goals = int(match["FTHome"])
    away_goals = int(match["FTAway"])

    team_history[home].append(
        (home_goals, away_goals)
    )

    team_history[away].append(
        (away_goals, home_goals)
    )


goal_df = pd.DataFrame(rows)

output_path = "data/processed/goal_features.csv"

goal_df.to_csv(
    output_path,
    index=False
)

print("=== GOAL FEATURES CREATED ===")
print(f"Rows: {len(goal_df):,}")
print(f"Columns: {len(goal_df.columns)}")

print("\nNew features:")
print("HomeAttack5")
print("HomeDefense5")
print("AwayAttack5")
print("AwayDefense5")
print("AttackDiff")
print("DefenseDiff")

print("\nFirst matches:")
print(
    goal_df[
        [
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "HomeAttack5",
            "HomeDefense5",
            "AwayAttack5",
            "AwayDefense5"
        ]
    ].head(10)
)

print("\nSaved to:")
print(output_path)