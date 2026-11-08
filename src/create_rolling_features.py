import pandas as pd
from collections import defaultdict, deque


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/big5_matches.csv")

df["MatchDate"] = pd.to_datetime(df["MatchDate"])

# Remove matches without Elo
df = df.dropna(subset=["HomeElo", "AwayElo"]).copy()

# Sort chronologically
df = df.sort_values("MatchDate").reset_index(drop=True)


# --------------------------------------------------
# TEAM HISTORY
# --------------------------------------------------

# For every team we store their previous match results.
team_history = defaultdict(lambda: deque(maxlen=5))


# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def calculate_form(history):
    """
    Calculate rolling statistics from a team's
    previous matches.

    history contains tuples:
        (points, goals_scored, goals_conceded)
    """

    if len(history) == 0:
        return {
            "points": 0,
            "goals_scored": 0,
            "goals_conceded": 0,
            "goal_difference": 0,
            "win_rate": 0,
        }

    points = sum(match[0] for match in history)
    goals_scored = sum(match[1] for match in history)
    goals_conceded = sum(match[2] for match in history)

    wins = sum(1 for match in history if match[0] == 3)

    return {
        "points": points,
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "goal_difference": goals_scored - goals_conceded,
        "win_rate": wins / len(history),
    }


# --------------------------------------------------
# CREATE FEATURES
# --------------------------------------------------

rows = []

for _, match in df.iterrows():

    home = match["HomeTeam"]
    away = match["AwayTeam"]

    # IMPORTANT:
    # Calculate form BEFORE adding the current match.
    home_form = calculate_form(team_history[home])
    away_form = calculate_form(team_history[away])

    row = match.copy()

    # Home form
    row["HomeForm5Points"] = home_form["points"]
    row["HomeForm5GoalsScored"] = home_form["goals_scored"]
    row["HomeForm5GoalsConceded"] = home_form["goals_conceded"]
    row["HomeForm5GoalDifference"] = home_form["goal_difference"]
    row["HomeForm5WinRate"] = home_form["win_rate"]

    # Away form
    row["AwayForm5Points"] = away_form["points"]
    row["AwayForm5GoalsScored"] = away_form["goals_scored"]
    row["AwayForm5GoalsConceded"] = away_form["goals_conceded"]
    row["AwayForm5GoalDifference"] = away_form["goal_difference"]
    row["AwayForm5WinRate"] = away_form["win_rate"]

    # Differences
    row["Form5PointsDiff"] = (
        home_form["points"] - away_form["points"]
    )

    row["Form5GoalsScoredDiff"] = (
        home_form["goals_scored"] - away_form["goals_scored"]
    )

    row["Form5GoalsConcededDiff"] = (
        home_form["goals_conceded"] - away_form["goals_conceded"]
    )

    row["Form5GoalDifferenceDiff"] = (
        home_form["goal_difference"] - away_form["goal_difference"]
    )

    row["Form5WinRateDiff"] = (
        home_form["win_rate"] - away_form["win_rate"]
    )

    rows.append(row)

    # --------------------------------------------------
    # UPDATE HISTORY AFTER FEATURES ARE CALCULATED
    # --------------------------------------------------

    home_goals = int(match["FTHome"])
    away_goals = int(match["FTAway"])

    if home_goals > away_goals:
        home_points = 3
        away_points = 0
    elif home_goals < away_goals:
        home_points = 0
        away_points = 3
    else:
        home_points = 1
        away_points = 1

    team_history[home].append(
        (home_points, home_goals, away_goals)
    )

    team_history[away].append(
        (away_points, away_goals, home_goals)
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

rolling_df = pd.DataFrame(rows)

output_path = "data/processed/rolling_features.csv"

rolling_df.to_csv(output_path, index=False)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("=== ROLLING FEATURES CREATED ===")

print(f"Rows: {len(rolling_df):,}")
print(f"Columns: {len(rolling_df.columns)}")

print("\nNew features:")

new_features = [
    "HomeForm5Points",
    "AwayForm5Points",
    "HomeForm5GoalsScored",
    "AwayForm5GoalsScored",
    "HomeForm5GoalsConceded",
    "AwayForm5GoalsConceded",
    "HomeForm5GoalDifference",
    "AwayForm5GoalDifference",
    "HomeForm5WinRate",
    "AwayForm5WinRate",
    "Form5PointsDiff",
    "Form5GoalsScoredDiff",
    "Form5GoalsConcededDiff",
    "Form5GoalDifferenceDiff",
    "Form5WinRateDiff",
]

for feature in new_features:
    print(feature)

print("\nSaved to:")
print(output_path)