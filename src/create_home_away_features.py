import pandas as pd
from collections import defaultdict, deque

df = pd.read_csv("data/processed/opponent_strength_features.csv")

df["MatchDate"] = pd.to_datetime(df["MatchDate"])
df = df.sort_values("MatchDate").reset_index(drop=True)


# Store the previous 5 home matches for each team
home_history = defaultdict(lambda: deque(maxlen=5))

# Store the previous 5 away matches for each team
away_history = defaultdict(lambda: deque(maxlen=5))


rows = []


for _, match in df.iterrows():

    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    # ---------------------------------------------------------
    # PRE-MATCH HOME FORM
    # ---------------------------------------------------------

    previous_home = list(home_history[home_team])

    if previous_home:

        home_points = sum(x["points"] for x in previous_home)
        home_goals_scored = sum(x["goals_scored"] for x in previous_home)
        home_goals_conceded = sum(x["goals_conceded"] for x in previous_home)
        home_wins = sum(x["win"] for x in previous_home)

        home_goal_difference = (
            home_goals_scored - home_goals_conceded
        )

        home_win_rate = home_wins / len(previous_home)

    else:

        home_points = 0
        home_goals_scored = 0
        home_goals_conceded = 0
        home_goal_difference = 0
        home_win_rate = 0


    # ---------------------------------------------------------
    # PRE-MATCH AWAY FORM
    # ---------------------------------------------------------

    previous_away = list(away_history[away_team])

    if previous_away:

        away_points = sum(x["points"] for x in previous_away)
        away_goals_scored = sum(x["goals_scored"] for x in previous_away)
        away_goals_conceded = sum(x["goals_conceded"] for x in previous_away)
        away_wins = sum(x["win"] for x in previous_away)

        away_goal_difference = (
            away_goals_scored - away_goals_conceded
        )

        away_win_rate = away_wins / len(previous_away)

    else:

        away_points = 0
        away_goals_scored = 0
        away_goals_conceded = 0
        away_goal_difference = 0
        away_win_rate = 0


    # ---------------------------------------------------------
    # CREATE PRE-MATCH ROW
    # ---------------------------------------------------------

    row = match.copy()

    row["HomeTeamHomeForm5Points"] = home_points
    row["AwayTeamAwayForm5Points"] = away_points

    row["HomeTeamHomeGoalsScored5"] = home_goals_scored
    row["AwayTeamAwayGoalsScored5"] = away_goals_scored

    row["HomeTeamHomeGoalsConceded5"] = home_goals_conceded
    row["AwayTeamAwayGoalsConceded5"] = away_goals_conceded

    row["HomeTeamHomeGoalDifference5"] = home_goal_difference
    row["AwayTeamAwayGoalDifference5"] = away_goal_difference

    row["HomeTeamHomeWinRate5"] = home_win_rate
    row["AwayTeamAwayWinRate5"] = away_win_rate

    row["HomeAwayPointsDiff"] = (
        home_points - away_points
    )

    row["HomeAwayGoalDifferenceDiff"] = (
        home_goal_difference - away_goal_difference
    )

    row["HomeAwayWinRateDiff"] = (
        home_win_rate - away_win_rate
    )

    rows.append(row)


    # ---------------------------------------------------------
    # UPDATE HISTORY AFTER THE MATCH
    # ---------------------------------------------------------

    home_goals = int(match["FTHome"])
    away_goals = int(match["FTAway"])


    if home_goals > away_goals:
        home_points_result = 3
        away_points_result = 0

        home_win = 1
        away_win = 0

    elif home_goals < away_goals:
        home_points_result = 0
        away_points_result = 3

        home_win = 0
        away_win = 1

    else:
        home_points_result = 1
        away_points_result = 1

        home_win = 0
        away_win = 0


    home_history[home_team].append(
        {
            "points": home_points_result,
            "goals_scored": home_goals,
            "goals_conceded": away_goals,
            "win": home_win,
        }
    )


    away_history[away_team].append(
        {
            "points": away_points_result,
            "goals_scored": away_goals,
            "goals_conceded": home_goals,
            "win": away_win,
        }
    )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output = pd.DataFrame(rows)

output_path = "data/processed/home_away_features.csv"

output.to_csv(output_path, index=False)

print("Home/away features created successfully!")
print(f"Rows: {len(output):,}")
print(f"Columns: {len(output.columns)}")
print(f"Saved to: {output_path}")

print("\nSample:")
print(
    output[
        [
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "HomeTeamHomeForm5Points",
            "AwayTeamAwayForm5Points",
            "HomeAwayPointsDiff",
        ]
    ].head(15)
)