import pandas as pd

df = pd.read_csv("data/processed/home_away_features.csv")

print("=== FEATURE SUMMARY ===")

print(
    df[
        [
            "HomeTeamHomeForm5Points",
            "AwayTeamAwayForm5Points",
            "HomeTeamHomeGoalsScored5",
            "AwayTeamAwayGoalsScored5",
            "HomeTeamHomeGoalsConceded5",
            "AwayTeamAwayGoalsConceded5",
            "HomeTeamHomeGoalDifference5",
            "AwayTeamAwayGoalDifference5",
            "HomeTeamHomeWinRate5",
            "AwayTeamAwayWinRate5",
            "HomeAwayPointsDiff",
            "HomeAwayGoalDifferenceDiff",
            "HomeAwayWinRateDiff",
        ]
    ].describe()
)

print("\n=== LATEST MATCHES ===")

print(
    df[
        [
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "HomeTeamHomeForm5Points",
            "AwayTeamAwayForm5Points",
            "HomeAwayPointsDiff",
            "HomeAwayGoalDifferenceDiff",
            "HomeAwayWinRateDiff",
        ]
    ].tail(10)
)