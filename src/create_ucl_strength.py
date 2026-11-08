import pandas as pd


INPUT_FILE = "data/raw/ucl/results_2627.csv"
OUTPUT_FILE = "data/processed/ucl_team_strength.csv"


def get_result_points(result, is_home):
    if result == "D":
        return 1

    if is_home and result == "H":
        return 3

    if not is_home and result == "A":
        return 3

    return 0


def main():
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    teams = sorted(set(df["home"]) | set(df["away"]))

    records = []

    for team in teams:
        home = df[df["home"] == team]
        away = df[df["away"] == team]

        matches = len(home) + len(away)

        home_wins = (home["ft_result"] == "H").sum()
        away_wins = (away["ft_result"] == "A").sum()

        home_draws = (home["ft_result"] == "D").sum()
        away_draws = (away["ft_result"] == "D").sum()

        home_losses = (home["ft_result"] == "A").sum()
        away_losses = (away["ft_result"] == "H").sum()

        wins = home_wins + away_wins
        draws = home_draws + away_draws
        losses = home_losses + away_losses

        goals_scored = (
            home["ft_home"].sum()
            + away["ft_away"].sum()
        )

        goals_conceded = (
            home["ft_away"].sum()
            + away["ft_home"].sum()
        )

        points = (wins * 3) + draws

        records.append({
            "Team": team,
            "Matches": matches,
            "Wins": wins,
            "Draws": draws,
            "Losses": losses,
            "Points": points,
            "GoalsScored": goals_scored,
            "GoalsConceded": goals_conceded,
            "GoalDifference": goals_scored - goals_conceded,
            "PointsPerGame": points / matches,
            "GoalsScoredPerGame": goals_scored / matches,
            "GoalsConcededPerGame": goals_conceded / matches,
        })

    strength = pd.DataFrame(records)

    strength = strength.sort_values(
        ["Points", "GoalDifference", "GoalsScored"],
        ascending=False
    ).reset_index(drop=True)

    strength["Rank"] = strength.index + 1

    strength = strength[
        [
            "Rank",
            "Team",
            "Matches",
            "Wins",
            "Draws",
            "Losses",
            "Points",
            "GoalsScored",
            "GoalsConceded",
            "GoalDifference",
            "PointsPerGame",
            "GoalsScoredPerGame",
            "GoalsConcededPerGame",
        ]
    ]

    strength.to_csv(OUTPUT_FILE, index=False)

    print("\nUCL TEAM STRENGTH")
    print("-----------------")
    print(strength.to_string(index=False))

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()