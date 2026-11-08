import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_PATH = BASE_DIR / "data" / "raw" / "ucl" / "results_2627.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "ucl_form.csv"


# ---------------------------------------------------------
# Load results
# ---------------------------------------------------------

df = pd.read_csv(RESULTS_PATH)

df["date"] = pd.to_datetime(df["date"])


# ---------------------------------------------------------
# Build team-level records
# ---------------------------------------------------------

team_rows = []

for _, match in df.iterrows():

    home = match["home"]
    away = match["away"]

    home_goals = match["ft_home"]
    away_goals = match["ft_away"]

    # Home team
    if home_goals > away_goals:
        home_points = 3
    elif home_goals == away_goals:
        home_points = 1
    else:
        home_points = 0

    # Away team
    if away_goals > home_goals:
        away_points = 3
    elif away_goals == home_goals:
        away_points = 1
    else:
        away_points = 0

    team_rows.append({
        "Team": home,
        "Matches": 1,
        "Points": home_points,
        "GoalsFor": home_goals,
        "GoalsAgainst": away_goals,
    })

    team_rows.append({
        "Team": away,
        "Matches": 1,
        "Points": away_points,
        "GoalsFor": away_goals,
        "GoalsAgainst": home_goals,
    })


team_df = pd.DataFrame(team_rows)


# ---------------------------------------------------------
# Aggregate
# ---------------------------------------------------------

form = (
    team_df
    .groupby("Team")
    .agg(
        Matches=("Matches", "sum"),
        Points=("Points", "sum"),
        GoalsFor=("GoalsFor", "sum"),
        GoalsAgainst=("GoalsAgainst", "sum"),
    )
    .reset_index()
)


# ---------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------

form["PointsPerGame"] = (
    form["Points"] / form["Matches"]
)

form["GoalsForPerGame"] = (
    form["GoalsFor"] / form["Matches"]
)

form["GoalsAgainstPerGame"] = (
    form["GoalsAgainst"] / form["Matches"]
)

form["GoalDifference"] = (
    form["GoalsFor"] - form["GoalsAgainst"]
)

form["GoalDifferencePerGame"] = (
    form["GoalDifference"] / form["Matches"]
)


# ---------------------------------------------------------
# Sort
# ---------------------------------------------------------

form = form.sort_values(
    ["Points", "GoalDifference", "GoalsFor"],
    ascending=False
).reset_index(drop=True)

form["UCLRank"] = range(1, len(form) + 1)


form = form[
    [
        "UCLRank",
        "Team",
        "Matches",
        "Points",
        "PointsPerGame",
        "GoalsFor",
        "GoalsAgainst",
        "GoalsForPerGame",
        "GoalsAgainstPerGame",
        "GoalDifference",
        "GoalDifferencePerGame",
    ]
]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

form.to_csv(
    OUTPUT_PATH,
    index=False
)


print("CURRENT UCL FORM")
print("----------------")
print(form.to_string(index=False))

print()
print(f"Saved to: {OUTPUT_PATH}")