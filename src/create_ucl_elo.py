import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

UCL_RESULTS = BASE_DIR / "data" / "raw" / "ucl" / "results_2627.csv"
OUTPUT = BASE_DIR / "data" / "processed" / "ucl_final_elo.csv"


# ---------------------------------------------------------
# Current EloFootball ratings
#
# These are used as the current baseline.
# We DO NOT replay the completed UCL matches because
# these ratings already reflect current results.
# ---------------------------------------------------------

ELOFOOTBALL_RATINGS = {
    "AEK Athens": 1950,
    "Arsenal": 2383,
    "Aston Villa": 2176,
    "Atleti": 2175,
    "Barcelona": 2335,
    "Bayern München": 2403,
    "Bodø/Glimt": 2058,
    "B. Dortmund": 2176,
    "Club Brugge": 2134,
    "Como": 2169,
    "Fenerbahçe": 1989,
    "Feyenoord": 1923,
    "Galatasaray": 2041,
    "Inter": 2222,
    "LASK": 1878,
    "Leipzig": 2061,
    "Lens": 2001,
    "Lille": 1995,
    "Liverpool": 2171,
    "Man City": 2289,
    "Man Utd": 2085,
    "Napoli": 2044,
    "Paris": 2363,
    "Porto": 2094,
    "PSV": 2018,
    "Real Betis": 2084,
    "Real Madrid": 2302,
    "Roma": 2135,
    "Sabah": 1782,
    "Shakhtar": 1930,
    "Slavia Praha": 1992,
    "S. Bratislava": 1764,
    "Sporting CP": 2136,
    "Stuttgart": 2074,
    "Viking": 1964,
    "Villarreal": 2006,
}


# ---------------------------------------------------------
# Load UCL results
# ---------------------------------------------------------

results = pd.read_csv(UCL_RESULTS)

results["date"] = pd.to_datetime(results["date"])


# ---------------------------------------------------------
# Verify that every UCL team has an Elo rating
# ---------------------------------------------------------

teams = sorted(
    set(results["home"]).union(set(results["away"]))
)

missing = [team for team in teams if team not in ELOFOOTBALL_RATINGS]

if missing:
    raise ValueError(
        f"Missing Elo ratings for: {missing}"
    )


# ---------------------------------------------------------
# Create final table
# ---------------------------------------------------------

rows = []

for team in teams:

    team_matches = results[
        (results["home"] == team) |
        (results["away"] == team)
    ]

    rows.append({
        "Team": team,
        "Elo": ELOFOOTBALL_RATINGS[team],
        "UCLMatchesPlayed": len(team_matches),
    })


final_elo = pd.DataFrame(rows)

final_elo = final_elo.sort_values(
    "Elo",
    ascending=False
).reset_index(drop=True)

final_elo["Rank"] = range(1, len(final_elo) + 1)

final_elo = final_elo[
    [
        "Rank",
        "Team",
        "Elo",
        "UCLMatchesPlayed"
    ]
]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_elo.to_csv(
    OUTPUT,
    index=False
)


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("UCL CURRENT ELO")
print("---------------")
print(final_elo.to_string(index=False))

print()
print(f"Saved to: {OUTPUT}")