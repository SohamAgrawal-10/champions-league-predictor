import pandas as pd
from collections import defaultdict


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/rolling_features.csv")

df["MatchDate"] = pd.to_datetime(df["MatchDate"])

df = df.sort_values("MatchDate").reset_index(drop=True)


# --------------------------------------------------
# ELO SETTINGS
# --------------------------------------------------

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 60


# Each team's current Elo rating
ratings = defaultdict(lambda: INITIAL_ELO)


# --------------------------------------------------
# CREATE PRE-MATCH ELO FEATURES
# --------------------------------------------------

rows = []

for _, match in df.iterrows():

    home = match["HomeTeam"]
    away = match["AwayTeam"]

    # ----------------------------------------------
    # IMPORTANT:
    # Capture ratings BEFORE the match
    # ----------------------------------------------

    home_elo = ratings[home]
    away_elo = ratings[away]

    elo_diff = (
        home_elo
        + HOME_ADVANTAGE
        - away_elo
    )

    row = match.copy()

    row["OurHomeElo"] = home_elo
    row["OurAwayElo"] = away_elo
    row["OurEloDiff"] = elo_diff

    rows.append(row)


    # ----------------------------------------------
    # CALCULATE MATCH RESULT
    # ----------------------------------------------

    home_goals = int(match["FTHome"])
    away_goals = int(match["FTAway"])


    # Actual result
    if home_goals > away_goals:
        actual_home = 1
        actual_away = 0

    elif home_goals < away_goals:
        actual_home = 0
        actual_away = 1

    else:
        actual_home = 0.5
        actual_away = 0.5


    # ----------------------------------------------
    # EXPECTED RESULT
    # ----------------------------------------------

    expected_home = 1 / (
        1 + 10 ** (
            -elo_diff / 400
        )
    )

    expected_away = 1 - expected_home


    # ----------------------------------------------
    # UPDATE ELO
    # ----------------------------------------------

    ratings[home] = (
        home_elo
        + K_FACTOR * (
            actual_home - expected_home
        )
    )

    ratings[away] = (
        away_elo
        + K_FACTOR * (
            actual_away - expected_away
        )
    )


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

elo_df = pd.DataFrame(rows)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_path = "data/processed/elo_features.csv"

elo_df.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("=== OUR ELO FEATURES ===")

print(f"Rows: {len(elo_df):,}")
print(f"Columns: {len(elo_df.columns)}")

print("\nNew features:")
print("OurHomeElo")
print("OurAwayElo")
print("OurEloDiff")

print("\nElo range:")

print(
    f"Home Elo: "
    f"{elo_df['OurHomeElo'].min():.1f} → "
    f"{elo_df['OurHomeElo'].max():.1f}"
)

print(
    f"Away Elo: "
    f"{elo_df['OurAwayElo'].min():.1f} → "
    f"{elo_df['OurAwayElo'].max():.1f}"
)

print("\nSample:")
print(
    elo_df[
        [
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "OurHomeElo",
            "OurAwayElo",
            "OurEloDiff"
        ]
    ].head(15).to_string(index=False)
)

print("\nSaved to:")
print(output_path)