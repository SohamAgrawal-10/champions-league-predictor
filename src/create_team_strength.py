import pandas as pd
from collections import defaultdict, deque


INPUT_FILE = "data/processed/big5_matches.csv"
OUTPUT_FILE = "data/processed/team_strength_features.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)
df["MatchDate"] = pd.to_datetime(df["MatchDate"])

df = df.sort_values("MatchDate").reset_index(drop=True)


# ---------------------------------------------------------
# TEAM MATCH HISTORY
# ---------------------------------------------------------

team_history = defaultdict(lambda: deque(maxlen=5))

features = []


# ---------------------------------------------------------
# CREATE PRE-MATCH FEATURES
# ---------------------------------------------------------

for _, row in df.iterrows():

    home = row["HomeTeam"]
    away = row["AwayTeam"]

    home_history = team_history[home]
    away_history = team_history[away]

    # Home team's recent attacking/defensive strength
    if len(home_history) > 0:
        home_attack = (
            sum(x["goals_scored"] for x in home_history)
            / len(home_history)
        )

        home_defence = (
            sum(x["goals_conceded"] for x in home_history)
            / len(home_history)
        )

    else:
        home_attack = 1.5
        home_defence = 1.5

    # Away team's recent attacking/defensive strength
    if len(away_history) > 0:
        away_attack = (
            sum(x["goals_scored"] for x in away_history)
            / len(away_history)
        )

        away_defence = (
            sum(x["goals_conceded"] for x in away_history)
            / len(away_history)
        )

    else:
        away_attack = 1.5
        away_defence = 1.5

    # Store pre-match features
    features.append({
        "HomeAttackStrength5": home_attack,
        "HomeDefenceStrength5": home_defence,
        "AwayAttackStrength5": away_attack,
        "AwayDefenceStrength5": away_defence,

        "AttackStrengthDiff": home_attack - away_attack,

        "DefenceStrengthDiff": away_defence - home_defence,
    })

    # -----------------------------------------------------
    # UPDATE HISTORY AFTER MATCH
    # -----------------------------------------------------

    team_history[home].append({
        "goals_scored": row["FTHome"],
        "goals_conceded": row["FTAway"],
    })

    team_history[away].append({
        "goals_scored": row["FTAway"],
        "goals_conceded": row["FTHome"],
    })


# ---------------------------------------------------------
# COMBINE FEATURES WITH ORIGINAL DATA
# ---------------------------------------------------------

feature_df = pd.DataFrame(features)

result = pd.concat([df, feature_df], axis=1)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

result.to_csv(OUTPUT_FILE, index=False)


# ---------------------------------------------------------
# CHECK
# ---------------------------------------------------------

print(f"Saved: {OUTPUT_FILE}")
print(f"Rows: {len(result)}")
print(f"Columns: {len(result.columns)}")

print("\nNew features:")
print(feature_df.columns.tolist())

print("\nFeature summary:")
print(
    feature_df[
        [
            "HomeAttackStrength5",
            "HomeDefenceStrength5",
            "AwayAttackStrength5",
            "AwayDefenceStrength5",
        ]
    ].describe()
)

print("\nFirst 5 rows:")
print(
    result[
        [
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "HomeAttackStrength5",
            "HomeDefenceStrength5",
            "AwayAttackStrength5",
            "AwayDefenceStrength5",
        ]
    ].head()
)