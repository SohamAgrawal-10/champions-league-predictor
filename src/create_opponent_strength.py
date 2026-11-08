import pandas as pd
from collections import defaultdict, deque

df = pd.read_csv("data/processed/elo_features.csv")

df["MatchDate"] = pd.to_datetime(df["MatchDate"])
df = df.sort_values("MatchDate").reset_index(drop=True)

# Store the previous 5 opponent Elo ratings for each team
opponent_history = defaultdict(lambda: deque(maxlen=5))

rows = []

for _, match in df.iterrows():

    home = match["HomeTeam"]
    away = match["AwayTeam"]

    # ---------------------------------------------------------
    # PRE-MATCH FEATURES
    # ---------------------------------------------------------

    home_opponents = list(opponent_history[home])
    away_opponents = list(opponent_history[away])

    if home_opponents:
        home_opponent_elo = sum(home_opponents) / len(home_opponents)
    else:
        home_opponent_elo = 1500

    if away_opponents:
        away_opponent_elo = sum(away_opponents) / len(away_opponents)
    else:
        away_opponent_elo = 1500

    opponent_strength_diff = (
        home_opponent_elo - away_opponent_elo
    )

    row = match.copy()

    row["HomeOpponentElo5"] = home_opponent_elo
    row["AwayOpponentElo5"] = away_opponent_elo
    row["OpponentStrengthDiff"] = opponent_strength_diff

    rows.append(row)

    # ---------------------------------------------------------
    # UPDATE HISTORY AFTER THE MATCH
    # ---------------------------------------------------------

    home_elo = match["OurHomeElo"]
    away_elo = match["OurAwayElo"]

    opponent_history[home].append(away_elo)
    opponent_history[away].append(home_elo)


output = pd.DataFrame(rows)

output_path = "data/processed/opponent_strength_features.csv"

output.to_csv(output_path, index=False)

print("Opponent strength features created successfully!")
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
            "HomeOpponentElo5",
            "AwayOpponentElo5",
            "OpponentStrengthDiff",
        ]
    ].head(10)
)