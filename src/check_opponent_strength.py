import pandas as pd

df = pd.read_csv("data/processed/opponent_strength_features.csv")

print("=== FEATURE SUMMARY ===")

print(
    df[
        [
            "HomeOpponentElo5",
            "AwayOpponentElo5",
            "OpponentStrengthDiff",
        ]
    ].describe()
)

print("\n=== LARGEST OPPONENT STRENGTH DIFFERENCES ===")

print(
    df[
        [
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "HomeOpponentElo5",
            "AwayOpponentElo5",
            "OpponentStrengthDiff",
        ]
    ]
    .sort_values("OpponentStrengthDiff", ascending=False)
    .head(10)
)

print("\n=== LATEST MATCHES ===")

print(
    df[
        [
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "HomeOpponentElo5",
            "AwayOpponentElo5",
            "OpponentStrengthDiff",
        ]
    ].tail(10)
)