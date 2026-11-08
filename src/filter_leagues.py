import pandas as pd

matches = pd.read_csv(
    "data/raw/Matches.csv",
    low_memory=False
)

top_leagues = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1"
}

print("=== TOP 5 EUROPEAN LEAGUES ===\n")

for code, name in top_leagues.items():
    league = matches[matches["Division"] == code]

    print(f"{name} ({code})")
    print(f"Matches: {len(league):,}")
    print(f"Date range: {league['MatchDate'].min()} → {league['MatchDate'].max()}")
    print()