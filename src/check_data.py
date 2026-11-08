import pandas as pd

matches = pd.read_csv(
    "data/raw/Matches.csv",
    low_memory=False
)

elo = pd.read_csv(
    "data/raw/EloRatings.csv"
)

print("=== DATASET OVERVIEW ===")

print("\nMatches shape:")
print(matches.shape)

print("\nDate range:")
print(matches["MatchDate"].min(), "to", matches["MatchDate"].max())

print("\nNumber of divisions:")
print(matches["Division"].nunique())

print("\nDivisions:")
print(sorted(matches["Division"].dropna().unique()))

print("\n\n=== RESULT DISTRIBUTION ===")
print(matches["FTResult"].value_counts(dropna=False))

print("\n\n=== MISSING VALUES ===")
print(matches.isna().sum().sort_values(ascending=False).head(15))

print("\n\n=== LATEST 10 MATCHES ===")
print(
    matches[
        ["MatchDate", "Division", "HomeTeam", "AwayTeam", "FTHome", "FTAway", "FTResult"]
    ].tail(10)
)

print("\n\n=== ELO DATA ===")

print("Shape:")
print(elo.shape)

print("\nDate range:")
print(elo["date"].min(), "to", elo["date"].max())

print("\nCountries:")
print(sorted(elo["country"].dropna().unique()))