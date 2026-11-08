import pandas as pd

df = pd.read_csv(
    "data/raw/Matches.csv",
    low_memory=False
)

print("=== FORM FEATURES ===")

print("\nForm3Home:")
print(df["Form3Home"].describe())

print("\nForm5Home:")
print(df["Form5Home"].describe())

print("\nForm3Away:")
print(df["Form3Away"].describe())

print("\nForm5Away:")
print(df["Form5Away"].describe())


print("\n=== UNIQUE VALUES ===")

print("\nForm3Home:")
print(sorted(df["Form3Home"].dropna().unique())[:50])

print("\nForm5Home:")
print(sorted(df["Form5Home"].dropna().unique())[:50])


print("\n=== SAMPLE MATCHES ===")

columns = [
    "MatchDate",
    "HomeTeam",
    "AwayTeam",
    "Form3Home",
    "Form5Home",
    "Form3Away",
    "Form5Away",
    "FTHome",
    "FTAway",
    "FTResult"
]

print(df[columns].head(20).to_string(index=False))