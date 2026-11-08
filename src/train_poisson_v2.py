import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/goal_features.csv")
league = pd.read_csv("data/processed/league_strength.csv")

df["MatchDate"] = pd.to_datetime(df["MatchDate"])

# --------------------------------------------------
# ADD SEASON
# --------------------------------------------------

df["Season"] = df["MatchDate"].dt.year
df.loc[df["MatchDate"].dt.month < 8, "Season"] -= 1

# --------------------------------------------------
# MERGE LEAGUE STRENGTH
# --------------------------------------------------

df = df.merge(
    league,
    on="Division",
    how="left"
)

# --------------------------------------------------
# REMOVE MATCHES WITHOUT ELO
# --------------------------------------------------

df = df.dropna(
    subset=[
        "OurEloDiff",
        "HomeAttack5",
        "AwayAttack5",
        "HomeDefense5",
        "AwayDefense5"
    ]
).copy()

# --------------------------------------------------
# CREATE LEAGUE-ADJUSTED FEATURES
# --------------------------------------------------

df["LeagueHomeGoals"] = df["AvgHomeGoals"]
df["LeagueAwayGoals"] = df["AvgAwayGoals"]

df["HomeAttackRatio"] = (
    df["HomeAttack5"] /
    df["AvgHomeGoals"]
)

df["AwayAttackRatio"] = (
    df["AwayAttack5"] /
    df["AvgAwayGoals"]
)

df["HomeDefenseRatio"] = (
    df["HomeDefense5"] /
    df["AvgAwayGoals"]
)

df["AwayDefenseRatio"] = (
    df["AwayDefense5"] /
    df["AvgHomeGoals"]
)

# Replace problematic values
ratio_columns = [
    "HomeAttackRatio",
    "AwayAttackRatio",
    "HomeDefenseRatio",
    "AwayDefenseRatio"
]

df[ratio_columns] = (
    df[ratio_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(1.0)
)

# --------------------------------------------------
# CREATE EXPECTED GOAL FEATURES
# --------------------------------------------------

df["HomeBaseGoals"] = (
    df["LeagueHomeGoals"] *
    df["HomeAttackRatio"] *
    df["AwayDefenseRatio"]
)

df["AwayBaseGoals"] = (
    df["LeagueAwayGoals"] *
    df["AwayAttackRatio"] *
    df["HomeDefenseRatio"]
)

# --------------------------------------------------
# TIME SPLIT
# --------------------------------------------------

train = df[df["MatchDate"] < "2022-01-01"].copy()

validation = df[
    (df["MatchDate"] >= "2022-01-01") &
    (df["MatchDate"] < "2024-01-01")
].copy()

test = df[df["MatchDate"] >= "2024-01-01"].copy()

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

features = [
    "OurEloDiff",
    "HomeBaseGoals",
    "AwayBaseGoals",
    "HomeAttack5",
    "AwayAttack5",
    "HomeDefense5",
    "AwayDefense5"
]

X_train = train[features]
X_val = validation[features]
X_test = test[features]

y_home_train = train["FTHome"]
y_home_val = validation["FTHome"]
y_home_test = test["FTHome"]

y_away_train = train["FTAway"]
y_away_val = validation["FTAway"]
y_away_test = test["FTAway"]

# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

home_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000
)

away_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000
)

home_model.fit(X_train, y_home_train)
away_model.fit(X_train, y_away_train)

# --------------------------------------------------
# PREDICT
# --------------------------------------------------

home_val_pred = home_model.predict(X_val)
away_val_pred = away_model.predict(X_val)

home_test_pred = home_model.predict(X_test)
away_test_pred = away_model.predict(X_test)

# --------------------------------------------------
# EVALUATE
# --------------------------------------------------

print("=== LEAGUE-ADJUSTED POISSON ===")

print("\nVALIDATION")
print(
    "Home goal deviance:",
    round(
        mean_poisson_deviance(
            y_home_val,
            home_val_pred
        ),
        4
    )
)

print(
    "Away goal deviance:",
    round(
        mean_poisson_deviance(
            y_away_val,
            away_val_pred
        ),
        4
    )
)

print("\nTEST")
print(
    "Home goal deviance:",
    round(
        mean_poisson_deviance(
            y_home_test,
            home_test_pred
        ),
        4
    )
)

print(
    "Away goal deviance:",
    round(
        mean_poisson_deviance(
            y_away_test,
            away_test_pred
        ),
        4
    )
)

print("\n=== SAMPLE EXPECTED GOALS ===")

samples = validation.head(10).copy()

sample_home = home_model.predict(
    samples[features]
)

sample_away = away_model.predict(
    samples[features]
)

for i, (_, row) in enumerate(samples.iterrows()):
    print(
        f"{row['HomeTeam']} vs {row['AwayTeam']} "
        f"-> {sample_home[i]:.2f} - {sample_away[i]:.2f}"
    )