import pandas as pd
import numpy as np

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/goal_features.csv")

df["MatchDate"] = pd.to_datetime(df["MatchDate"])

df = df.dropna(
    subset=[
        "OurHomeElo",
        "OurAwayElo"
    ]
).copy()


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train = df[df["MatchDate"] < "2022-01-01"].copy()

validation = df[
    (df["MatchDate"] >= "2022-01-01") &
    (df["MatchDate"] < "2024-01-01")
].copy()

test = df[df["MatchDate"] >= "2024-01-01"].copy()


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    "OurEloDiff",

    "HomeAttack5",
    "AwayAttack5",

    "HomeDefense5",
    "AwayDefense5",

    "AttackDiff",
    "DefenseDiff"
]


# ============================================================
# TRAIN HOME GOAL MODEL
# ============================================================

X_train = train[feature_columns]
y_home_train = train["FTHome"]

X_validation = validation[feature_columns]
y_home_validation = validation["FTHome"]

X_test = test[feature_columns]
y_home_test = test["FTHome"]


home_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000
)

home_model.fit(
    X_train,
    y_home_train
)


# ============================================================
# TRAIN AWAY GOAL MODEL
# ============================================================

y_away_train = train["FTAway"]

y_away_validation = validation["FTAway"]

y_away_test = test["FTAway"]


away_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000
)

away_model.fit(
    X_train,
    y_away_train
)


# ============================================================
# PREDICT EXPECTED GOALS
# ============================================================

home_validation_xg = home_model.predict(
    X_validation
)

away_validation_xg = away_model.predict(
    X_validation
)

home_test_xg = home_model.predict(
    X_test
)

away_test_xg = away_model.predict(
    X_test
)


# ============================================================
# EVALUATE GOAL PREDICTIONS
# ============================================================

print("=== HOME GOAL MODEL ===")

print(
    "Validation Poisson Deviance:",
    round(
        mean_poisson_deviance(
            y_home_validation,
            home_validation_xg
        ),
        4
    )
)

print(
    "Test Poisson Deviance:",
    round(
        mean_poisson_deviance(
            y_home_test,
            home_test_xg
        ),
        4
    )
)


print("\n=== AWAY GOAL MODEL ===")

print(
    "Validation Poisson Deviance:",
    round(
        mean_poisson_deviance(
            y_away_validation,
            away_validation_xg
        ),
        4
    )
)

print(
    "Test Poisson Deviance:",
    round(
        mean_poisson_deviance(
            y_away_test,
            away_test_xg
        ),
        4
    )
)


# ============================================================
# SAMPLE EXPECTED GOALS
# ============================================================

print("\n=== SAMPLE EXPECTED GOALS ===")

samples = test.iloc[:10].copy()

sample_home_xg = home_model.predict(
    samples[feature_columns]
)

sample_away_xg = away_model.predict(
    samples[feature_columns]
)

for i, (_, match) in enumerate(samples.iterrows()):

    print(
        f"{match['HomeTeam']} vs {match['AwayTeam']} "
        f"-> "
        f"Expected goals: "
        f"{sample_home_xg[i]:.2f} - "
        f"{sample_away_xg[i]:.2f}"
    )