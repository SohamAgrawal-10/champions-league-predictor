import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance


# --------------------------------------------------
# LOAD TEAM STRENGTH FEATURES
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/team_strength_features.csv"
)

df["MatchDate"] = pd.to_datetime(df["MatchDate"])


# --------------------------------------------------
# LOAD DYNAMIC ELO FEATURES
# --------------------------------------------------

elo = pd.read_csv(
    "data/processed/elo_features.csv"
)

elo["MatchDate"] = pd.to_datetime(elo["MatchDate"])


# --------------------------------------------------
# MERGE DYNAMIC ELO
# --------------------------------------------------

df = df.merge(
    elo[
        [
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "OurHomeElo",
            "OurAwayElo",
            "OurEloDiff"
        ]
    ],
    on=[
        "MatchDate",
        "HomeTeam",
        "AwayTeam"
    ],
    how="left"
)


# --------------------------------------------------
# REMOVE MISSING VALUES
# --------------------------------------------------

df = df.dropna(
    subset=[
        "OurEloDiff",
        "HomeAttackStrength5",
        "HomeDefenceStrength5",
        "AwayAttackStrength5",
        "AwayDefenceStrength5",
        "FTHome",
        "FTAway"
    ]
).copy()


# --------------------------------------------------
# TIME-BASED SPLIT
# --------------------------------------------------

train = df[
    df["MatchDate"] < "2022-01-01"
].copy()

validation = df[
    (df["MatchDate"] >= "2022-01-01") &
    (df["MatchDate"] < "2024-01-01")
].copy()

test = df[
    df["MatchDate"] >= "2024-01-01"
].copy()


print("DATASET SPLIT")
print("-------------")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

features = [
    "OurEloDiff",

    "HomeAttackStrength5",
    "HomeDefenceStrength5",

    "AwayAttackStrength5",
    "AwayDefenceStrength5",

    "AttackStrengthDiff",
    "DefenceStrengthDiff"
]


X_train = train[features]
X_val = validation[features]
X_test = test[features]


# --------------------------------------------------
# TARGETS
# --------------------------------------------------

y_home_train = train["FTHome"]
y_home_val = validation["FTHome"]
y_home_test = test["FTHome"]

y_away_train = train["FTAway"]
y_away_val = validation["FTAway"]
y_away_test = test["FTAway"]


# --------------------------------------------------
# TRAIN HOME GOAL MODEL
# --------------------------------------------------

home_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000
)

home_model.fit(
    X_train,
    y_home_train
)


# --------------------------------------------------
# TRAIN AWAY GOAL MODEL
# --------------------------------------------------

away_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000
)

away_model.fit(
    X_train,
    y_away_train
)


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

print("\n=== TEAM-STRENGTH POISSON V2 ===")

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


# --------------------------------------------------
# SAMPLE EXPECTED GOALS
# --------------------------------------------------

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
        f"-> "
        f"{sample_home[i]:.2f} - "
        f"{sample_away[i]:.2f}"
    )