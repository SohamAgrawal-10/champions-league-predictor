import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import PoissonRegressor


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "big5_matches.csv"

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "strength_goal_model.csv"
)


# ---------------------------------------------------------
# Load historical matches
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["MatchDate"] = pd.to_datetime(df["MatchDate"])

df = df.dropna(
    subset=[
        "HomeElo",
        "AwayElo",
        "FTHome",
        "FTAway",
    ]
).copy()


# ---------------------------------------------------------
# Relative strength
#
# Instead of feeding absolute Elo values into the model,
# use the difference between the teams.
#
# +1 = 100 Elo points stronger
# ---------------------------------------------------------

df["EloDiff"] = (
    df["HomeElo"] - df["AwayElo"]
) / 100


# ---------------------------------------------------------
# Time split
# ---------------------------------------------------------

train = df[
    df["MatchDate"] < "2022-01-01"
].copy()

test = df[
    df["MatchDate"] >= "2022-01-01"
].copy()


X_train = train[["EloDiff"]]
X_test = test[["EloDiff"]]


# ---------------------------------------------------------
# Home goals
# ---------------------------------------------------------

home_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000
)

home_model.fit(
    X_train,
    train["FTHome"]
)


# ---------------------------------------------------------
# Away goals
#
# Away goals should decrease as home advantage increases.
# ---------------------------------------------------------

away_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000
)

away_model.fit(
    X_train,
    train["FTAway"]
)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

home_predictions = home_model.predict(X_test)
away_predictions = away_model.predict(X_test)

home_predictions = np.maximum(
    home_predictions,
    0.01
)

away_predictions = np.maximum(
    away_predictions,
    0.01
)


# ---------------------------------------------------------
# Output model
# ---------------------------------------------------------

print()
print("HOME GOAL MODEL")
print("----------------")

print(
    f"Intercept: "
    f"{home_model.intercept_:.4f}"
)

print(
    f"Elo difference coefficient: "
    f"{home_model.coef_[0]:.4f}"
)


print()
print("AWAY GOAL MODEL")
print("----------------")

print(
    f"Intercept: "
    f"{away_model.intercept_:.4f}"
)

print(
    f"Elo difference coefficient: "
    f"{away_model.coef_[0]:.4f}"
)


# ---------------------------------------------------------
# Historical test performance
# ---------------------------------------------------------

print()
print("AVERAGE TEST PREDICTIONS")
print("------------------------")

print(
    f"Home goals: "
    f"{home_predictions.mean():.3f}"
)

print(
    f"Away goals: "
    f"{away_predictions.mean():.3f}"
)

print(
    f"Actual home goals: "
    f"{test['FTHome'].mean():.3f}"
)

print(
    f"Actual away goals: "
    f"{test['FTAway'].mean():.3f}"
)


# ---------------------------------------------------------
# Save coefficients
# ---------------------------------------------------------

model_summary = pd.DataFrame([
    {
        "Target": "HomeGoals",
        "Intercept": home_model.intercept_,
        "EloDiffCoefficient": home_model.coef_[0],
    },
    {
        "Target": "AwayGoals",
        "Intercept": away_model.intercept_,
        "EloDiffCoefficient": away_model.coef_[0],
    },
])

model_summary.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print(f"Saved to: {OUTPUT_PATH}")