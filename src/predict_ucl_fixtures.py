import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import poisson


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

STRENGTH_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ucl_current_strength.csv"
)

FIXTURES_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "ucl"
    / "fixtures_2627.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ucl_predictions.csv"
)


# ---------------------------------------------------------
# Learned goal model
# ---------------------------------------------------------

HOME_INTERCEPT = 0.3910
HOME_ELO_COEF = 0.1837

AWAY_INTERCEPT = 0.1013
AWAY_ELO_COEF = -0.1795


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

strength = pd.read_csv(STRENGTH_PATH)
fixtures = pd.read_csv(FIXTURES_PATH)


# ---------------------------------------------------------
# Current strength lookup
# ---------------------------------------------------------

current_strength = dict(
    zip(
        strength["Team"],
        strength["CurrentStrength"]
    )
)


# ---------------------------------------------------------
# Check all fixture teams exist
# ---------------------------------------------------------

fixture_teams = (
    set(fixtures["home"])
    | set(fixtures["away"])
)

missing = [
    team
    for team in fixture_teams
    if team not in current_strength
]

if missing:
    raise ValueError(
        f"Missing current strength for: {missing}"
    )


# ---------------------------------------------------------
# Poisson probability calculation
# ---------------------------------------------------------

def calculate_probabilities(
    home_xg,
    away_xg,
    max_goals=10
):

    goals = np.arange(
        max_goals + 1
    )

    home_pmf = poisson.pmf(
        goals,
        home_xg
    )

    away_pmf = poisson.pmf(
        goals,
        away_xg
    )

    score_matrix = np.outer(
        home_pmf,
        away_pmf
    )

    home_win = np.tril(
        score_matrix,
        -1
    ).sum()

    draw = np.trace(
        score_matrix
    )

    away_win = np.triu(
        score_matrix,
        1
    ).sum()

    total = (
        home_win
        + draw
        + away_win
    )

    return (
        home_win / total,
        draw / total,
        away_win / total
    )


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

predictions = []

for _, fixture in fixtures.iterrows():

    home = fixture["home"]
    away = fixture["away"]

    home_strength = current_strength[home]
    away_strength = current_strength[away]

    # Relative strength in 100-Elo units
    elo_difference = (
        home_strength - away_strength
    ) / 100

    # Expected goals
    home_xg = np.exp(
        HOME_INTERCEPT
        + HOME_ELO_COEF
        * elo_difference
    )

    away_xg = np.exp(
        AWAY_INTERCEPT
        + AWAY_ELO_COEF
        * elo_difference
    )

    (
        home_win,
        draw,
        away_win
    ) = calculate_probabilities(
        home_xg,
        away_xg
    )

    predictions.append({
        "gw": fixture["gw"],
        "date": fixture["date"],
        "home": home,
        "away": away,
        "HomeStrength": home_strength,
        "AwayStrength": away_strength,
        "EloDifference": (
            home_strength - away_strength
        ),
        "HomeXG": home_xg,
        "AwayXG": away_xg,
        "HomeWinProb": home_win,
        "DrawProb": draw,
        "AwayWinProb": away_win,
    })


# ---------------------------------------------------------
# DataFrame
# ---------------------------------------------------------

predictions = pd.DataFrame(
    predictions
)


# ---------------------------------------------------------
# Most likely outcome
# ---------------------------------------------------------

def predicted_outcome(row):

    probabilities = {
        "H": row["HomeWinProb"],
        "D": row["DrawProb"],
        "A": row["AwayWinProb"],
    }

    return max(
        probabilities,
        key=probabilities.get
    )


predictions["PredictedOutcome"] = (
    predictions.apply(
        predicted_outcome,
        axis=1
    )
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

predictions.to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print()
print("UPCOMING UCL FIXTURE PREDICTIONS")
print("--------------------------------")

display_columns = [
    "gw",
    "home",
    "away",
    "HomeXG",
    "AwayXG",
    "HomeWinProb",
    "DrawProb",
    "AwayWinProb",
    "PredictedOutcome",
]

print(
    predictions[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "HomeXG": "{:.2f}".format,
            "AwayXG": "{:.2f}".format,
            "HomeWinProb": "{:.1%}".format,
            "DrawProb": "{:.1%}".format,
            "AwayWinProb": "{:.1%}".format,
        }
    )
)

print()
print(f"Saved to: {OUTPUT_PATH}")