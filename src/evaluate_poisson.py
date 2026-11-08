import pandas as pd
import numpy as np

from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/goal_features.csv")
df["MatchDate"] = pd.to_datetime(df["MatchDate"])

df = df.dropna(
    subset=["OurHomeElo", "OurAwayElo"]
).copy()


# ============================================================
# SPLIT
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
# TRAIN MODELS
# ============================================================

from sklearn.linear_model import PoissonRegressor

X_train = train[feature_columns]

home_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000
)

away_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000
)

home_model.fit(X_train, train["FTHome"])
away_model.fit(X_train, train["FTAway"])


# ============================================================
# CONVERT EXPECTED GOALS → H/D/A PROBABILITIES
# ============================================================

def outcome_probabilities(home_xg, away_xg, max_goals=10):

    home_goals = np.arange(max_goals + 1)
    away_goals = np.arange(max_goals + 1)

    home_probs = poisson.pmf(
        home_goals,
        home_xg
    )

    away_probs = poisson.pmf(
        away_goals,
        away_xg
    )

    # Score probability matrix
    score_matrix = np.outer(
        home_probs,
        away_probs
    )

    home_win = np.tril(
        score_matrix,
        k=-1
    ).sum()

    draw = np.trace(score_matrix)

    away_win = np.triu(
        score_matrix,
        k=1
    ).sum()

    total = home_win + draw + away_win

    return (
        home_win / total,
        draw / total,
        away_win / total
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(dataset, name):

    X = dataset[feature_columns]

    home_xg = home_model.predict(X)
    away_xg = away_model.predict(X)

    probabilities = []

    for h_xg, a_xg in zip(
        home_xg,
        away_xg
    ):

        home_win, draw, away_win = outcome_probabilities(
            h_xg,
            a_xg
        )

        probabilities.append(
            [
                away_win,
                draw,
                home_win
            ]
        )

    probabilities = np.array(probabilities)

    # Classes must match:
    # A = away
    # D = draw
    # H = home

    classes = np.array(["A", "D", "H"])

    predictions = classes[
        probabilities.argmax(axis=1)
    ]

    actual = dataset["FTResult"].values

    accuracy = accuracy_score(
        actual,
        predictions
    )

    loss = log_loss(
        actual,
        probabilities,
        labels=["A", "D", "H"]
    )

    print(f"\n=== {name.upper()} RESULTS ===")

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Log Loss:",
        round(loss, 4)
    )

    print("\nAverage probabilities:")

    print(
        f"Away:  {probabilities[:, 0].mean():.3f}"
    )

    print(
        f"Draw:  {probabilities[:, 1].mean():.3f}"
    )

    print(
        f"Home:  {probabilities[:, 2].mean():.3f}"
    )

    return probabilities


# ============================================================
# VALIDATION
# ============================================================

validation_probabilities = evaluate(
    validation,
    "Validation"
)


# ============================================================
# TEST
# ============================================================

test_probabilities = evaluate(
    test,
    "Test"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n=== SAMPLE PREDICTIONS ===")

samples = test.iloc[:10].copy()

sample_home_xg = home_model.predict(
    samples[feature_columns]
)

sample_away_xg = away_model.predict(
    samples[feature_columns]
)

for i, (_, match) in enumerate(
    samples.iterrows()
):

    home_win, draw, away_win = outcome_probabilities(
        sample_home_xg[i],
        sample_away_xg[i]
    )

    print(
        f"{match['HomeTeam']} vs "
        f"{match['AwayTeam']} "
        f"-> "
        f"A: {away_win:.3f}, "
        f"D: {draw:.3f}, "
        f"H: {home_win:.3f}"
    )