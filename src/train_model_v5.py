import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# 1. LOAD TRAIN / VALIDATION / TEST DATA
# ============================================================

train = pd.read_csv("data/processed/train_v3.csv")
validation = pd.read_csv("data/processed/validation_v3.csv")
test = pd.read_csv("data/processed/test_v3.csv")

for dataset in [train, validation, test]:
    dataset["MatchDate"] = pd.to_datetime(dataset["MatchDate"])


# ============================================================
# 2. LOAD ENGINEERED FEATURES
# ============================================================

features = pd.read_csv(
    "data/processed/home_away_features.csv"
)

features["MatchDate"] = pd.to_datetime(features["MatchDate"])


# We only need the newly engineered features.
# The existing V3 datasets already contain:
# - Dynamic Elo
# - Rolling form
#
# The home/away dataset contains:
# - Opponent strength
# - Home/away-specific form

merge_columns = [
    "MatchDate",
    "HomeTeam",
    "AwayTeam",
]

new_features = features[
    merge_columns
    + [
        # Opponent strength
        "HomeOpponentElo5",
        "AwayOpponentElo5",
        "OpponentStrengthDiff",

        # Home/away-specific form
        "HomeTeamHomeForm5Points",
        "AwayTeamAwayForm5Points",

        "HomeTeamHomeGoalsScored5",
        "AwayTeamAwayGoalsScored5",

        "HomeTeamHomeGoalsConceded5",
        "AwayTeamAwayGoalsConceded5",

        "HomeTeamHomeGoalDifference5",
        "AwayTeamAwayGoalDifference5",

        "HomeTeamHomeWinRate5",
        "AwayTeamAwayWinRate5",

        "HomeAwayPointsDiff",
        "HomeAwayGoalDifferenceDiff",
        "HomeAwayWinRateDiff",
    ]
].copy()


# ============================================================
# 3. MERGE FEATURES INTO EACH DATASET
# ============================================================

train = train.merge(
    new_features,
    on=merge_columns,
    how="left",
    validate="one_to_one",
)

validation = validation.merge(
    new_features,
    on=merge_columns,
    how="left",
    validate="one_to_one",
)

test = test.merge(
    new_features,
    on=merge_columns,
    how="left",
    validate="one_to_one",
)


# Check for failed merges
new_feature_columns = [
    "HomeOpponentElo5",
    "AwayOpponentElo5",
    "OpponentStrengthDiff",

    "HomeTeamHomeForm5Points",
    "AwayTeamAwayForm5Points",

    "HomeTeamHomeGoalsScored5",
    "AwayTeamAwayGoalsScored5",

    "HomeTeamHomeGoalsConceded5",
    "AwayTeamAwayGoalsConceded5",

    "HomeTeamHomeGoalDifference5",
    "AwayTeamAwayGoalDifference5",

    "HomeTeamHomeWinRate5",
    "AwayTeamAwayWinRate5",

    "HomeAwayPointsDiff",
    "HomeAwayGoalDifferenceDiff",
    "HomeAwayWinRateDiff",
]

print("=== MERGE CHECK ===")

print(
    "Missing engineered features:",
    train[new_feature_columns].isna().sum().sum(),
    validation[new_feature_columns].isna().sum().sum(),
    test[new_feature_columns].isna().sum().sum(),
)


# ============================================================
# 4. DEFINE FEATURES
# ============================================================

feature_columns = [

    # --------------------------------------------------------
    # Dynamic Elo
    # --------------------------------------------------------

    "OurHomeElo",
    "OurAwayElo",
    "OurEloDiff",

    # --------------------------------------------------------
    # General rolling form
    # --------------------------------------------------------

    "HomeForm5Points",
    "AwayForm5Points",
    "Form5PointsDiff",

    "HomeForm5GoalsScored",
    "AwayForm5GoalsScored",
    "Form5GoalsScoredDiff",

    "HomeForm5GoalsConceded",
    "AwayForm5GoalsConceded",
    "Form5GoalsConcededDiff",

    "HomeForm5GoalDifference",
    "AwayForm5GoalDifference",
    "Form5GoalDifferenceDiff",

    "HomeForm5WinRate",
    "AwayForm5WinRate",
    "Form5WinRateDiff",

    # --------------------------------------------------------
    # Opponent strength
    # --------------------------------------------------------

    "HomeOpponentElo5",
    "AwayOpponentElo5",
    "OpponentStrengthDiff",

    # --------------------------------------------------------
    # Home / away-specific form
    # --------------------------------------------------------

    "HomeTeamHomeForm5Points",
    "AwayTeamAwayForm5Points",

    "HomeTeamHomeGoalsScored5",
    "AwayTeamAwayGoalsScored5",

    "HomeTeamHomeGoalsConceded5",
    "AwayTeamAwayGoalsConceded5",

    "HomeTeamHomeGoalDifference5",
    "AwayTeamAwayGoalDifference5",

    "HomeTeamHomeWinRate5",
    "AwayTeamAwayWinRate5",

    "HomeAwayPointsDiff",
    "HomeAwayGoalDifferenceDiff",
    "HomeAwayWinRateDiff",
]


# ============================================================
# 5. PREPARE TRAINING DATA
# ============================================================

X_train = train[feature_columns]
y_train = train["FTResult"]

X_validation = validation[feature_columns]
y_validation = validation["FTResult"]

X_test = test[feature_columns]
y_test = test["FTResult"]


# ============================================================
# 6. BUILD MODEL
# ============================================================

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000
            )
        ),
    ]
)


# ============================================================
# 7. TRAIN
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# 8. VALIDATION EVALUATION
# ============================================================

validation_predictions = model.predict(
    X_validation
)

validation_probabilities = model.predict_proba(
    X_validation
)


print("\n=== VALIDATION RESULTS ===")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_validation,
            validation_predictions
        ),
        4
    )
)

print(
    "Log Loss:",
    round(
        log_loss(
            y_validation,
            validation_probabilities
        ),
        4
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        validation_predictions,
        zero_division=0
    )
)


# ============================================================
# 9. TEST EVALUATION
# ============================================================

test_predictions = model.predict(
    X_test
)

test_probabilities = model.predict_proba(
    X_test
)


print("\n=== TEST RESULTS ===")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            test_predictions
        ),
        4
    )
)

print(
    "Log Loss:",
    round(
        log_loss(
            y_test,
            test_probabilities
        ),
        4
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# ============================================================
# 10. AVERAGE PREDICTED PROBABILITIES
# ============================================================

probability_df = pd.DataFrame(
    test_probabilities,
    columns=model.classes_
)

print("\n=== AVERAGE TEST PROBABILITIES ===")

print(
    probability_df.mean().round(4)
)


# ============================================================
# 11. FEATURE COUNT
# ============================================================

print("\n=== MODEL INFORMATION ===")

print(
    f"Number of features: {len(feature_columns)}"
)

print("\nFeatures:")

for feature in feature_columns:
    print(f"- {feature}")