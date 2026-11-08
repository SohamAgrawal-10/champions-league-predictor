import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

train = pd.read_csv("data/processed/train_v2.csv")
validation = pd.read_csv("data/processed/validation_v2.csv")
test = pd.read_csv("data/processed/test_v2.csv")


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

feature_columns = [
    # Elo
    "HomeElo",
    "AwayElo",
    "EloDiff",

    # Recent points
    "HomeForm5Points",
    "AwayForm5Points",
    "Form5PointsDiff",

    # Goals scored
    "HomeForm5GoalsScored",
    "AwayForm5GoalsScored",
    "Form5GoalsScoredDiff",

    # Goals conceded
    "HomeForm5GoalsConceded",
    "AwayForm5GoalsConceded",
    "Form5GoalsConcededDiff",

    # Goal difference
    "HomeForm5GoalDifference",
    "AwayForm5GoalDifference",
    "Form5GoalDifferenceDiff",

    # Win rate
    "HomeForm5WinRate",
    "AwayForm5WinRate",
    "Form5WinRateDiff",
]


X_train = train[feature_columns]
y_train = train["FTResult"]

X_validation = validation[feature_columns]
y_validation = validation["FTResult"]

X_test = test[feature_columns]
y_test = test["FTResult"]


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000
    ))
])


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

validation_predictions = model.predict(X_validation)
validation_probabilities = model.predict_proba(X_validation)

print("=== V2 VALIDATION RESULTS ===")

print(
    f"Accuracy: "
    f"{accuracy_score(y_validation, validation_predictions):.4f}"
)

print(
    f"Log Loss: "
    f"{log_loss(y_validation, validation_probabilities, labels=model.classes_):.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_validation,
        validation_predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_validation,
        validation_predictions
    )
)


# --------------------------------------------------
# TEST
# --------------------------------------------------

test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)

print("\n\n=== V2 TEST RESULTS ===")

print(
    f"Accuracy: "
    f"{accuracy_score(y_test, test_predictions):.4f}"
)

print(
    f"Log Loss: "
    f"{log_loss(y_test, test_probabilities, labels=model.classes_):.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


# --------------------------------------------------
# SAMPLE PROBABILITIES
# --------------------------------------------------

print("\n\n=== V2 SAMPLE PREDICTIONS ===")

sample = test.iloc[:10]

sample_probabilities = model.predict_proba(
    sample[feature_columns]
)

for i, (_, row) in enumerate(sample.iterrows()):

    probabilities = sample_probabilities[i]

    print(
        f"{row['HomeTeam']} vs {row['AwayTeam']} "
        f"-> "
        f"A: {probabilities[0]:.3f}, "
        f"D: {probabilities[1]:.3f}, "
        f"H: {probabilities[2]:.3f}"
    )