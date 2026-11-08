import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix
)

# Load data
train = pd.read_csv("data/processed/train_v3.csv")
validation = pd.read_csv("data/processed/validation_v3.csv")
test = pd.read_csv("data/processed/test_v3.csv")

# Features
feature_columns = [
    "OurHomeElo",
    "OurAwayElo",
    "OurEloDiff",

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
]

X_train = train[feature_columns]
y_train = train["FTResult"]

X_validation = validation[feature_columns]
y_validation = validation["FTResult"]

X_test = test[feature_columns]
y_test = test["FTResult"]

# Model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])

# Train
model.fit(X_train, y_train)

# -------------------------
# VALIDATION
# -------------------------

validation_predictions = model.predict(X_validation)
validation_probabilities = model.predict_proba(X_validation)

print("=== VALIDATION RESULTS ===")

print(
    "Accuracy:",
    round(accuracy_score(y_validation, validation_predictions), 4)
)

print(
    "Log Loss:",
    round(log_loss(y_validation, validation_probabilities), 4)
)

print("\nClassification Report:")
print(
    classification_report(
        y_validation,
        validation_predictions,
        labels=["A", "D", "H"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_validation,
        validation_predictions,
        labels=["A", "D", "H"]
    )
)

# -------------------------
# TEST
# -------------------------

test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)

print("\n\n=== TEST RESULTS ===")

print(
    "Accuracy:",
    round(accuracy_score(y_test, test_predictions), 4)
)

print(
    "Log Loss:",
    round(log_loss(y_test, test_probabilities), 4)
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=["A", "D", "H"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=["A", "D", "H"]
    )
)

# -------------------------
# SAMPLE PROBABILITIES
# -------------------------

print("\n\n=== SAMPLE PREDICTIONS ===")

samples = test.iloc[:10].copy()

sample_probabilities = model.predict_proba(
    samples[feature_columns]
)

classes = model.named_steps["classifier"].classes_

for i, (_, match) in enumerate(samples.iterrows()):
    probabilities = dict(
        zip(classes, sample_probabilities[i])
    )

    print(
        f"{match['HomeTeam']} vs {match['AwayTeam']} "
        f"-> "
        f"A: {probabilities.get('A', 0):.3f}, "
        f"D: {probabilities.get('D', 0):.3f}, "
        f"H: {probabilities.get('H', 0):.3f}"
    )