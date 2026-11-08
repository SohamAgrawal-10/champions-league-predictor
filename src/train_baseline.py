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


# Load data
train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")


# Features
feature_columns = [
    "HomeElo",
    "AwayElo",
    "EloDiff",
    "Form3Home",
    "Form3Away",
    "Form3Diff",
    "Form5Home",
    "Form5Away",
    "Form5Diff",
]


X_train = train[feature_columns]
y_train = train["FTResult"]

X_validation = validation[feature_columns]
y_validation = validation["FTResult"]

X_test = test[feature_columns]
y_test = test["FTResult"]


# Logistic Regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000
    ))
])


# Train
model.fit(X_train, y_train)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

validation_predictions = model.predict(X_validation)
validation_probabilities = model.predict_proba(X_validation)

print("=== VALIDATION RESULTS ===")

print(f"Accuracy: {accuracy_score(y_validation, validation_predictions):.4f}")

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
print(confusion_matrix(y_validation, validation_predictions))


# --------------------------------------------------
# TEST
# --------------------------------------------------

test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)

print("\n\n=== TEST RESULTS ===")

print(f"Accuracy: {accuracy_score(y_test, test_predictions):.4f}")

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
print(confusion_matrix(y_test, test_predictions))


# --------------------------------------------------
# COEFFICIENTS
# --------------------------------------------------

classifier = model.named_steps["classifier"]

print("\n\n=== MODEL COEFFICIENTS ===")

coefficients = pd.DataFrame(
    classifier.coef_,
    columns=feature_columns,
    index=classifier.classes_
)

print(coefficients.round(4))


# --------------------------------------------------
# SAMPLE PROBABILITIES
# --------------------------------------------------

print("\n\n=== SAMPLE PREDICTIONS ===")

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