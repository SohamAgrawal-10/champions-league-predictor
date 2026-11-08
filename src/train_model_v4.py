import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Load data
train = pd.read_csv("data/processed/train_v3.csv")
validation = pd.read_csv("data/processed/validation_v3.csv")
test = pd.read_csv("data/processed/test_v3.csv")

# Load opponent-strength features
opponent = pd.read_csv(
    "data/processed/opponent_strength_features.csv"
)

opponent["MatchDate"] = pd.to_datetime(opponent["MatchDate"])

# Merge using match identity
merge_columns = [
    "MatchDate",
    "HomeTeam",
    "AwayTeam",
]

opponent_features = opponent[
    merge_columns
    + [
        "HomeOpponentElo5",
        "AwayOpponentElo5",
        "OpponentStrengthDiff",
    ]
].copy()

for dataset_name, dataset in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    dataset["MatchDate"] = pd.to_datetime(dataset["MatchDate"])

    merged = dataset.merge(
        opponent_features,
        on=merge_columns,
        how="left",
        validate="one_to_one",
    )

    if dataset_name == "train":
        train = merged
    elif dataset_name == "validation":
        validation = merged
    else:
        test = merged


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

    # New opponent-strength features
    "HomeOpponentElo5",
    "AwayOpponentElo5",
    "OpponentStrengthDiff",
]

X_train = train[feature_columns]
y_train = train["FTResult"]

X_validation = validation[feature_columns]
y_validation = validation["FTResult"]

X_test = test[feature_columns]
y_test = test["FTResult"]


# Model
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000
            ),
        ),
    ]
)

model.fit(X_train, y_train)


# Validation
validation_predictions = model.predict(X_validation)
validation_probabilities = model.predict_proba(X_validation)

print("=== VALIDATION RESULTS ===")

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


# Test
test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)

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


# Probability distribution
probability_df = pd.DataFrame(
    test_probabilities,
    columns=model.classes_
)

print("\n=== AVERAGE TEST PROBABILITIES ===")

print(
    probability_df.mean().round(4)
)