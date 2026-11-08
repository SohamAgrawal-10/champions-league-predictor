import pandas as pd
from pathlib import Path


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "league_simulation_results.csv"
)


# =========================================================
# Load results
# =========================================================

results = pd.read_csv(RESULTS_PATH)


# =========================================================
# Validate probability totals
# =========================================================

print()
print("VALIDATING LEAGUE SIMULATION")
print("============================")


# ---------------------------------------------------------
# 1. Every team should have exactly one finishing position
# ---------------------------------------------------------

position_columns = [
    f"Position{i}Probability"
    for i in range(1, 37)
]

results["PositionProbabilityTotal"] = results[
    position_columns
].sum(axis=1)


print()
print("Position probability totals:")

print(
    results[
        ["Team", "PositionProbabilityTotal"]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# 2. Validate mutually exclusive outcome categories
# ---------------------------------------------------------
#
# Top 8:
#     positions 1-8
#
# Knockout Playoff:
#     positions 9-24
#
# Eliminated:
#     positions 25-36
#
# IMPORTANT:
# "KnockoutQualificationProbability" in our CSV
# means positions 1-24, so we must subtract Top 8
# to get positions 9-24.
# ---------------------------------------------------------

results["KnockoutPlayoffProbability"] = (
    results["KnockoutQualificationProbability"]
    - results["Top8Probability"]
)

results["OutcomeProbabilityTotal"] = (
    results["Top8Probability"]
    + results["KnockoutPlayoffProbability"]
    + results["EliminationProbability"]
)


print()
print("Mutually exclusive outcome probability totals:")

print(
    results[
        [
            "Team",
            "Top8Probability",
            "KnockoutPlayoffProbability",
            "EliminationProbability",
            "OutcomeProbabilityTotal",
        ]
    ].to_string(index=False)
)


# =========================================================
# Expected position validation
# =========================================================

print()
print("Expected position range:")

print(
    f"Minimum: "
    f"{results['ExpectedPosition'].min():.2f}"
)

print(
    f"Maximum: "
    f"{results['ExpectedPosition'].max():.2f}"
)


# =========================================================
# Validation checks
# =========================================================

position_valid = (
    results["PositionProbabilityTotal"]
    .between(0.999, 1.001)
    .all()
)


outcome_valid = (
    results["OutcomeProbabilityTotal"]
    .between(0.999, 1.001)
    .all()
)


expected_position_valid = (
    results["ExpectedPosition"]
    .between(1, 36)
    .all()
)


# =========================================================
# Additional consistency checks
# =========================================================

# Top 8 cannot be greater than positions 1-24
top8_subset_valid = (
    results["Top8Probability"]
    <= results["KnockoutQualificationProbability"]
).all()


# Elimination is equivalent to positions 25-36
elimination_probability = (
    results[
        [
            f"Position{i}Probability"
            for i in range(25, 37)
        ]
    ].sum(axis=1)
)


elimination_valid = (
    abs(
        elimination_probability
        - results["EliminationProbability"]
    )
    < 1e-9
).all()


# =========================================================
# Print checks
# =========================================================

print()
print("CHECKS")
print("------")

print(
    f"Position probabilities sum to 100%: "
    f"{position_valid}"
)

print(
    f"Mutually exclusive outcomes sum to 100%: "
    f"{outcome_valid}"
)

print(
    f"Expected positions between 1 and 36: "
    f"{expected_position_valid}"
)

print(
    f"Top 8 is a subset of positions 1-24: "
    f"{top8_subset_valid}"
)

print(
    f"Elimination probability matches positions 25-36: "
    f"{elimination_valid}"
)


# =========================================================
# Final result
# =========================================================

if (
    position_valid
    and outcome_valid
    and expected_position_valid
    and top8_subset_valid
    and elimination_valid
):

    print()
    print("VALIDATION PASSED")
    print(
        "The league-phase simulation is ready "
        "for the knockout stage."
    )

else:

    print()
    print("VALIDATION FAILED")
    print(
        "Inspect the results before continuing."
    )