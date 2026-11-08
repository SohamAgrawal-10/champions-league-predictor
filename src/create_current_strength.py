import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

ELO_PATH = BASE_DIR / "data" / "processed" / "ucl_final_elo.csv"
FORM_PATH = BASE_DIR / "data" / "processed" / "ucl_form.csv"

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ucl_current_strength.csv"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

elo = pd.read_csv(ELO_PATH)
form = pd.read_csv(FORM_PATH)


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

df = elo.merge(
    form,
    on="Team",
    how="inner"
)


# ---------------------------------------------------------
# Standardize UCL performance
#
# We convert current UCL performance into a z-score.
#
# Points per game captures results.
# Goal difference per game captures dominance.
# ---------------------------------------------------------

ppg_z = (
    df["PointsPerGame"] - df["PointsPerGame"].mean()
) / df["PointsPerGame"].std()

gd_z = (
    df["GoalDifferencePerGame"]
    - df["GoalDifferencePerGame"].mean()
) / df["GoalDifferencePerGame"].std()


# ---------------------------------------------------------
# UCL performance score
#
# Results are slightly more important than goal difference.
# ---------------------------------------------------------

df["UCLPerformanceScore"] = (
    0.65 * ppg_z
    + 0.35 * gd_z
)


# ---------------------------------------------------------
# Convert performance into Elo points
#
# 40 Elo points represents a meaningful but controlled
# adjustment after only four matches.
#
# The adjustment is deliberately capped at +/- 80.
# ---------------------------------------------------------

df["UCLAdjustment"] = (
    40 * df["UCLPerformanceScore"]
)

df["UCLAdjustment"] = (
    df["UCLAdjustment"]
    .clip(-80, 80)
)


# ---------------------------------------------------------
# Current strength
# ---------------------------------------------------------

df["CurrentStrength"] = (
    df["Elo"] + df["UCLAdjustment"]
)


# ---------------------------------------------------------
# Rank
# ---------------------------------------------------------

df = df.sort_values(
    "CurrentStrength",
    ascending=False
).reset_index(drop=True)

df["CurrentStrengthRank"] = (
    range(1, len(df) + 1)
)


# ---------------------------------------------------------
# Select columns
# ---------------------------------------------------------

df = df[
    [
        "CurrentStrengthRank",
        "Team",
        "Elo",
        "PointsPerGame",
        "GoalDifferencePerGame",
        "UCLPerformanceScore",
        "UCLAdjustment",
        "CurrentStrength",
    ]
]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("CURRENT UCL TEAM STRENGTH")
print("-------------------------")

print(
    df.to_string(
        index=False,
        formatters={
            "UCLPerformanceScore": "{:.3f}".format,
            "UCLAdjustment": "{:.1f}".format,
            "CurrentStrength": "{:.1f}".format,
        }
    )
)

print()
print(f"Saved to: {OUTPUT_PATH}")