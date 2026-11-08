import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FORM_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ucl_form.csv"
)

FIXTURES_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "ucl"
    / "fixtures_2627.csv"
)

PREDICTIONS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ucl_predictions.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "league_simulation_results.csv"
)


# =========================================================
# Simulation settings
# =========================================================

N_SIMULATIONS = 10_000

RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# =========================================================
# Load data
# =========================================================

form = pd.read_csv(FORM_PATH)

fixtures = pd.read_csv(FIXTURES_PATH)

predictions = pd.read_csv(PREDICTIONS_PATH)


# =========================================================
# Teams
# =========================================================

teams = form["Team"].tolist()

print()
print(f"Teams: {len(teams)}")
print(f"Simulations: {N_SIMULATIONS:,}")


# =========================================================
# Current standings
# =========================================================

standings = {}

for team in teams:

    row = form[
        form["Team"] == team
    ].iloc[0]

    standings[team] = {
        "points": int(row["Points"]),
        "goals_for": int(row["GoalsFor"]),
        "goals_against": int(row["GoalsAgainst"]),
    }


# =========================================================
# Remaining fixtures
# =========================================================

remaining_fixtures = fixtures.copy()

print(
    f"Remaining fixtures: "
    f"{len(remaining_fixtures)}"
)


# =========================================================
# Check prediction coverage
# =========================================================

prediction_lookup = {}

for _, row in predictions.iterrows():

    key = (
        row["home"],
        row["away"]
    )

    prediction_lookup[key] = {
        "home_xg": float(row["HomeXG"]),
        "away_xg": float(row["AwayXG"]),
    }


missing_predictions = []

for _, fixture in remaining_fixtures.iterrows():

    key = (
        fixture["home"],
        fixture["away"]
    )

    if key not in prediction_lookup:

        missing_predictions.append(key)


if missing_predictions:

    raise ValueError(
        "Missing predictions for:\n"
        + "\n".join(
            f"{home} vs {away}"
            for home, away in missing_predictions
        )
    )


# =========================================================
# Simulate one match
# =========================================================

def simulate_match(home, away):

    prediction = prediction_lookup[
        (home, away)
    ]

    home_xg = prediction["home_xg"]
    away_xg = prediction["away_xg"]

    # Sample actual goals from the Poisson
    # distributions implied by the model.

    home_goals = rng.poisson(
        home_xg
    )

    away_goals = rng.poisson(
        away_xg
    )

    return home_goals, away_goals


# =========================================================
# Rank standings
# =========================================================

def rank_standings(table):

    rows = []

    for team, stats in table.items():

        goal_difference = (
            stats["goals_for"]
            - stats["goals_against"]
        )

        rows.append({
            "Team": team,
            "Points": stats["points"],
            "GoalDifference": goal_difference,
            "GoalsFor": stats["goals_for"],
        })

    standings_df = pd.DataFrame(rows)

    # First-version tiebreakers:
    #
    # 1. Points
    # 2. Goal difference
    # 3. Goals scored
    #
    # UEFA's full tiebreaking procedure will be
    # implemented later when we build the final
    # tournament simulator.

    standings_df = standings_df.sort_values(
        by=[
            "Points",
            "GoalDifference",
            "GoalsFor",
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    standings_df["Position"] = (
        np.arange(
            len(standings_df)
        ) + 1
    )

    return standings_df


# =========================================================
# Storage for simulation results
# =========================================================

position_counts = {
    team: np.zeros(
        len(teams),
        dtype=np.int32
    )
    for team in teams
}

top8_counts = {
    team: 0
    for team in teams
}

qualification_counts = {
    team: 0
    for team in teams
}

eliminated_counts = {
    team: 0
    for team in teams
}


# =========================================================
# Run Monte Carlo simulation
# =========================================================

print()
print("Running simulations...")


for simulation in range(
    N_SIMULATIONS
):

    # -----------------------------------------------------
    # Start from the actual current standings
    # -----------------------------------------------------

    table = {
        team: {
            "points": stats["points"],
            "goals_for": stats["goals_for"],
            "goals_against": stats["goals_against"],
        }
        for team, stats in standings.items()
    }


    # -----------------------------------------------------
    # Simulate every remaining fixture
    # -----------------------------------------------------

    for _, fixture in remaining_fixtures.iterrows():

        home = fixture["home"]
        away = fixture["away"]

        home_goals, away_goals = simulate_match(
            home,
            away
        )


        # -------------------------------------------------
        # Update goals
        # -------------------------------------------------

        table[home]["goals_for"] += home_goals

        table[home]["goals_against"] += away_goals

        table[away]["goals_for"] += away_goals

        table[away]["goals_against"] += home_goals


        # -------------------------------------------------
        # Update points
        # -------------------------------------------------

        if home_goals > away_goals:

            table[home]["points"] += 3

        elif away_goals > home_goals:

            table[away]["points"] += 3

        else:

            table[home]["points"] += 1

            table[away]["points"] += 1


    # -----------------------------------------------------
    # Rank final simulated table
    # -----------------------------------------------------

    final_table = rank_standings(
        table
    )


    # -----------------------------------------------------
    # Record results
    # -----------------------------------------------------

    for _, row in final_table.iterrows():

        team = row["Team"]

        position = int(
            row["Position"]
        )


        # Position distribution

        position_counts[
            team
        ][position - 1] += 1


        # Top 8 = direct knockout qualification

        if position <= 8:

            top8_counts[team] += 1


        # Positions 1-24 = knockout phase

        if position <= 24:

            qualification_counts[team] += 1


        # Positions 25-36 = eliminated

        if position >= 25:

            eliminated_counts[team] += 1


# =========================================================
# Build summary
# =========================================================

summary = []


for team in teams:

    position_distribution = (
        position_counts[team]
        / N_SIMULATIONS
    )


    # Expected finishing position

    expected_position = sum(
        (i + 1) * probability
        for i, probability
        in enumerate(
            position_distribution
        )
    )


    row = {
        "Team": team,

        "ExpectedPosition":
            expected_position,

        "Top8Probability":
            top8_counts[team]
            / N_SIMULATIONS,

        "KnockoutQualificationProbability":
            qualification_counts[team]
            / N_SIMULATIONS,

        "EliminationProbability":
            eliminated_counts[team]
            / N_SIMULATIONS,
    }


    # Probability of every finishing position

    for i in range(
        len(teams)
    ):

        row[
            f"Position{i + 1}Probability"
        ] = position_distribution[i]


    summary.append(row)


summary = pd.DataFrame(
    summary
)


# =========================================================
# Sort by expected position
# =========================================================

summary = summary.sort_values(
    by="ExpectedPosition"
).reset_index(
    drop=True
)


# =========================================================
# Save results
# =========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# Display results
# =========================================================

print()
print("LEAGUE-PHASE SIMULATION")
print("========================")


display_columns = [
    "Team",
    "ExpectedPosition",
    "Top8Probability",
    "KnockoutQualificationProbability",
    "EliminationProbability",
]


display_df = summary[
    display_columns
].copy()


display_df["ExpectedPosition"] = (
    display_df[
        "ExpectedPosition"
    ].map(
        lambda x: f"{x:.2f}"
    )
)


for column in [
    "Top8Probability",
    "KnockoutQualificationProbability",
    "EliminationProbability",
]:

    display_df[column] = (
        display_df[column].map(
            lambda x: f"{x:.1%}"
        )
    )


print(
    display_df.to_string(
        index=False
    )
)


print()
print(
    f"Saved to: {OUTPUT_PATH}"
)