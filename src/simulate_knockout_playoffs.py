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
# Team strength
# =========================================================

strength = {}

for _, row in form.iterrows():

    strength[row["Team"]] = float(
        row["PointsPerGame"]
    )


# =========================================================
# Prediction lookup
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


# =========================================================
# Current standings
# =========================================================

current_standings = {}

for _, row in form.iterrows():

    current_standings[row["Team"]] = {
        "points": int(row["Points"]),
        "goals_for": int(row["GoalsFor"]),
        "goals_against": int(row["GoalsAgainst"]),
    }


# =========================================================
# Simulate a match
# =========================================================

def simulate_match(home, away):

    # -----------------------------------------------------
    # First use a model prediction if this exact fixture
    # exists in the league-phase predictions.
    #
    # For knockout matches that don't exist there, create
    # an approximate xG from current team strength.
    # -----------------------------------------------------

    key = (home, away)

    if key in prediction_lookup:

        home_xg = prediction_lookup[key]["home_xg"]
        away_xg = prediction_lookup[key]["away_xg"]

    else:

        # Use Points Per Game as a simple relative strength
        # measure for knockout matches.
        #
        # This is deliberately simple for V1. We will later
        # replace this with the project's CurrentStrength
        # model.

        home_strength = strength[home]
        away_strength = strength[away]

        strength_difference = (
            home_strength
            - away_strength
        )

        # Base expected goals

        home_xg = 1.55 + (
            strength_difference * 0.35
        )

        away_xg = 1.15 - (
            strength_difference * 0.20
        )

        # Prevent unrealistic negative xG

        home_xg = max(
            0.25,
            home_xg
        )

        away_xg = max(
            0.20,
            away_xg
        )

    home_goals = rng.poisson(
        home_xg
    )

    away_goals = rng.poisson(
        away_xg
    )

    return home_goals, away_goals


# =========================================================
# Simulate two-legged tie
# =========================================================

def simulate_two_legged_tie(
    seeded,
    unseeded
):

    # -----------------------------------------------------
    # First leg:
    #
    # Unseeded team hosts first.
    # -----------------------------------------------------

    first_home = unseeded
    first_away = seeded

    first_home_goals, first_away_goals = (
        simulate_match(
            first_home,
            first_away
        )
    )


    # -----------------------------------------------------
    # Second leg:
    #
    # Seeded team hosts second.
    # -----------------------------------------------------

    second_home = seeded
    second_away = unseeded

    second_home_goals, second_away_goals = (
        simulate_match(
            second_home,
            second_away
        )
    )


    # -----------------------------------------------------
    # Aggregate score
    # -----------------------------------------------------

    seeded_goals = (
        first_away_goals
        + second_home_goals
    )

    unseeded_goals = (
        first_home_goals
        + second_away_goals
    )


    # -----------------------------------------------------
    # Determine winner
    # -----------------------------------------------------

    if seeded_goals > unseeded_goals:

        return seeded

    if unseeded_goals > seeded_goals:

        return unseeded


    # -----------------------------------------------------
    # Aggregate draw
    #
    # Resolve by relative strength.
    # -----------------------------------------------------

    seeded_strength = strength[seeded]

    unseeded_strength = strength[unseeded]

    strength_gap = (
        seeded_strength
        - unseeded_strength
    )

    seeded_probability = (
        0.50
        + 0.20 * np.tanh(
            strength_gap
        )
    )

    if rng.random() < seeded_probability:

        return seeded

    return unseeded


# =========================================================
# Create playoff draw
# =========================================================

def create_playoff_draw(
    seeded_teams,
    unseeded_teams
):

    # -----------------------------------------------------
    # Shuffle both groups.
    #
    # V1 deliberately uses a random valid pairing.
    # UEFA's exact draw constraints can be implemented
    # later.
    # -----------------------------------------------------

    seeded = list(seeded_teams)

    unseeded = list(unseeded_teams)

    rng.shuffle(seeded)

    rng.shuffle(unseeded)

    pairings = []

    for i in range(8):

        pairings.append(
            (
                seeded[i],
                unseeded[i]
            )
        )

    return pairings


# =========================================================
# Test one playoff draw
# =========================================================

teams = list(
    current_standings.keys()
)

print()
print(f"Teams: {len(teams)}")
print(
    f"Simulations: {N_SIMULATIONS:,}"
)

print()
print("Testing playoff simulation...")


# =========================================================
# Result storage
# =========================================================

playoff_advance_counts = {
    team: 0
    for team in teams
}

playoff_elimination_counts = {
    team: 0
    for team in teams
}


# =========================================================
# Run simulations
# =========================================================

for simulation in range(
    N_SIMULATIONS
):

    # -----------------------------------------------------
    # Build a simulated league table
    #
    # For this standalone test, we use the actual current
    # league standings and randomly perturb them slightly.
    #
    # The full version will receive the simulated league
    # table directly from simulate_league_phase.py.
    # -----------------------------------------------------

    table_rows = []

    for team, stats in current_standings.items():

        table_rows.append({
            "Team": team,
            "Points": stats["points"]
            + rng.integers(
                0,
                10
            ),
            "GoalDifference": (
                stats["goals_for"]
                - stats["goals_against"]
                + rng.integers(
                    -5,
                    6
                )
            ),
            "GoalsFor": (
                stats["goals_for"]
                + rng.integers(
                    0,
                    6
                )
            ),
        })

    table = pd.DataFrame(
        table_rows
    )

    table = table.sort_values(
        by=[
            "Points",
            "GoalDifference",
            "GoalsFor",
        ],
        ascending=False
    ).reset_index(
        drop=True
    )


    # -----------------------------------------------------
    # Extract positions
    # -----------------------------------------------------

    seeded = table.iloc[
        8:16
    ]["Team"].tolist()

    unseeded = table.iloc[
        16:24
    ]["Team"].tolist()


    # -----------------------------------------------------
    # Create draw
    # -----------------------------------------------------

    draw = create_playoff_draw(
        seeded,
        unseeded
    )


    # -----------------------------------------------------
    # Simulate all eight ties
    # -----------------------------------------------------

    for seeded_team, unseeded_team in draw:

        winner = simulate_two_legged_tie(
            seeded_team,
            unseeded_team
        )

        loser = (
            unseeded_team
            if winner == seeded_team
            else seeded_team
        )

        playoff_advance_counts[
            winner
        ] += 1

        playoff_elimination_counts[
            loser
        ] += 1


# =========================================================
# Build results
# =========================================================

results = []

for team in teams:

    appearances = (
        playoff_advance_counts[team]
        + playoff_elimination_counts[team]
    )

    if appearances == 0:

        advance_probability = 0.0
        elimination_probability = 0.0

    else:

        advance_probability = (
            playoff_advance_counts[team]
            / appearances
        )

        elimination_probability = (
            playoff_elimination_counts[team]
            / appearances
        )

    results.append({
        "Team": team,
        "PlayoffAppearances": appearances,
        "AdvanceProbability": (
            advance_probability
        ),
        "EliminationProbability": (
            elimination_probability
        ),
    })


results = pd.DataFrame(
    results
)


# =========================================================
# Display results
# =========================================================

results = results[
    results["PlayoffAppearances"] > 0
].sort_values(
    by="AdvanceProbability",
    ascending=False
)


print()
print("KNOCKOUT PLAYOFF SIMULATION")
print("============================")

display_df = results.copy()

display_df[
    "AdvanceProbability"
] = display_df[
    "AdvanceProbability"
].map(
    lambda x: f"{x:.1%}"
)

display_df[
    "EliminationProbability"
] = display_df[
    "EliminationProbability"
].map(
    lambda x: f"{x:.1%}"
)

print(
    display_df.to_string(
        index=False
    )
)


print()
print("Playoff simulation completed.")