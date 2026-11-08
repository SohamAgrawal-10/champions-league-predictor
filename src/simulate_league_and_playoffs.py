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
    / "league_playoff_simulation_results.csv"
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
# Team list
# =========================================================

teams = form["Team"].tolist()


print()
print(f"Teams: {len(teams)}")
print(f"Simulations: {N_SIMULATIONS:,}")
print(f"Remaining fixtures: {len(fixtures)}")


# =========================================================
# Current standings
# =========================================================

initial_standings = {}

for _, row in form.iterrows():

    initial_standings[row["Team"]] = {
        "points": int(row["Points"]),
        "goals_for": int(row["GoalsFor"]),
        "goals_against": int(row["GoalsAgainst"]),
    }


# =========================================================
# Match prediction lookup
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
# Check prediction coverage
# =========================================================

missing_predictions = []

for _, fixture in fixtures.iterrows():

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
# Simulate a match using xG
# =========================================================

def simulate_match(home, away):

    prediction = prediction_lookup[
        (home, away)
    ]

    home_xg = prediction["home_xg"]
    away_xg = prediction["away_xg"]

    home_goals = rng.poisson(
        home_xg
    )

    away_goals = rng.poisson(
        away_xg
    )

    return home_goals, away_goals


# =========================================================
# Rank league table
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

    standings = pd.DataFrame(rows)

    standings = standings.sort_values(
        by=[
            "Points",
            "GoalDifference",
            "GoalsFor",
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    standings["Position"] = (
        np.arange(
            len(standings)
        ) + 1
    )

    return standings


# =========================================================
# Team strength for knockout matches
# =========================================================
#
# We use the current-strength ratings already produced by
# the project rather than Points Per Game.
#
# This keeps the knockout stage consistent with the model
# used for the league phase.
# =========================================================

STRENGTH_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ucl_current_strength.csv"
)

strength_df = pd.read_csv(
    STRENGTH_PATH
)

team_strength = dict(
    zip(
        strength_df["Team"],
        strength_df["CurrentStrength"]
    )
)


# =========================================================
# Simulate knockout match
# =========================================================

def simulate_knockout_match(
    home,
    away
):

    # -----------------------------------------------------
    # The league-phase prediction file contains only the
    # remaining league fixtures, so knockout fixtures need
    # their own xG estimate.
    #
    # We use the same strength -> goals relationship that
    # produced the league-phase predictions.
    # -----------------------------------------------------

    home_strength = team_strength[home]

    away_strength = team_strength[away]

    elo_diff = (
        home_strength
        - away_strength
    ) / 100.0

    # Coefficients learned from the domestic dataset.

    home_xg = np.exp(
        0.3910
        + 0.1837 * elo_diff
    )

    away_xg = np.exp(
        0.1013
        - 0.1795 * elo_diff
    )

    home_goals = rng.poisson(
        home_xg
    )

    away_goals = rng.poisson(
        away_xg
    )

    return home_goals, away_goals


# =========================================================
# Simulate two-legged playoff
# =========================================================

def simulate_two_legged_playoff(
    seeded,
    unseeded
):

    # -----------------------------------------------------
    # Leg 1
    #
    # Unseeded team hosts first.
    # -----------------------------------------------------

    first_home = unseeded
    first_away = seeded

    first_home_goals, first_away_goals = (
        simulate_knockout_match(
            first_home,
            first_away
        )
    )


    # -----------------------------------------------------
    # Leg 2
    #
    # Seeded team hosts second.
    # -----------------------------------------------------

    second_home = seeded
    second_away = unseeded

    second_home_goals, second_away_goals = (
        simulate_knockout_match(
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
    # Winner
    # -----------------------------------------------------

    if seeded_goals > unseeded_goals:

        return seeded

    if unseeded_goals > seeded_goals:

        return unseeded


    # -----------------------------------------------------
    # Aggregate draw
    #
    # No away-goals rule.
    #
    # Resolve using relative CurrentStrength.
    # -----------------------------------------------------

    seeded_strength = team_strength[
        seeded
    ]

    unseeded_strength = team_strength[
        unseeded
    ]

    strength_gap = (
        seeded_strength
        - unseeded_strength
    )

    seeded_probability = (
        0.50
        + 0.20 * np.tanh(
            strength_gap / 100
        )
    )

    if rng.random() < seeded_probability:

        return seeded

    return unseeded


# =========================================================
# Create playoff draw
# =========================================================

def create_playoff_draw(
    seeded,
    unseeded
):

    seeded = list(seeded)

    unseeded = list(unseeded)

    rng.shuffle(seeded)

    rng.shuffle(unseeded)

    return list(
        zip(
            seeded,
            unseeded
        )
    )


# =========================================================
# Result storage
# =========================================================

top8_counts = {
    team: 0
    for team in teams
}

playoff_qualification_counts = {
    team: 0
    for team in teams
}

playoff_advance_counts = {
    team: 0
    for team in teams
}

elimination_counts = {
    team: 0
    for team in teams
}


# =========================================================
# Run simulations
# =========================================================

print()
print("Running league + playoff simulations...")


for simulation in range(
    N_SIMULATIONS
):

    # -----------------------------------------------------
    # Start from actual current standings
    # -----------------------------------------------------

    table = {
        team: {
            "points": stats["points"],
            "goals_for": stats["goals_for"],
            "goals_against": stats["goals_against"],
        }
        for team, stats
        in initial_standings.items()
    }


    # -----------------------------------------------------
    # Simulate remaining league fixtures
    # -----------------------------------------------------

    for _, fixture in fixtures.iterrows():

        home = fixture["home"]
        away = fixture["away"]

        home_goals, away_goals = (
            simulate_match(
                home,
                away
            )
        )


        # Goals

        table[home]["goals_for"] += (
            home_goals
        )

        table[home]["goals_against"] += (
            away_goals
        )

        table[away]["goals_for"] += (
            away_goals
        )

        table[away]["goals_against"] += (
            home_goals
        )


        # Points

        if home_goals > away_goals:

            table[home]["points"] += 3

        elif away_goals > home_goals:

            table[away]["points"] += 3

        else:

            table[home]["points"] += 1

            table[away]["points"] += 1


    # -----------------------------------------------------
    # Final league table
    # -----------------------------------------------------

    league_table = rank_standings(
        table
    )


    # -----------------------------------------------------
    # Record league-phase outcomes
    # -----------------------------------------------------

    for _, row in league_table.iterrows():

        team = row["Team"]

        position = int(
            row["Position"]
        )

        if position <= 8:

            top8_counts[team] += 1

        elif position <= 24:

            playoff_qualification_counts[
                team
            ] += 1

        else:

            elimination_counts[
                team
            ] += 1


    # -----------------------------------------------------
    # Extract playoff teams
    # -----------------------------------------------------

    seeded = league_table.iloc[
        8:16
    ]["Team"].tolist()

    unseeded = league_table.iloc[
        16:24
    ]["Team"].tolist()


    # -----------------------------------------------------
    # Create playoff draw
    # -----------------------------------------------------

    playoff_draw = create_playoff_draw(
        seeded,
        unseeded
    )


    # -----------------------------------------------------
    # Simulate eight playoff ties
    # -----------------------------------------------------

    for seeded_team, unseeded_team in playoff_draw:

        winner = simulate_two_legged_playoff(
            seeded_team,
            unseeded_team
        )

        playoff_advance_counts[
            winner
        ] += 1


# =========================================================
# Build final results
# =========================================================

results = []

for team in teams:

    playoff_appearances = (
        playoff_qualification_counts[team]
    )

    if playoff_appearances > 0:

        playoff_advance_probability = (
            playoff_advance_counts[team]
            / playoff_appearances
        )

    else:

        playoff_advance_probability = 0.0


    results.append({

        "Team": team,

        "Top8Probability":
            top8_counts[team]
            / N_SIMULATIONS,

        "KnockoutPlayoffProbability":
            playoff_qualification_counts[team]
            / N_SIMULATIONS,

        "PlayoffAdvanceProbability":
            playoff_advance_probability,

        "EliminationProbability":
            elimination_counts[team]
            / N_SIMULATIONS,

    })


results = pd.DataFrame(
    results
)


# =========================================================
# Overall probability of reaching Round of 16
# =========================================================
#
# A team reaches the Round of 16 if:
#
# 1. It finishes top 8
# OR
# 2. It finishes 9-24 and wins its playoff
#
# =========================================================

results[
    "RoundOf16Probability"
] = (
    results["Top8Probability"]
    + (
        results["KnockoutPlayoffProbability"]
        * results["PlayoffAdvanceProbability"]
    )
)


# =========================================================
# Sort
# =========================================================

results = results.sort_values(
    by="RoundOf16Probability",
    ascending=False
).reset_index(
    drop=True
)


# =========================================================
# Save
# =========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# Display
# =========================================================

print()
print("LEAGUE + KNOCKOUT PLAYOFF SIMULATION")
print("=====================================")

display_df = results.copy()

for column in [
    "Top8Probability",
    "KnockoutPlayoffProbability",
    "PlayoffAdvanceProbability",
    "EliminationProbability",
    "RoundOf16Probability",
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