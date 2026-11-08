import numpy as np
import pandas as pd
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

N_SIMULATIONS = 10_000
RANDOM_SEED = 42

BASE_DIR = "data/processed"

FORM_FILE = f"{BASE_DIR}/ucl_form.csv"
FIXTURES_FILE = "data/raw/ucl/fixtures_2627.csv"
PREDICTIONS_FILE = f"{BASE_DIR}/ucl_predictions.csv"
STRENGTH_FILE = f"{BASE_DIR}/ucl_current_strength.csv"

OUTPUT_FILE = f"{BASE_DIR}/tournament_simulation_results.csv"


# Strength -> goals model
HOME_INTERCEPT = 0.3910
HOME_ELO_COEF = 0.1837

AWAY_INTERCEPT = 0.1013
AWAY_ELO_COEF = -0.1795


# ============================================================
# LOAD DATA
# ============================================================

form = pd.read_csv(FORM_FILE)
fixtures = pd.read_csv(FIXTURES_FILE)
predictions = pd.read_csv(PREDICTIONS_FILE)
strength_df = pd.read_csv(STRENGTH_FILE)

teams = form["Team"].tolist()

strength = dict(
    zip(
        strength_df["Team"],
        strength_df["CurrentStrength"]
    )
)

remaining_fixtures = fixtures.merge(
    predictions[
        [
            "gw",
            "home",
            "away",
            "HomeXG",
            "AwayXG",
        ]
    ],
    on=["gw", "home", "away"],
    how="inner",
)


# ============================================================
# MATCH MODEL
# ============================================================

def calculate_xg(home, away):
    """
    Convert current-strength difference into expected goals.
    """

    elo_diff = (
        strength[home] - strength[away]
    ) / 100.0

    home_xg = np.exp(
        HOME_INTERCEPT +
        HOME_ELO_COEF * elo_diff
    )

    away_xg = np.exp(
        AWAY_INTERCEPT +
        AWAY_ELO_COEF * elo_diff
    )

    return home_xg, away_xg


def simulate_match(home, away, rng):
    """
    Simulate a 90-minute match.
    """

    home_xg, away_xg = calculate_xg(home, away)

    home_goals = rng.poisson(home_xg)
    away_goals = rng.poisson(away_xg)

    return home_goals, away_goals


# ============================================================
# LEAGUE PHASE
# ============================================================

def simulate_league_phase(rng):
    """
    Simulate all remaining league-phase matches.
    """

    table = {}

    for _, row in form.iterrows():

        table[row["Team"]] = {
            "points": int(row["Points"]),
            "gf": int(row["GoalsFor"]),
            "ga": int(row["GoalsAgainst"]),
        }

    for _, fixture in remaining_fixtures.iterrows():

        home = fixture["home"]
        away = fixture["away"]

        home_xg = fixture["HomeXG"]
        away_xg = fixture["AwayXG"]

        home_goals = rng.poisson(home_xg)
        away_goals = rng.poisson(away_xg)

        table[home]["gf"] += home_goals
        table[home]["ga"] += away_goals

        table[away]["gf"] += away_goals
        table[away]["ga"] += home_goals

        if home_goals > away_goals:

            table[home]["points"] += 3

        elif away_goals > home_goals:

            table[away]["points"] += 3

        else:

            table[home]["points"] += 1
            table[away]["points"] += 1

    rows = []

    for team, data in table.items():

        rows.append(
            {
                "Team": team,
                "Points": data["points"],
                "GF": data["gf"],
                "GA": data["ga"],
                "GD": data["gf"] - data["ga"],
            }
        )

    league_table = pd.DataFrame(rows)

    # Simplified ranking:
    # Points → Goal Difference → Goals Scored

    league_table = league_table.sort_values(
        ["Points", "GD", "GF"],
        ascending=False
    ).reset_index(drop=True)

    league_table["Position"] = (
        np.arange(len(league_table)) + 1
    )

    return league_table


# ============================================================
# PLAYOFF DRAW
# ============================================================

def create_playoff_draw(league_table, rng):
    """
    Positions 9–16 are seeded.
    Positions 17–24 are unseeded.

    V1 uses random seeded-vs-unseeded pairing.
    """

    seeded = league_table.iloc[8:16]["Team"].tolist()
    unseeded = league_table.iloc[16:24]["Team"].tolist()

    rng.shuffle(seeded)
    rng.shuffle(unseeded)

    return list(zip(seeded, unseeded))


# ============================================================
# PENALTY SHOOTOUT
# ============================================================

def simulate_penalty_shootout(team1, team2, rng):
    """
    Simplified penalty shootout.

    Stronger team receives a small advantage.
    """

    strength_gap = (
        strength[team1] -
        strength[team2]
    )

    team1_probability = (
        0.50 +
        0.10 * np.tanh(strength_gap / 100)
    )

    if rng.random() < team1_probability:
        return team1

    return team2


# ============================================================
# TWO-LEGGED TIE
# ============================================================

def simulate_two_legged_tie(
    team1,
    team2,
    rng,
):
    """
    Simulate a two-legged knockout tie.

    Team2 hosts the first leg.
    Team1 hosts the second leg.

    If aggregate score is tied, simulate
    extra time and then penalties.
    """

    # --------------------------------------------------------
    # LEG 1
    # --------------------------------------------------------

    leg1_home_goals, leg1_away_goals = simulate_match(
        team2,
        team1,
        rng,
    )

    # --------------------------------------------------------
    # LEG 2
    # --------------------------------------------------------

    leg2_home_goals, leg2_away_goals = simulate_match(
        team1,
        team2,
        rng,
    )

    team1_total = (
        leg1_away_goals +
        leg2_home_goals
    )

    team2_total = (
        leg1_home_goals +
        leg2_away_goals
    )

    # --------------------------------------------------------
    # AGGREGATE WINNER
    # --------------------------------------------------------

    if team1_total > team2_total:
        return team1

    if team2_total > team1_total:
        return team2

    # --------------------------------------------------------
    # EXTRA TIME
    # --------------------------------------------------------

    # Approximate extra-time scoring as 1/3
    # of normal 90-minute expected goals.

    team1_xg_leg1, team2_xg_leg1 = calculate_xg(
        team2,
        team1,
    )

    team1_xg_leg2, team2_xg_leg2 = calculate_xg(
        team1,
        team2,
    )

    team1_et_xg = (
        team1_xg_leg1 +
        team1_xg_leg2
    ) / 3

    team2_et_xg = (
        team2_xg_leg1 +
        team2_xg_leg2
    ) / 3

    team1_et_goals = rng.poisson(team1_et_xg)
    team2_et_goals = rng.poisson(team2_et_xg)

    if team1_et_goals > team2_et_goals:
        return team1

    if team2_et_goals > team1_et_goals:
        return team2

    # --------------------------------------------------------
    # PENALTIES
    # --------------------------------------------------------

    return simulate_penalty_shootout(
        team1,
        team2,
        rng,
    )


# ============================================================
# PLAYOFFS
# ============================================================

def simulate_playoffs(league_table, rng):

    draw = create_playoff_draw(
        league_table,
        rng,
    )

    winners = []

    for team1, team2 in draw:

        winner = simulate_two_legged_tie(
            team1,
            team2,
            rng,
        )

        winners.append(winner)

    return winners


# ============================================================
# ROUND OF 16 DRAW
# ============================================================

def create_round_of_16_draw(
    league_table,
    playoff_winners,
    rng,
):
    """
    Positions 1–8 are seeded.

    The eight playoff winners are unseeded.

    V1 uses random seeded-vs-unseeded pairing.
    """

    seeded = league_table.iloc[:8]["Team"].tolist()

    unseeded = playoff_winners.copy()

    rng.shuffle(seeded)
    rng.shuffle(unseeded)

    return list(zip(seeded, unseeded))


# ============================================================
# ROUND OF 16
# ============================================================

def simulate_round_of_16(
    league_table,
    playoff_winners,
    rng,
):

    draw = create_round_of_16_draw(
        league_table,
        playoff_winners,
        rng,
    )

    quarter_finalists = []

    for team1, team2 in draw:

        winner = simulate_two_legged_tie(
            team1,
            team2,
            rng,
        )

        quarter_finalists.append(winner)

    return quarter_finalists


# ============================================================
# QUARTER-FINAL DRAW
# ============================================================

def create_quarter_final_draw(
    quarter_finalists,
    rng,
):
    """
    Randomly pair the eight quarter-finalists.
    """

    teams = quarter_finalists.copy()

    rng.shuffle(teams)

    return [
        (teams[i], teams[i + 1])
        for i in range(0, 8, 2)
    ]


# ============================================================
# QUARTER-FINALS
# ============================================================

def simulate_quarter_finals(
    quarter_finalists,
    rng,
):

    draw = create_quarter_final_draw(
        quarter_finalists,
        rng,
    )

    semi_finalists = []

    for team1, team2 in draw:

        winner = simulate_two_legged_tie(
            team1,
            team2,
            rng,
        )

        semi_finalists.append(winner)

    return semi_finalists


# ============================================================
# SEMI-FINAL DRAW
# ============================================================

def create_semi_final_draw(
    semi_finalists,
    rng,
):
    """
    Randomly pair the four semi-finalists.
    """

    teams = semi_finalists.copy()

    rng.shuffle(teams)

    return [
        (teams[i], teams[i + 1])
        for i in range(0, 4, 2)
    ]


# ============================================================
# SEMI-FINALS
# ============================================================

def simulate_semi_finals(
    semi_finalists,
    rng,
):

    draw = create_semi_final_draw(
        semi_finalists,
        rng,
    )

    finalists = []

    for team1, team2 in draw:

        winner = simulate_two_legged_tie(
            team1,
            team2,
            rng,
        )

        finalists.append(winner)

    return finalists


# ============================================================
# FINAL
# ============================================================

def simulate_final(
    finalists,
    rng,
):
    """
    Simulate the Champions League final.

    The final is a single match.
    """

    team1 = finalists[0]
    team2 = finalists[1]

    team1_goals, team2_goals = simulate_match(
        team1,
        team2,
        rng,
    )

    if team1_goals > team2_goals:
        return team1

    if team2_goals > team1_goals:
        return team2

    # Draw after 90 minutes:
    # approximate extra time.

    team1_xg, team2_xg = calculate_xg(
        team1,
        team2,
    )

    team1_et_goals = rng.poisson(
        team1_xg / 3
    )

    team2_et_goals = rng.poisson(
        team2_xg / 3
    )

    if team1_et_goals > team2_et_goals:
        return team1

    if team2_et_goals > team1_et_goals:
        return team2

    # Still tied → penalties

    return simulate_penalty_shootout(
        team1,
        team2,
        rng,
    )


# ============================================================
# MONTE CARLO TOURNAMENT
# ============================================================

def run_simulation():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    r16_count = defaultdict(int)
    qf_count = defaultdict(int)
    sf_count = defaultdict(int)
    final_count = defaultdict(int)
    winner_count = defaultdict(int)

    # --------------------------------------------------------
    # SIMULATIONS
    # --------------------------------------------------------

    for simulation in range(N_SIMULATIONS):

        # ====================================================
        # 1. LEAGUE PHASE
        # ====================================================

        league_table = simulate_league_phase(rng)

        top8 = (
            league_table
            .iloc[:8]["Team"]
            .tolist()
        )

        # ====================================================
        # 2. PLAYOFFS
        # ====================================================

        playoff_winners = simulate_playoffs(
            league_table,
            rng,
        )

        # ====================================================
        # 3. ROUND OF 16 QUALIFICATION
        # ====================================================

        r16_teams = (
            top8 +
            playoff_winners
        )

        if len(r16_teams) != 16:
            raise RuntimeError(
                "Round of 16 does not contain exactly 16 teams."
            )

        for team in r16_teams:
            r16_count[team] += 1

        # ====================================================
        # 4. ROUND OF 16
        # ====================================================

        quarter_finalists = simulate_round_of_16(
            league_table,
            playoff_winners,
            rng,
        )

        if len(quarter_finalists) != 8:
            raise RuntimeError(
                "Quarter-finals do not contain exactly 8 teams."
            )

        for team in quarter_finalists:
            qf_count[team] += 1

        # ====================================================
        # 5. QUARTER-FINALS
        # ====================================================

        semi_finalists = simulate_quarter_finals(
            quarter_finalists,
            rng,
        )

        if len(semi_finalists) != 4:
            raise RuntimeError(
                "Semi-finals do not contain exactly 4 teams."
            )

        for team in semi_finalists:
            sf_count[team] += 1

        # ====================================================
        # 6. SEMI-FINALS
        # ====================================================

        finalists = simulate_semi_finals(
            semi_finalists,
            rng,
        )

        if len(finalists) != 2:
            raise RuntimeError(
                "Final does not contain exactly 2 teams."
            )

        for team in finalists:
            final_count[team] += 1

        # ====================================================
        # 7. FINAL
        # ====================================================

        winner = simulate_final(
            finalists,
            rng,
        )

        winner_count[winner] += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (simulation + 1) % 1000 == 0:

            print(
                f"Completed "
                f"{simulation + 1:,}/"
                f"{N_SIMULATIONS:,} simulations"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    for team in teams:

        results.append(
            {
                "Team": team,

                "RoundOf16Probability":
                    r16_count[team] /
                    N_SIMULATIONS,

                "QuarterFinalProbability":
                    qf_count[team] /
                    N_SIMULATIONS,

                "SemiFinalProbability":
                    sf_count[team] /
                    N_SIMULATIONS,

                "FinalProbability":
                    final_count[team] /
                    N_SIMULATIONS,

                "WinnerProbability":
                    winner_count[team] /
                    N_SIMULATIONS,
            }
        )

    results = pd.DataFrame(results)

    # Sort by winner probability
    results = results.sort_values(
        "WinnerProbability",
        ascending=False,
    ).reset_index(drop=True)

    results["Rank"] = (
        np.arange(len(results)) + 1
    )

    results = results[
        [
            "Rank",
            "Team",
            "RoundOf16Probability",
            "QuarterFinalProbability",
            "SemiFinalProbability",
            "FinalProbability",
            "WinnerProbability",
        ]
    ]

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 80)
    print("TOURNAMENT VALIDATION")
    print("=" * 80)

    r16_total = (
        results["RoundOf16Probability"]
        .sum()
    )

    qf_total = (
        results["QuarterFinalProbability"]
        .sum()
    )

    sf_total = (
        results["SemiFinalProbability"]
        .sum()
    )

    final_total = (
        results["FinalProbability"]
        .sum()
    )

    winner_total = (
        results["WinnerProbability"]
        .sum()
    )

    print(
        f"R16 probability total:     {r16_total:.4f}"
    )

    print(
        f"QF probability total:      {qf_total:.4f}"
    )

    print(
        f"SF probability total:      {sf_total:.4f}"
    )

    print(
        f"Final probability total:   {final_total:.4f}"
    )

    print(
        f"Winner probability total:  {winner_total:.4f}"
    )

    print()

    if not np.isclose(
        r16_total,
        16.0,
        atol=0.001,
    ):
        raise RuntimeError(
            "R16 probability total should equal 16."
        )

    if not np.isclose(
        qf_total,
        8.0,
        atol=0.001,
    ):
        raise RuntimeError(
            "QF probability total should equal 8."
        )

    if not np.isclose(
        sf_total,
        4.0,
        atol=0.001,
    ):
        raise RuntimeError(
            "SF probability total should equal 4."
        )

    if not np.isclose(
        final_total,
        2.0,
        atol=0.001,
    ):
        raise RuntimeError(
            "Final probability total should equal 2."
        )

    if not np.isclose(
        winner_total,
        1.0,
        atol=0.001,
    ):
        raise RuntimeError(
            "Winner probability total should equal 1."
        )

    print("VALIDATION PASSED")

    # ========================================================
    # SAVE
    # ========================================================

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FULL CHAMPIONS LEAGUE TOURNAMENT SIMULATION"
    )
    print("=" * 80)

    display = results.copy()

    probability_columns = [
        "RoundOf16Probability",
        "QuarterFinalProbability",
        "SemiFinalProbability",
        "FinalProbability",
        "WinnerProbability",
    ]

    for column in probability_columns:

        display[column] = (
            display[column]
            .map(lambda x: f"{x:.1%}")
        )

    print(
        display.to_string(index=False)
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_simulation()