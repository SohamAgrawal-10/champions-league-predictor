import numpy as np
from scipy.stats import poisson


def calculate_match_probabilities(
    home_xg,
    away_xg,
    max_goals=10
):
    """
    Convert expected goals into
    Home Win / Draw / Away Win probabilities.
    """

    home_goals = np.arange(max_goals + 1)
    away_goals = np.arange(max_goals + 1)

    home_probs = poisson.pmf(
        home_goals,
        home_xg
    )

    away_probs = poisson.pmf(
        away_goals,
        away_xg
    )

    # Scoreline probability matrix
    score_matrix = np.outer(
        home_probs,
        away_probs
    )

    # Home wins: home goals > away goals
    home_win = np.tril(
        score_matrix,
        k=-1
    ).sum()

    # Draws: home goals == away goals
    draw = np.trace(score_matrix)

    # Away wins: away goals > home goals
    away_win = np.triu(
        score_matrix,
        k=1
    ).sum()

    # Account for tiny probability lost beyond max_goals
    total = home_win + draw + away_win

    home_win /= total
    draw /= total
    away_win /= total

    return home_win, draw, away_win


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    home_xg = 1.22
    away_xg = 1.84

    home_win, draw, away_win = calculate_match_probabilities(
        home_xg,
        away_xg
    )

    print("Example: Arsenal vs Man City")
    print()
    print(f"Home win: {home_win:.3%}")
    print(f"Draw:     {draw:.3%}")
    print(f"Away win: {away_win:.3%}")

    print()
    print("Total probability:")
    print(
        f"{home_win + draw + away_win:.3%}"
    )