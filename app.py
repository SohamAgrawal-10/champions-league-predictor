import streamlit as st
import pandas as pd
from pathlib import Path


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Champions League Predictor",
    page_icon="⚽",
    layout="wide"
)


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

tournament_file = DATA_DIR / "tournament_simulation_results.csv"
fixtures_file = DATA_DIR / "ucl_predictions.csv"


# ==================================================
# LOAD DATA
# ==================================================

if not tournament_file.exists():
    st.error(
        "Tournament simulation results not found. "
        "Run src/simulate_tournament.py first."
    )
    st.stop()

tournament = pd.read_csv(tournament_file)

fixtures = None

if fixtures_file.exists():
    fixtures = pd.read_csv(fixtures_file)


# ==================================================
# HEADER
# ==================================================

st.title("Champions League Predictor")

st.caption(
    "2026/27 UEFA Champions League — "
    "10,000 Monte Carlo tournament simulations"
)

st.divider()


# ==================================================
# TOP CONTENDERS
# ==================================================

st.header("Top Contenders")

top_teams = tournament.sort_values(
    "WinnerProbability",
    ascending=False
).head(5).copy()

cols = st.columns(5)

for col, (_, team) in zip(cols, top_teams.iterrows()):

    with col:
        st.metric(
            team["Team"],
            f"{team['WinnerProbability'] * 100:.1f}%"
        )

st.caption("Probability of winning the Champions League")


# ==================================================
# WINNER PROBABILITY
# ==================================================

st.divider()

st.header("Winner Probability")

top_teams = tournament.sort_values(
    "WinnerProbability",
    ascending=False
).head(10).copy()

for _, row in top_teams.iterrows():

    team = row["Team"]
    probability = row["WinnerProbability"] * 100

    col1, col2 = st.columns([2, 5])

    with col1:
        st.write(f"**{team}**")

    with col2:
        st.progress(
            int(probability),
            text=f"{probability:.1f}%"
        )


# ==================================================
# TOURNAMENT PROGRESSION
# ==================================================

st.divider()

st.header("Tournament Progression")

progression = tournament.sort_values(
    "WinnerProbability",
    ascending=False
).copy()

display_columns = [
    "Team",
    "RoundOf16Probability",
    "QuarterFinalProbability",
    "SemiFinalProbability",
    "FinalProbability",
    "WinnerProbability"
]

progression_display = progression[display_columns].copy()

progression_display.columns = [
    "Team",
    "Round of 16",
    "Quarter-final",
    "Semi-final",
    "Final",
    "Winner"
]

for column in progression_display.columns[1:]:
    progression_display[column] *= 100

st.dataframe(
    progression_display.style.format(
        {
            "Round of 16": "{:.1f}%",
            "Quarter-final": "{:.1f}%",
            "Semi-final": "{:.1f}%",
            "Final": "{:.1f}%",
            "Winner": "{:.1f}%"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ==================================================
# UPCOMING FIXTURES
# ==================================================

st.divider()

st.header("Upcoming Fixtures")

if fixtures is None:

    st.warning(
        "Upcoming fixture predictions not found."
    )

else:

    matchdays = sorted(
        fixtures["gw"].unique()
    )

    selected_gw = st.selectbox(
        "Select Matchday",
        matchdays,
        format_func=lambda x: f"Matchday {x}"
    )

    selected_fixtures = fixtures[
        fixtures["gw"] == selected_gw
    ]

    for _, match in selected_fixtures.iterrows():

        home = match["home"]
        away = match["away"]

        home_prob = match["HomeWinProb"] * 100
        draw_prob = match["DrawProb"] * 100
        away_prob = match["AwayWinProb"] * 100

        home_xg = match["HomeXG"]
        away_xg = match["AwayXG"]

        st.subheader(
            f"{home} vs {away}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Home Win",
                f"{home_prob:.1f}%"
            )

        with col2:
            st.metric(
                "Draw",
                f"{draw_prob:.1f}%"
            )

        with col3:
            st.metric(
                "Away Win",
                f"{away_prob:.1f}%"
            )

        st.caption(
            f"Expected goals: "
            f"{home} {home_xg:.2f} — "
            f"{away_xg:.2f} {away}"
        )


# ==================================================
# TEAM EXPLORER
# ==================================================

st.divider()

st.header("Team Explorer")

teams = sorted(
    tournament["Team"].unique()
)

selected_team = st.selectbox(
    "Select a team",
    teams
)

team_data = tournament[
    tournament["Team"] == selected_team
].iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Round of 16",
        f"{team_data['RoundOf16Probability'] * 100:.1f}%"
    )

with col2:
    st.metric(
        "Quarter-final",
        f"{team_data['QuarterFinalProbability'] * 100:.1f}%"
    )

with col3:
    st.metric(
        "Semi-final",
        f"{team_data['SemiFinalProbability'] * 100:.1f}%"
    )

with col4:
    st.metric(
        "Final",
        f"{team_data['FinalProbability'] * 100:.1f}%"
    )

with col5:
    st.metric(
        "Winner",
        f"{team_data['WinnerProbability'] * 100:.1f}%"
    )


# ==================================================
# METHODOLOGY
# ==================================================

st.divider()

st.header("How the Prediction Works")

st.markdown(
    """
    The predictor combines historical domestic match data,
    Elo-based team strength, recent Champions League performance,
    and a Poisson expected-goals model.

    The resulting match probabilities are used to simulate the
    remaining Champions League fixtures.

    The tournament is simulated **10,000 times** using Monte Carlo
    simulation. The percentage of simulations in which a team wins
    the tournament becomes its estimated championship probability.
    """
)

st.info(
    "These probabilities are model estimates, not guarantees. "
    "They depend on the assumptions and limitations of the underlying "
    "strength and goal models."
)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Champions League Predictor • Built with Python, "
    "Pandas, Scikit-learn and Streamlit"
)