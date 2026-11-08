# Champions League Predictor

> **A machine learning and Monte Carlo simulation pipeline that estimates 2026/27 UEFA Champions League match, progression, and winner probabilities.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Live-success.svg)]()

## Live Demo

**[Open the live Champions League Predictor](https://champions-league-predictor-wrgugpbrc9fjzanpq2ht2j.streamlit.app/)**

The application provides:

- Current tournament winner probabilities
- Round of 16, quarter-final, semi-final and final probabilities
- Upcoming fixture predictions
- Expected goals for upcoming matches
- Matchday-by-matchday predictions
- Interactive team explorer
- Explanation of the underlying prediction methodology

---

## 1. Project Overview

The **Champions League Predictor** is an end-to-end machine learning project designed to answer a simple question:

> **Given the teams, their current strength, recent performance, and remaining fixtures, who is most likely to win the 2026/27 UEFA Champions League?**

Rather than directly training a model to predict the tournament winner, the project decomposes the problem into smaller, more interpretable stages.

Historical domestic football matches are used to build a leakage-safe team strength and match prediction pipeline. Current Champions League performance is then combined with external Elo ratings to estimate each team's present strength.

Those strength estimates are converted into expected goals using a Poisson regression model. Match probabilities are then generated from the resulting expected goals.

Finally, the remaining Champions League is simulated **10,000 times**, including:

- League phase
- League-phase standings
- Playoff qualification
- Two-legged playoff ties
- Round of 16
- Quarter-finals
- Semi-finals
- Final

The proportion of simulations in which each team wins the competition becomes its estimated championship probability.

The resulting predictions are presented through an interactive **Streamlit dashboard**.

---

## 2. Why I Built This

Predicting a football tournament is more interesting than predicting an individual match because the two problems are connected.

A team's probability of winning the Champions League depends not only on how strong it is, but also on:

- Its remaining fixtures
- Its position in the league phase
- Whether it qualifies directly for the Round of 16
- Whether it enters the knockout playoffs
- Potential opponents in later rounds
- Home advantage
- Expected goals
- Randomness in individual matches

This made the project an opportunity to build a complete ML system rather than simply train a classifier on historical match results.

The main goal was therefore to build something that demonstrates:

**Data Engineering → Feature Engineering → Machine Learning → Probabilistic Modeling → Simulation → Productization**

---

## 3. Project Architecture

```text
Historical Football Data
          │
          ▼
Data Cleaning & Filtering
          │
          ▼
Leakage-Safe Match Features
          │
          ├── Elo Ratings
          ├── Rolling Form
          ├── Home/Away Form
          └── Team Strength
          │
          ▼
Historical Goal Model
          │
          ▼
Current UCL Team Strength
          │
          ▼
Poisson Expected-Goals Model
          │
          ▼
Match Win / Draw / Loss Probabilities
          │
          ▼
10,000 Monte Carlo Tournament Simulations
          │
          ├── League Phase
          ├── Playoff
          ├── Round of 16
          ├── Quarter-finals
          ├── Semi-finals
          └── Final
          │
          ▼
Tournament Probabilities
          │
          ▼
Streamlit Dashboard
```

---

## 4. Key Features

### Tournament Winner Prediction

The dashboard ranks teams according to their simulated probability of winning the Champions League.

Current simulation results place the leading contenders approximately as:

| Rank | Team | Winner Probability |
|---:|---|---:|
| 1 | Paris | 23.1% |
| 2 | Bayern München | 22.0% |
| 3 | Arsenal | 20.2% |
| 4 | Barcelona | 11.8% |
| 5 | Real Madrid | 10.9% |
| 6 | Man City | 5.9% |
| 7 | Como | 1.2% |
| 8 | Aston Villa | 1.0% |

These numbers represent the output of the current simulation and **should not be interpreted as guaranteed or objective probabilities**.

---

### Tournament Progression

The model estimates the probability of each team reaching every major stage:

```text
Round of 16
     ↓
Quarter-final
     ↓
Semi-final
     ↓
Final
     ↓
Winner
```

For example, the current simulation produced approximately:

| Team | R16 | QF | SF | Final | Winner |
|---|---:|---:|---:|---:|---:|
| Paris | 100.0% | 87.8% | 61.8% | 39.9% | 23.1% |
| Bayern München | 99.5% | 85.4% | 59.7% | 38.1% | 22.0% |
| Arsenal | 99.7% | 84.9% | 59.0% | 36.2% | 20.2% |
| Barcelona | 98.6% | 77.0% | 46.1% | 24.5% | 11.8% |
| Real Madrid | 100.0% | 80.8% | 47.0% | 23.8% | 10.9% |

---

### Upcoming Match Predictions

The dashboard also provides predictions for upcoming Champions League fixtures.

For every match, the model estimates:

- Home win probability
- Draw probability
- Away win probability
- Home expected goals
- Away expected goals
- Predicted outcome

Users can select individual matchdays to explore the upcoming schedule.

---

### Team Explorer

The interactive Team Explorer allows users to select any participating team and inspect its:

- Round of 16 probability
- Quarter-final probability
- Semi-final probability
- Final probability
- Winner probability

---

## 5. Modeling Approach

### 5.1 Historical Match Dataset

The initial historical dataset contains football matches from 2000 onward.

For the core modeling pipeline, the project focuses on the five major European domestic leagues:

- Premier League
- La Liga
- Bundesliga
- Serie A
- Ligue 1

The historical dataset contains more than **45,000 matches** after filtering.

The data was used to construct pre-match features without using information that would only have been available after the match.

---

### 5.2 Avoiding Data Leakage

One of the major design considerations was avoiding future information leaking into the prediction.

Features such as team form and strength are calculated using information available **before the match being predicted**.

For example, a team's rolling form is calculated from its previous matches rather than including the result of the current match.

The model therefore attempts to answer:

> "What would the model have known before this match happened?"

rather than:

> "What information describes the match after it happened?"

This distinction is particularly important when evaluating sports prediction models.

---

### 5.3 Elo-Based Team Strength

The project uses Elo-style ratings as one of the primary measures of team strength.

The model also incorporates current Champions League performance to adjust the baseline team ratings.

The current strength pipeline combines:

- External Elo rating
- Champions League points per game
- Champions League goal difference per game

The resulting score is used as the team's current tournament strength.

---

### 5.4 Expected Goals Model

Instead of directly predicting the final result from the team-strength score, the project estimates expected goals for both teams.

The historical relationship was modeled using Poisson regression.

The resulting models use Elo difference as the main predictor.

The current formulation is approximately:

```text
EloDiff = (HomeStrength - AwayStrength) / 100

Home xG = exp(0.3910 + 0.1837 × EloDiff)

Away xG = exp(0.1013 - 0.1795 × EloDiff)
```

This produces expected goals rather than directly predicting a categorical result.

---

### 5.5 Match Probabilities

Once expected goals are estimated, Poisson distributions are used to calculate the probability of different scorelines.

For example:

```text
Home xG = 1.22
Away xG = 1.84
```

The model evaluates possible scorelines such as:

```text
0–0
0–1
1–0
1–1
1–2
2–1
...
```

The probabilities of all home wins are summed to produce the home-win probability.

Similarly:

```text
Draw probability
Away-win probability
```

are calculated.

---

## 6. Monte Carlo Tournament Simulation

The most important part of the project is the tournament simulator.

Instead of trying to directly predict the winner, the entire competition is simulated repeatedly.

### Simulation process

Each simulation:

1. Starts with the current Champions League standings.
2. Simulates the remaining league-phase fixtures.
3. Calculates the final league-phase table.
4. Determines direct Round of 16 qualification.
5. Determines playoff qualification.
6. Simulates two-legged playoff ties.
7. Simulates the Round of 16.
8. Simulates the quarter-finals.
9. Simulates the semi-finals.
10. Simulates the final.
11. Records the tournament winner.

This process is repeated **10,000 times**.

The final probability is:

```text
Winner Probability =
Number of simulations won
────────────────────────
Total simulations
```

For example, if Arsenal wins:

```text
2,020 / 10,000 simulations
```

then its estimated tournament probability is:

```text
20.2%
```

---

## 7. Tournament Validation

The simulation contains validation checks to make sure the probability distributions behave correctly.

The following were validated:

```text
R16 probabilities       → total = 16 teams
Quarter-final           → total = 8 teams
Semi-final              → total = 4 teams
Final                   → total = 2 teams
Winner                  → total = 1 team
```

The simulation also validates league-phase outcomes and playoff qualification.

The full tournament simulation currently passes these validation checks.

---

## 8. Technology Stack

### Programming

- **Python 3.12**

### Data Processing

- **Pandas**
- **NumPy**

### Machine Learning

- **Scikit-learn**
- Poisson regression
- Logistic regression
- Random Forest experimentation

### Statistical Modeling

- Poisson distributions
- Expected-goals modeling
- Elo ratings
- Monte Carlo simulation

### Visualization / Dashboard

- **Streamlit**

### Development

- Visual Studio Code
- Git
- GitHub

---

## 9. Project Structure

```text
champions-league-predictor/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── Matches.csv
│   │   ├── EloRatings.csv
│   │   └── ucl/
│   │       ├── results_2627.csv
│   │       └── fixtures_2627.csv
│   │
│   └── processed/
│       ├── big5_matches.csv
│       ├── model_data.csv
│       ├── model_data_v2.csv
│       ├── model_data_v3.csv
│       ├── elo_features.csv
│       ├── ucl_form.csv
│       ├── ucl_final_elo.csv
│       ├── ucl_current_strength.csv
│       ├── ucl_predictions.csv
│       ├── league_simulation_results.csv
│       ├── league_playoff_simulation_results.csv
│       └── tournament_simulation_results.csv
│
├── src/
│   ├── create_dataset.py
│   ├── create_rolling_features.py
│   ├── create_elo_features.py
│   ├── create_team_strength.py
│   ├── create_ucl_strength.py
│   ├── create_ucl_form.py
│   ├── create_current_strength.py
│   ├── create_ucl_elo.py
│   ├── train_strength_to_goals.py
│   ├── predict_ucl_fixtures.py
│   ├── simulate_league_phase.py
│   ├── simulate_league_and_playoffs.py
│   └── simulate_tournament.py
│
└── results/
```

---

## 10. Installation

### Requirements

- Python 3.12+
- Git
- Internet connection for downloading the datasets

### Clone the repository

```bash
git clone https://github.com/SohamAgrawal-10/champions-league-predictor.git

cd champions-league-predictor
```

### Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 11. Running the Dashboard

Start Streamlit:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

The dashboard reads the processed model outputs from:

```text
data/processed/
```

Therefore, the required processed data must exist before running the application.

---

## 12. Using the Dashboard

The application is divided into several sections.

### Top Contenders

Provides a quick view of the teams with the highest simulated championship probabilities.

### Winner Probability

Displays the top teams and their estimated probability of winning.

### Tournament Progression

Shows how likely every team is to reach:

```text
R16 → QF → SF → Final → Winner
```

### Upcoming Fixtures

Select a matchday to see the model's predictions for upcoming matches.

### Team Explorer

Select a team to inspect its tournament progression probabilities.

### Methodology

Explains the modeling and simulation pipeline behind the predictions.

---

## 13. Challenges and Design Decisions

### Predicting the tournament rather than individual matches

A major design challenge was deciding how to model a tournament outcome.

Directly predicting:

```text
Team → Champions League Winner
```

would discard much of the structure of the competition.

Instead, the project models individual matches and uses those probabilities to reconstruct the tournament through simulation.

---

### Preventing leakage

Rolling team features can easily introduce future information into a model.

The feature engineering pipeline was therefore designed so that each match only uses information available before that match.

---

### Choosing meaningful features

Several different feature combinations and models were tested during development.

The project ultimately prioritized:

- Interpretability
- Leakage safety
- Time-based validation
- Probabilistic outputs
- Usefulness for tournament simulation

rather than endlessly optimizing a single classification accuracy number.

---

### Lack of reliable current player data

An attempt was made to incorporate player-level statistics.

However, historical player statistics from the previous season cannot safely be transferred directly into the 2026/27 season because:

- Players transfer between clubs.
- Squad roles change.
- Injuries and availability change.
- Playing time changes.
- New signings may not appear in historical datasets.

Rather than introducing unreliable player features, the current version intentionally excludes them.

This is an important modeling decision: **missing data is preferable to knowingly introducing misleading data.**

---

## 14. Limitations

This project is a probabilistic model, not a prediction oracle.

Important limitations include:

### Simplified tournament rules

The simulator approximates some UEFA competition mechanics.

In particular, some knockout draw and tie-breaking behavior is simplified compared with the complete official competition rules.

### Limited feature set

The current model does not include:

- Player availability
- Injuries
- Suspensions
- Lineups
- Transfer activity
- Manager changes
- Tactical matchups
- Travel fatigue
- Fixture congestion
- Detailed shot-quality metrics
- Advanced player-level statistics

### External Elo ratings

Current team strength incorporates external Elo ratings, which means the model is not completely self-contained.

### Expected-goals assumptions

The Poisson model assumes a particular relationship between team strength and scoring.

Real football matches can deviate significantly from these assumptions.

### Simulation uncertainty

10,000 simulations provide a useful approximation, but the output remains subject to Monte Carlo variation and model assumptions.

### Probability ≠ certainty

A team with a 23% championship probability is not "predicted to win" with certainty.

It means that under the current model assumptions, that team won approximately 23% of simulated tournaments.

---

## 15. Current Results

The current simulation produces the following leading championship probabilities:

| Team | Championship Probability |
|---|---:|
| Paris | **23.1%** |
| Bayern München | **22.0%** |
| Arsenal | **20.2%** |
| Barcelona | **11.8%** |
| Real Madrid | **10.9%** |
| Man City | **5.9%** |
| Como | **1.2%** |
| Aston Villa | **1.0%** |
| Inter | **1.0%** |
| Liverpool | **1.0%** |

These values will change as new Champions League results become available and the model inputs are updated.

---

## 16. Future Improvements

The project is intentionally designed so additional features can be added later.

Potential improvements include:

- [ ] Automated collection of new Champions League results
- [ ] Automated model/data refresh
- [ ] Better UEFA draw constraints
- [ ] Exact official tiebreaker implementation
- [ ] Improved knockout-round simulation
- [ ] Player availability features
- [ ] Squad-strength modeling
- [ ] Injury and suspension data
- [ ] Advanced shot/xG statistics
- [ ] Calibration analysis
- [ ] Prediction confidence intervals
- [ ] Historical backtesting of tournament predictions
- [ ] Automated deployment updates
- [ ] Interactive visualizations
- [ ] Model comparison dashboard

---

## 17. Credits & Data Sources

### Historical Football Data

Historical club match and Elo data were obtained from publicly available football datasets used during development.

### Champions League Data

Current 2026/27 Champions League fixtures and completed results were sourced from the `open-matchday` football data repository.

### Libraries

This project makes use of:

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)

---

## 18. Author

**Soham Agrawal**

GitHub: **[@SohamAgrawal-10](https://github.com/SohamAgrawal-10)**

---

## 19. License

This project is intended primarily as an educational and portfolio project.

If you reuse the project or its data pipeline, please respect the licensing and attribution requirements of the original datasets and repositories.