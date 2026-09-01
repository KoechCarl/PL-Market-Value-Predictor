# Premier League Player Valuation Model

Predicting a Premier League player's estimated market value from their 2024/25 season performance statistics.

## Problem Statement

Given a player's on-pitch performance data for the 2024/25 Premier League season, can we estimate their market value? This project builds a first-pass regression model to answer that question, following the CRISP-DM methodology (Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation).

## Data Sources

| Source | Content | Rows |
|---|---|---|
| [Kaggle: EPL Player Stats 2024/25](https://www.kaggle.com/datasets/aesika/english-premier-league-player-stats-2425) | Season-aggregate performance stats (goals, assists, shots, passing, defensive actions) | 562 players |
| [Kaggle: PL Market Value Dataset](https://www.kaggle.com/datasets/piyushsharma37/premier-league-market-value-dataset-2025) | Player market valuations (Transfermarkt-derived), age, current club | 499 rows (477 after deduplication) |
| Premier League 2024/25 final table | Used to construct a 3-tier club prestige feature | Manual reference |

## Methodology

### Data Preparation
- Deduplicated market value records (22 duplicate player entries, averaged)
- Merged performance and valuation data on normalized player name (accent-stripped, manual alias mapping for known mismatches e.g. nicknames, name-order differences)
- Filtered to outfield players with ≥450 minutes played (~5 full matches), to exclude small-sample noise
- Excluded goalkeepers (fundamentally different value drivers — saves, clean sheets — vs. outfield stats)
- Excluded three clubs relegated at the end of 2024/25 (Ipswich Town, Leicester City, Southampton), whose players are absent from the current-roster valuation source
- Final modeling dataset: **249 players** (94 DEF / 123 MID / 32 FWD)

### Feature Engineering
- `club_tier`: players bucketed into Top6 / MidTable / Bottom based on 2024/25 final league position, as a proxy for club prestige
- `Position`: one-hot encoded (DEF / MID / FWD)
- Target variable log-transformed (`log(market_value + 1)`) to address strong right-skew in raw market values

### Model
- **Linear Regression** on: Age, Goals, Assists, Shots, Touches, Progressive Carries, Possession Won, ground-duel win %, Position, Club Tier
- Multicollinearity checked via VIF (all features < 4 after correctly including a constant term — no problematic collinearity)

## Results

| Metric | Value |
|---|---|
| R² (single 80/20 split, log space) | 0.470 |
| R² (5-fold cross-validation, mean) | 0.544 (range: 0.37–0.71) |
| MAE (test set, € terms) | €9.35M |
| Baseline MAE (always predict median) | €15.18M |
| **Improvement over baseline** | **38.4%** |

### Feature Importance (standardized coefficients)
1. **Age** (-0.68) — by far the strongest predictor; younger players valued higher, holding stats equal
2. **Touches** (+0.29) — proxy for playing involvement/importance to the team
3. **Club Tier: MidTable** (-0.19) — see limitations below re: this feature's imperfections
4. **Shots** (+0.20), **Goals** (+0.16) — attacking output

## Known Limitations

- Performance data covers the 2024/25 season; market values reflect a later/current snapshot rather than a same-season figure. The model estimates "given last season's form, what is this player valued at today?" rather than a same-period valuation.
- Players from clubs relegated at the end of 2024/25 are excluded, since current-roster valuation data doesn't cover them.
- Goalkeepers are excluded and would need a separate model (different value drivers entirely).
- `club_tier`, derived from single-season league position, imperfectly proxies true club prestige/wage bill — it notably misclassifies historically "big" clubs (Manchester United, Tottenham) as low-tier after a below-par season, which can distort predictions for their players.
- The model systematically overvalues young or squad-rotation players at reputable clubs, and undervalues established veterans at smaller clubs. Adding `Minutes played` as a feature was tested as a fix and did not meaningfully improve this (R² unchanged) — the deeper issue is that no feature captures "market reputation independent of this season's stats," a factor real-world valuations clearly weight.
- Forwards are underrepresented (n=32 of 249) and have the widest value range, making their predictions the least reliable of the three position groups.
- Sample size (n=249) yields real cross-validation variance (R² 0.37–0.71 across folds); the 0.54 mean should be read as an estimate with meaningful uncertainty.

## Future Work
- Separate goalkeeper valuation model
- Better club-prestige feature (e.g., wage bill data or multi-season average league position, rather than single-season table position)
- Add a "reputation" proxy (international caps, years as a club regular) to address the young-player overvaluation pattern
- Expand to multiple seasons for a larger, more stable training set
- Deploy as a simple Streamlit app for interactive predictions

## Tech Stack
Python · pandas · scikit-learn · statsmodels · matplotlib

## Project Structure
```
├── README.md
├── requirements.txt
├── notebook.ipynb          # Full analysis, from data collection through modeling
└── data/                   # (not included — see Data Sources above)
```

## Disclaimer
Market value figures (via Transfermarkt) are crowd-sourced/algorithmic estimates, not actual transfer fees. This model estimates *market value*, not what a club would actually pay for a player.
