# Boom or Bust - Fantasy Football Start/Sit Advisor

A machine learning tool that tells you who to start and who to sit in fantasy football. Enter two players, pick a week, and get a head-to-head verdict backed by real numbers: matchup quality, recent form, Vegas lines, snap rate, and red zone usage.

---

## How It Works

The core of this project is four XGBoost models, one per position (QB, RB, WR, TE), trained on four seasons of NFL data (2021-2024). Each model takes in a set of engineered features for a given player-week and outputs a predicted PPR fantasy score.

**Features fed into each model:**
- Rolling 3 and 4-game averages for position-relevant stats (passing yards, rushing yards, receiving yards, targets, carries)
- Trend signal (3-game average minus 4-game average, catches players heating up or cooling down)
- Defensive rank against each position and cumulative points allowed
- Vegas implied team total and spread line
- Snap rate and red zone target/carry share (3-game rolling)

**Start/Sit logic:**
- Each week, every player in the dataset gets a predicted score
- The positional median for that week is the threshold. Above it is START, below is SIT.
- BOOM flag = top 25% of that position for the week
- BUST flag = bottom 25%

**Head-to-head compare:**
- Both players are run through their respective position models independently
- The higher predicted score wins — recommendation overrides to START/SIT based on the matchup

**Plain-English reasoning:**
- After prediction, a rules-based engine converts the raw feature values into 3-4 readable sentences covering matchup, form, Vegas context, and usage
- This is not auto-generated text. It is deterministic logic built around the same features the model uses.

---

## Model Accuracy

Measured on a held-out test set (Weeks 15-18, 2024 season) using start/sit accuracy — the percentage of times the model correctly identified whether a player should start or sit relative to the positional median.

| Position | Start/Sit Accuracy |
|----------|--------------------|
| QB       | 83.4%              |
| RB       | 76.7%              |
| WR       | 71.7%              |
| TE       | 69.0%              |

TE accuracy reflects how top-heavy the position is — a handful of elite tight ends dominate, making the rest genuinely hard to separate.

---

## Data Sources

All data pulled via `nfl_data_py`:
- Weekly player stats (4 seasons: 2021–2024)
- NFL schedules and opponent data
- Snap counts and participation rates
- Play-by-play data for red zone stats

---

## Tech Stack

- **Python** — data pipeline, feature engineering, model training, inference
- **XGBoost** — gradient boosting models for each position
- **FastAPI** — REST API backend
- **Pandas / NumPy** — data processing
- **Vanilla JS / HTML / CSS** — frontend UI (no framework)

---

## Running Locally

**Requirements:** Python 3.10+

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload
```

Open `http://localhost:8000` in your browser.

> The processed feature data (`data/processed/`) is required to run the app. If it is missing, run `python -m ml.pipeline` first to regenerate it from the raw data.

---

## Project Structure

```
ml/
  pipeline.py     # data ingestion and feature engineering
  train.py        # model training (one XGBoost model per position)
  evaluate.py     # accuracy evaluation on held-out test weeks
  predict.py      # inference layer, takes player + week, returns ranked predictions
  reasoning.py    # rules-based plain-English explanation generator

app/
  main.py         # FastAPI app with /compare, /players/search, /latest-week endpoints
  static/
    index.html    # frontend UI

models/
  {qb,rb,wr,te}/
    model.ubj     # trained XGBoost model
    features.json # feature list used during training
    eval.json     # accuracy results
```
