"""
ml/train.py — Step 2: Model training for Fantasy Football Start-or-Sit AI.

Trains one XGBoost regression model per position (QB, RB, WR, TE).
Uses a time-based train/test split to prevent data leakage:
    Train: weeks 2–14
    Test:  weeks 15–18

Outputs per position (saved to models/{pos}/):
    model.ubj       — trained XGBoost model
    features.json   — ordered list of feature columns the model expects

Usage:
    python -m ml.train
    python -m ml.train --positions QB WR
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")
POSITIONS = ["QB", "RB", "WR", "TE"]

TARGET = "fantasy_points_ppr"
TRAIN_WEEKS = range(2, 15)   # weeks 2–14
TEST_WEEKS  = range(15, 19)  # weeks 15–18

# Columns to drop before training — identifiers, raw stats, and leaky targets
DROP_COLS = [
    "player_id", "player_name", "player_display_name",
    "position", "position_group", "headshot_url",
    "team", "season", "week", "season_type", "opponent_team",
    # raw stat columns (model uses rolling averages instead)
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost",
    "passing_air_yards", "passing_yards_after_catch", "passing_first_downs",
    "passing_2pt_conversions", "pacr", "dakota",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
    "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa",
    "rushing_2pt_conversions",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
    "receiving_2pt_conversions", "racr", "target_share", "air_yards_share",
    "wopr", "special_teams_tds",
    # other target (non-PPR) -- keep only fantasy_points_ppr
    "fantasy_points",
    # raw red zone counts -- model uses rolling + trend versions instead
    "rz_targets", "rz_carries", "rz_target_share", "rz_carry_share",
]

XGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(position: str) -> pd.DataFrame:
    path = PROC_DIR / f"{position.lower()}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}. Run ml.pipeline first.")
    return pd.read_csv(path)


def build_xy(df: pd.DataFrame, weeks) -> tuple[pd.DataFrame, pd.Series]:
    """Filter to given weeks, drop non-feature columns, return (X, y)."""
    subset = df[df["week"].isin(weeks)].copy()
    y = subset[TARGET]
    X = subset.drop(columns=[c for c in DROP_COLS + [TARGET] if c in subset.columns])
    return X, y


def train_position(position: str) -> dict:
    print(f"\n--- {position} ---")

    df = load_features(position)

    X_train, y_train = build_xy(df, TRAIN_WEEKS)
    X_test,  y_test  = build_xy(df, TEST_WEEKS)

    print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"  Features: {X_train.shape[1]}")

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"  MAE : {mae:.3f} pts")
    print(f"  RMSE: {rmse:.3f} pts")

    # Feature importances (top 10)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top10 = importances.nlargest(10)
    print("  Top features:")
    for feat, score in top10.items():
        print(f"    {feat:<35} {score:.4f}")

    # Save model and feature list
    out_dir = MODELS_DIR / position.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.ubj"
    model.save_model(model_path)

    features_path = out_dir / "features.json"
    features_path.write_text(json.dumps(X_train.columns.tolist(), indent=2))

    print(f"  Saved -> {model_path}")
    print(f"  Saved -> {features_path}")

    return {
        "position": position,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_features": X_train.shape[1],
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_training(positions: list[str] = POSITIONS) -> list[dict]:
    print("\n=== Fantasy Football Model Training | Step 2 ===")
    print(f"  Train weeks : 2-14")
    print(f"  Test weeks  : 15-18")
    print(f"  Algorithm   : XGBoost")

    results = []
    for pos in positions:
        result = train_position(pos)
        results.append(result)

    print("\n=== Summary ===")
    print(f"  {'POS':<5} {'MAE':>7} {'RMSE':>7}")
    print(f"  {'-'*22}")
    for r in results:
        print(f"  {r['position']:<5} {r['mae']:>7.3f} {r['rmse']:>7.3f}")

    print("\n=== Training complete ===\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost models per position.")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=POSITIONS,
        choices=POSITIONS,
        help="Positions to train (default: all)",
    )
    args = parser.parse_args()
    run_training(positions=args.positions)
