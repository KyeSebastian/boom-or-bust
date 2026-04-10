"""
ml/evaluate.py: Evaluates each position model on the held-out test set (weeks 15-18).

Metrics computed:
    mae: mean absolute error in points
    rmse: root mean squared error in points
    start_sit_accuracy: percentage of head-to-head pairs where the model picked the higher scorer
    boom_precision: of players the model flagged as boom, how many actually scored in the top 25%
    bust_precision: of players the model flagged as bust, how many actually scored in the bottom 25%
    boom_recall: of actual boom weeks, how many did the model catch
    bust_recall: of actual bust weeks, how many did the model catch

Results are saved to models/{pos}/eval.json.

Usage:
    python -m ml.evaluate
    python -m ml.evaluate --positions QB WR
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROC_DIR   = Path("data/processed")
MODELS_DIR = Path("models")
POSITIONS  = ["QB", "RB", "WR", "TE"]

TARGET     = "fantasy_points_ppr"
TEST_WEEKS = range(15, 19)

# Percentile thresholds for boom / bust classification
BOOM_PCT = 75
BUST_PCT = 25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_features(position: str):
    model_path    = MODELS_DIR / position.lower() / "model.ubj"
    features_path = MODELS_DIR / position.lower() / "features.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}. Run ml.train first.")

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    features = json.loads(features_path.read_text())
    return model, features


def load_test_data(position: str, features: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    path = PROC_DIR / f"{position.lower()}_features.csv"
    df   = pd.read_csv(path)
    test = df[df["week"].isin(TEST_WEEKS)].copy()

    y    = test[TARGET].reset_index(drop=True)
    X    = test[features].reset_index(drop=True)
    meta = test[["player_display_name", "team", "season", "week"]].reset_index(drop=True)
    return X, y, meta


def start_sit_accuracy(y_true: pd.Series, y_pred: np.ndarray, meta: pd.DataFrame) -> float:
    """
    For every pair of players at the same position in the same season-week,
    check whether the model correctly ranked the higher scorer on top.

    Returns the fraction of pairs where the model was correct.
    """
    df = meta.copy()
    df["actual"] = y_true.values
    df["pred"]   = y_pred

    correct = 0
    total   = 0

    for (season, week), group in df.groupby(["season", "week"]):
        if len(group) < 2:
            continue
        idx = group.index.tolist()
        for i, j in combinations(idx, 2):
            a_actual, b_actual = df.loc[i, "actual"], df.loc[j, "actual"]
            a_pred,   b_pred   = df.loc[i, "pred"],   df.loc[j, "pred"]
            if a_actual == b_actual:
                continue
            model_picked_higher = (a_pred > b_pred) == (a_actual > b_actual)
            correct += int(model_picked_higher)
            total   += 1

    return correct / total if total > 0 else 0.0


def boom_bust_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
) -> dict:
    """
    Boom = actual score in top BOOM_PCT percentile for that position-week.
    Bust = actual score in bottom BUST_PCT percentile for that position-week.

    Computes precision and recall for both labels.
    """
    df = meta.copy()
    df["actual"] = y_true.values
    df["pred"]   = y_pred

    # Compute per-week percentile thresholds
    week_stats = df.groupby(["season", "week"])["actual"].agg(
        boom_thresh=lambda s: np.percentile(s, BOOM_PCT),
        bust_thresh=lambda s: np.percentile(s, BUST_PCT),
    ).reset_index()

    df = df.merge(week_stats, on=["season", "week"])

    pred_stats = df.groupby(["season", "week"])["pred"].agg(
        pred_boom_thresh=lambda s: np.percentile(s, BOOM_PCT),
        pred_bust_thresh=lambda s: np.percentile(s, BUST_PCT),
    ).reset_index()

    df = df.merge(pred_stats, on=["season", "week"])

    df["is_boom"]      = df["actual"] >= df["boom_thresh"]
    df["is_bust"]      = df["actual"] <= df["bust_thresh"]
    df["pred_boom"]    = df["pred"]   >= df["pred_boom_thresh"]
    df["pred_bust"]    = df["pred"]   <= df["pred_bust_thresh"]

    def precision(predicted, actual):
        tp = (predicted & actual).sum()
        return tp / predicted.sum() if predicted.sum() > 0 else 0.0

    def recall(predicted, actual):
        tp = (predicted & actual).sum()
        return tp / actual.sum() if actual.sum() > 0 else 0.0

    return {
        "boom_precision": round(precision(df["pred_boom"], df["is_boom"]), 3),
        "boom_recall":    round(recall(df["pred_boom"],    df["is_boom"]), 3),
        "bust_precision": round(precision(df["pred_bust"], df["is_bust"]), 3),
        "bust_recall":    round(recall(df["pred_bust"],    df["is_bust"]), 3),
    }


def biggest_misses(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    n: int = 5,
) -> list[dict]:
    """Returns the n largest absolute prediction errors with player context."""
    df = meta.copy()
    df["actual"] = y_true.values
    df["pred"]   = y_pred
    df["error"]  = (df["pred"] - df["actual"]).abs()
    df = df.nlargest(n, "error")

    return [
        {
            "player": row["player_display_name"],
            "team":   row["team"],
            "season": int(row["season"]),
            "week":   int(row["week"]),
            "actual": round(row["actual"], 1),
            "pred":   round(row["pred"], 1),
            "error":  round(row["error"], 1),
        }
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_position(position: str) -> dict:
    print(f"\n--- {position} ---")

    model, features = load_model_and_features(position)
    X_test, y_test, meta = load_test_data(position, features)

    preds = model.predict(X_test)

    mae  = round(mean_absolute_error(y_test, preds), 3)
    rmse = round(root_mean_squared_error(y_test, preds), 3)
    ss   = round(start_sit_accuracy(y_test, preds, meta), 3)
    bb   = boom_bust_metrics(y_test, preds, meta)
    misses = biggest_misses(y_test, preds, meta)

    print(f"  MAE              : {mae} pts")
    print(f"  RMSE             : {rmse} pts")
    print(f"  Start/Sit Acc.   : {ss*100:.1f}%")
    print(f"  Boom precision   : {bb['boom_precision']*100:.1f}%  |  recall: {bb['boom_recall']*100:.1f}%")
    print(f"  Bust precision   : {bb['bust_precision']*100:.1f}%  |  recall: {bb['bust_recall']*100:.1f}%")
    print(f"  Biggest misses:")
    for m in misses:
        print(f"    {m['player']:<22} Wk{m['week']} {m['season']}  actual={m['actual']:5.1f}  pred={m['pred']:5.1f}  err={m['error']:4.1f}")

    result = {
        "position": position,
        "test_rows": len(y_test),
        "mae": mae,
        "rmse": rmse,
        "start_sit_accuracy": ss,
        **bb,
        "biggest_misses": misses,
    }

    out_path = MODELS_DIR / position.lower() / "eval.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out_path}")

    return result


def run_evaluation(positions: list[str] = POSITIONS) -> list[dict]:
    print("\n=== Fantasy Football Model Evaluation | Step 3 ===")
    print(f"  Test set: weeks 15-18 (all seasons)")

    results = []
    for pos in positions:
        results.append(evaluate_position(pos))

    print("\n=== Summary ===")
    print(f"  {'POS':<5} {'MAE':>6} {'RMSE':>6} {'SS Acc':>8} {'Boom P':>8} {'Bust P':>8}")
    print(f"  {'-'*45}")
    for r in results:
        print(
            f"  {r['position']:<5} "
            f"{r['mae']:>6.3f} "
            f"{r['rmse']:>6.3f} "
            f"{r['start_sit_accuracy']*100:>7.1f}% "
            f"{r['boom_precision']*100:>7.1f}% "
            f"{r['bust_precision']*100:>7.1f}%"
        )

    print("\n=== Evaluation complete ===\n")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models on the held-out test set.")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=POSITIONS,
        choices=POSITIONS,
        help="Positions to evaluate (default: all)",
    )
    args = parser.parse_args()
    run_evaluation(positions=args.positions)
