"""
ml/predict.py — Step 4: Prediction inference layer for Fantasy Football Start-or-Sit AI.

Provides a Predictor class that:
    - Loads trained XGBoost models (lazy-loaded and cached per position)
    - Accepts a position, season, week, and optional player filter
    - Returns predicted PPR fantasy points with start/sit recommendations

Recommendation logic:
    START  — predicted score at or above the positional median for that week
    SIT    — predicted score below the positional median for that week
    BOOM   — predicted score in the top 25% for that week
    BUST   — predicted score in the bottom 25% for that week

Output per player:
    {
        "player_name": str,
        "team":        str,
        "position":    str,
        "season":      int,
        "week":        int,
        "predicted_pts": float,
        "rank":        int,          # 1 = highest predicted
        "recommendation": "START" | "SIT",
        "flag":        "BOOM" | "BUST" | null
    }

Usage:
    python -m ml.predict --position RB --season 2024 --week 16
    python -m ml.predict --position WR --season 2024 --week 17 --players "Tyreek Hill" "Davante Adams"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROC_DIR   = Path("data/processed")
MODELS_DIR = Path("models")
POSITIONS  = ["QB", "RB", "WR", "TE"]

BOOM_PCT = 75
BUST_PCT = 25


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """
    Inference wrapper around trained XGBoost position models.

    Models are loaded once per position and cached for the lifetime of the
    instance, so it is safe to reuse a single Predictor across many requests.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[xgb.XGBRegressor, list[str]]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, position: str) -> tuple[xgb.XGBRegressor, list[str]]:
        pos = position.upper()
        if pos not in self._cache:
            model_path    = MODELS_DIR / pos.lower() / "model.ubj"
            features_path = MODELS_DIR / pos.lower() / "features.json"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"No model found at {model_path}. Run `python -m ml.train` first."
                )

            model = xgb.XGBRegressor()
            model.load_model(model_path)
            features = json.loads(features_path.read_text())
            self._cache[pos] = (model, features)

        return self._cache[pos]

    def _load_week(self, position: str, season: int, week: int) -> pd.DataFrame:
        """Load processed features for a position, filtered to one season-week."""
        path = PROC_DIR / f"{position.lower()}_features.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {path}. Run `python -m ml.pipeline` first."
            )
        df = pd.read_csv(path)
        mask = (df["season"] == season) & (df["week"] == week)
        week_df = df[mask].copy()
        if week_df.empty:
            raise ValueError(
                f"No {position} data found for season={season} week={week}."
            )
        return week_df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        position: str,
        season: int,
        week: int,
        players: list[str] | None = None,
    ) -> list[dict]:
        """
        Generate start/sit predictions for a position in a given season-week.

        Parameters
        ----------
        position : str
            One of QB, RB, WR, TE (case-insensitive).
        season : int
            NFL season year (e.g. 2024).
        week : int
            Regular-season week number.
        players : list[str] | None
            Optional list of player names to filter results to. Matching is
            case-insensitive substring search. If None, all players are returned.

        Returns
        -------
        list[dict]
            Ranked list of player predictions, sorted by predicted_pts descending.
        """
        pos = position.upper()
        if pos not in POSITIONS:
            raise ValueError(f"Unknown position '{position}'. Choose from {POSITIONS}.")

        model, features = self._load(pos)
        full_df = self._load_week(pos, season, week)

        # Run predictions on the full pool first — rank and thresholds must
        # reflect the entire positional pool, not just the filtered subset.
        X_full     = full_df.reindex(columns=features, fill_value=0.0)
        preds_full = model.predict(X_full)

        median_thresh = float(np.median(preds_full))
        boom_thresh   = float(np.percentile(preds_full, BOOM_PCT))
        bust_thresh   = float(np.percentile(preds_full, BUST_PCT))

        # Build full ranked results
        results = full_df[["player_display_name", "team", "opponent_team"]].copy().reset_index(drop=True)
        results["predicted_pts"] = np.round(preds_full, 2)
        results = results.sort_values("predicted_pts", ascending=False).reset_index(drop=True)
        results["rank"] = results.index + 1

        # Filter to requested players after rank is assigned
        if players:
            pattern = "|".join(p.lower() for p in players)
            mask = results["player_display_name"].str.lower().str.contains(pattern, regex=True, na=False)
            results = results[mask]
            if results.empty:
                return []

        def _recommendation(pts: float) -> str:
            return "START" if pts >= median_thresh else "SIT"

        def _flag(pts: float) -> str | None:
            if pts >= boom_thresh:
                return "BOOM"
            if pts <= bust_thresh:
                return "BUST"
            return None

        results["recommendation"] = results["predicted_pts"].map(_recommendation)
        results["flag"]           = results["predicted_pts"].map(_flag)

        output = []
        for _, row in results.iterrows():
            output.append({
                "player_name":    row["player_display_name"],
                "team":           row["team"],
                "opponent_team":  row["opponent_team"],
                "position":       pos,
                "season":         season,
                "week":           week,
                "predicted_pts":  float(row["predicted_pts"]),
                "rank":           int(row["rank"]),
                "recommendation": row["recommendation"],
                "flag":           row["flag"],
            })

        return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results(results: list[dict]) -> None:
    if not results:
        print("  (no results)")
        return

    header = f"  {'Rank':<5} {'Player':<25} {'Team':<5} {'Pts':>6}  {'Rec':>5}  {'Flag'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        flag = r["flag"] or ""
        print(
            f"  {r['rank']:<5} {r['player_name']:<25} {r['team']:<5} "
            f"{r['predicted_pts']:>6.2f}  {r['recommendation']:>5}  {flag}"
        )


def run_predict(
    positions: list[str],
    season: int,
    week: int,
    players: list[str] | None,
) -> None:
    print(f"\n=== Fantasy Football Predictions | Step 4 ===")
    print(f"  Season : {season}")
    print(f"  Week   : {week}")
    if players:
        print(f"  Filter : {', '.join(players)}")

    predictor = Predictor()

    for pos in positions:
        print(f"\n--- {pos} ---")
        try:
            results = predictor.predict(pos, season, week, players)
            _print_results(results)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ERROR: {exc}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run start/sit predictions for a given week.")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=POSITIONS,
        choices=POSITIONS,
        metavar="POS",
        help="Positions to predict (default: all). E.g. --positions RB WR",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2024,
        help="NFL season year (default: 2024)",
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="Regular-season week number",
    )
    parser.add_argument(
        "--players",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Filter to specific player names (case-insensitive substring match)",
    )
    args = parser.parse_args()
    run_predict(args.positions, args.season, args.week, args.players)
