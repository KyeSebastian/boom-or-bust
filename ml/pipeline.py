"""
ml/pipeline.py: Data pipeline for the fantasy football start/sit tool.

Usage:
    python -m ml.pipeline                        # default: 2021-2024, no force refresh
    python -m ml.pipeline --years 2023 2024
    python -m ml.pipeline --force-refresh
"""

import argparse
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
from tqdm import tqdm

from ml.features import (
    compute_def_rank_allowed,
    POSITION_BUILDERS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASON_YEARS = [2021, 2022, 2023, 2024]
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
POSITIONS = ["QB", "RB", "WR", "TE"]

# Minimal PBP columns needed to compute red zone stats (keeps download small)
PBP_COLS = [
    "season", "week", "season_type", "posteam",
    "receiver_player_id", "rusher_player_id",
    "yardline_100", "complete_pass", "incomplete_pass", "rush_attempt",
]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_or_fetch(name: str, fetch_fn, year: int, force: bool) -> pd.DataFrame:
    """
    Return cached parquet if it exists (and force is False), otherwise
    call fetch_fn(year), cache to parquet, and return the result.
    """
    path = RAW_DIR / f"{name}_{year}.parquet"
    if path.exists() and not force:
        print(f"  [cache] Loading {path}")
        return pd.read_parquet(path)

    print(f"  [fetch] Downloading {name} for {year}...")
    df = fetch_fn(year)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  [cache] Saved to {path}")
    return df


def _load_multi_year(name: str, fetch_fn, years: list[int], force: bool) -> pd.DataFrame:
    """Load or fetch data for each year and concatenate."""
    frames = [_load_or_fetch(name, fetch_fn, y, force) for y in years]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Fetch functions
# ---------------------------------------------------------------------------

def fetch_weekly_stats(year: int) -> pd.DataFrame:
    df = nfl.import_weekly_data([year])
    return df[df["season_type"] == "REG"].copy()


def fetch_schedules(year: int) -> pd.DataFrame:
    df = nfl.import_schedules([year])
    return df[df["game_type"] == "REG"].copy()


def fetch_snap_counts(year: int) -> pd.DataFrame:
    df = nfl.import_snap_counts([year])
    return df.copy()


def fetch_rz_stats(year: int) -> pd.DataFrame:
    """
    Pulls red zone stats (inside the opponents' 20-yard line) from play-by-play data.

    Computes per player per week:
        rz_targets: pass targets received in the red zone
        rz_carries: rush attempts in the red zone
        rz_target_share: player rz_targets divided by team total rz_targets
        rz_carry_share: player rz_carries divided by team total rz_carries

    Merge key for downstream use: (season, week, player_id)
    """
    print(f"  [fetch] Downloading play-by-play for {year} (red zone stats)...")
    pbp = nfl.import_pbp_data([year], columns=PBP_COLS, include_participation=False)
    pbp = pbp[(pbp["season_type"] == "REG") & (pbp["yardline_100"] <= 20)].copy()

    # RZ targets: pass plays (complete or incomplete) with a receiver
    rz_pass = pbp[
        ((pbp["complete_pass"] == 1) | (pbp["incomplete_pass"] == 1)) &
        pbp["receiver_player_id"].notna()
    ]
    rz_targets = (
        rz_pass
        .groupby(["season", "week", "posteam", "receiver_player_id"], as_index=False)
        .size()
        .rename(columns={"receiver_player_id": "player_id", "posteam": "team", "size": "rz_targets"})
    )

    # RZ carries: rush attempts with a rusher
    rz_rush = pbp[
        (pbp["rush_attempt"] == 1) &
        pbp["rusher_player_id"].notna()
    ]
    rz_carries = (
        rz_rush
        .groupby(["season", "week", "posteam", "rusher_player_id"], as_index=False)
        .size()
        .rename(columns={"rusher_player_id": "player_id", "posteam": "team", "size": "rz_carries"})
    )

    # Team totals for share calculation
    team_rz_tgt = (
        rz_targets.groupby(["season", "week", "team"])["rz_targets"]
        .sum().reset_index(name="team_rz_targets")
    )
    team_rz_car = (
        rz_carries.groupby(["season", "week", "team"])["rz_carries"]
        .sum().reset_index(name="team_rz_carries")
    )

    # Combine targets and carries
    result = rz_targets.merge(rz_carries, on=["season", "week", "team", "player_id"], how="outer")
    result = result.merge(team_rz_tgt, on=["season", "week", "team"], how="left")
    result = result.merge(team_rz_car, on=["season", "week", "team"], how="left")

    result["rz_targets"] = result["rz_targets"].fillna(0)
    result["rz_carries"] = result["rz_carries"].fillna(0)
    result["rz_target_share"] = (result["rz_targets"] / result["team_rz_targets"]).fillna(0)
    result["rz_carry_share"] = (result["rz_carries"] / result["team_rz_carries"]).fillna(0)

    return result[["season", "week", "team", "player_id",
                   "rz_targets", "rz_carries", "rz_target_share", "rz_carry_share"]]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    years: list[int] = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Top-level entry point. Returns a dict mapping each position to its feature DataFrame.
    """
    if years is None:
        years = SEASON_YEARS

    print(f"\n=== Fantasy Football Data Pipeline | Seasons {years[0]}-{years[-1]} ===\n")

    # 1. Load or fetch raw data for all years
    weekly_raw = _load_multi_year("weekly_stats", fetch_weekly_stats, years, force_refresh)
    schedules  = _load_multi_year("schedules",    fetch_schedules,    years, force_refresh)
    snaps      = _load_multi_year("snap_counts",  fetch_snap_counts,  years, force_refresh)
    rz_stats   = _load_multi_year("rz_stats",     fetch_rz_stats,     years, force_refresh)

    # Standardize column names
    weekly_raw = weekly_raw.rename(columns={"recent_team": "team"})

    print(f"\n  weekly_stats rows : {len(weekly_raw):,}")
    print(f"  schedules rows   : {len(schedules):,}")
    print(f"  snap_counts rows : {len(snaps):,}")
    print(f"  rz_stats rows    : {len(rz_stats):,}\n")

    # 2. Pre-compute defensive rank (season-wide, all positions)
    print("  Computing defensive rank allowed...")
    def_rank = compute_def_rank_allowed(weekly_raw)

    # 3. Build position-specific feature sets
    features_by_pos: dict[str, pd.DataFrame] = {}

    for pos in tqdm(POSITIONS, desc="Building position features"):
        builder = POSITION_BUILDERS[pos]
        df = builder(weekly_raw, schedules, snaps, def_rank, rz_stats)
        features_by_pos[pos] = df
        print(f"    {pos}: {len(df):,} rows, {len(df.columns)} columns")

    # 4. Save outputs
    save_outputs(features_by_pos)

    print("\n=== Pipeline complete ===\n")
    return features_by_pos


def save_outputs(features_by_pos: dict[str, pd.DataFrame]) -> None:
    """Write one CSV per position to data/processed/."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    for pos, df in features_by_pos.items():
        path = PROC_DIR / f"{pos.lower()}_features.csv"
        df.to_csv(path, index=False)
        print(f"  [save] {path}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pull, clean, and cache NFL data; compute ML features by position."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=SEASON_YEARS,
        help=f"NFL season years to process (default: {SEASON_YEARS})",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Re-download raw data even if cached parquet files exist.",
    )
    args = parser.parse_args()
    run_pipeline(years=args.years, force_refresh=args.force_refresh)
