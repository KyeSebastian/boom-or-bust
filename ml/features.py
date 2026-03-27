"""
Pure feature engineering functions -- no I/O, no side effects.
All functions take DataFrames and return DataFrames.
"""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Position-specific stat columns used for rolling averages + trends
# ---------------------------------------------------------------------------

QB_STAT_COLS = [
    "passing_yards",
    "passing_tds",
    "interceptions",
    "passing_epa",
    "carries",
    "rushing_yards",
]

RB_STAT_COLS = [
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_epa",
    "targets",
    "receptions",
    "receiving_yards",
    "target_share",
]

WR_STAT_COLS = [
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_epa",
    "target_share",
    "air_yards_share",
    "wopr",
]

TE_STAT_COLS = WR_STAT_COLS  # same schema as WR

POSITION_STAT_COLS = {
    "QB": QB_STAT_COLS,
    "RB": RB_STAT_COLS,
    "WR": WR_STAT_COLS,
    "TE": TE_STAT_COLS,
}

# Red zone columns to roll + trend per position
RZ_STAT_COLS = {
    "QB": [],
    "RB": ["rz_targets", "rz_carries", "rz_target_share", "rz_carry_share"],
    "WR": ["rz_targets", "rz_target_share"],
    "TE": ["rz_targets", "rz_target_share"],
}


# ---------------------------------------------------------------------------
# Vegas / schedule features
# ---------------------------------------------------------------------------

def add_vegas_features(weekly: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Merges Vegas lines and rest-day info onto weekly stats.

    Pivot schedules from wide (one row/game) to long (one row/team/game),
    then compute implied_team_total and rest_days.

    Merge key: (season, week, team)
    """
    sched = schedules.copy()

    home = sched[["season", "week", "home_team", "away_team", "spread_line", "total_line", "home_rest"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent", "home_rest": "rest_days"})
    home["is_home"] = 1
    home["implied_team_total"] = (home["total_line"] / 2) - (home["spread_line"] / 2)

    away = sched[["season", "week", "away_team", "home_team", "spread_line", "total_line", "away_rest"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent", "away_rest": "rest_days"})
    away["is_home"] = 0
    away["implied_team_total"] = (away["total_line"] / 2) + (away["spread_line"] / 2)

    long_sched = pd.concat([home, away], ignore_index=True)
    long_sched["rest_days"] = long_sched["rest_days"].fillna(7)

    merge_cols = [
        "season", "week", "team",
        "spread_line", "total_line", "implied_team_total",
        "is_home", "rest_days",
    ]
    weekly = weekly.merge(long_sched[merge_cols], on=["season", "week", "team"], how="left")
    return weekly


# ---------------------------------------------------------------------------
# Defensive rank allowed
# ---------------------------------------------------------------------------

def compute_def_rank_allowed(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    For each (season, week, position), rank defenses by cumulative PPR points
    allowed to that position -- excluding the current week (no leakage).

    Returns DataFrame with columns:
        season, week, def_team, position, cum_pts_allowed, def_rank
    where def_rank 1 = easiest matchup (most points allowed).
    """
    df = weekly[["season", "week", "opponent_team", "position", "fantasy_points_ppr"]].copy()
    df = df.dropna(subset=["opponent_team", "fantasy_points_ppr"])

    pts = (
        df.groupby(["season", "week", "opponent_team", "position"], as_index=False)["fantasy_points_ppr"]
        .sum()
        .rename(columns={"opponent_team": "def_team", "fantasy_points_ppr": "pts_allowed_week"})
    )
    pts = pts.sort_values(["def_team", "position", "season", "week"])

    pts["cum_pts_allowed"] = (
        pts.groupby(["def_team", "position", "season"])["pts_allowed_week"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    pts["def_rank"] = (
        pts.groupby(["season", "week", "position"])["cum_pts_allowed"]
        .rank(ascending=False, method="min")
    )

    return pts[["season", "week", "def_team", "position", "cum_pts_allowed", "def_rank"]]


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    stat_cols: list[str],
    windows: list[int] = [3, 4],
) -> pd.DataFrame:
    """
    For each stat column and window, adds a rolling mean using the previous
    `window` games (shift(1) prevents current-week leakage).

    Groups by (player_id, season) to prevent rolling across season boundaries.

    Adds columns: {stat}_roll{window} for each stat x window combination.
    """
    df = df.sort_values(["player_id", "season", "week"]).copy()

    for col in stat_cols:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll{w}"] = (
                df.groupby(["player_id", "season"])[col]
                .transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
            )
    return df


# ---------------------------------------------------------------------------
# Trend features
# ---------------------------------------------------------------------------

def add_trend_features(df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    """
    Trend = roll3 - roll4 for each stat column.

    Positive value = player is improving (recent 3-game avg above 4-game avg).
    Negative value = player is declining or losing usage.

    Captures momentum shifts like a rookie earning more snaps each week,
    or a veteran losing targets to an emerging teammate.
    """
    for col in stat_cols:
        r3 = f"{col}_roll3"
        r4 = f"{col}_roll4"
        if r3 in df.columns and r4 in df.columns:
            df[f"{col}_trend"] = df[r3] - df[r4]
    return df


# ---------------------------------------------------------------------------
# Snap count features
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, remove common suffixes."""
    name = str(name).lower()
    name = re.sub(r"[''`]", "", name)
    name = re.sub(r"[^a-z ]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return name.strip()


def add_snap_features(weekly: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """
    Merges offense_snaps and offense_pct from snap-count data onto weekly.

    Uses normalized player names + team + week as a fallback key when
    player_id (GSIS) doesn't directly match PFR IDs in snap data.

    Rows with no snap match are filled with the positional-week median
    so no rows are dropped.
    """
    snaps = snaps.copy()
    weekly = weekly.copy()

    name_col = "player" if "player" in snaps.columns else "player_name"
    snaps["_name_norm"] = snaps[name_col].apply(_normalize_name)

    name_col_w = "player_display_name" if "player_display_name" in weekly.columns else "player_name"
    weekly["_name_norm"] = weekly[name_col_w].apply(_normalize_name)

    snap_merge_cols = ["_name_norm", "team", "season", "week", "offense_snaps", "offense_pct"]
    available = [c for c in snap_merge_cols if c in snaps.columns]
    merged = weekly.merge(snaps[available], on=["_name_norm", "team", "season", "week"], how="left")
    merged = merged.drop(columns=["_name_norm"], errors="ignore")

    for col in ["offense_snaps", "offense_pct"]:
        if col in merged.columns:
            medians = merged.groupby(["season", "week", "position"])[col].transform("median")
            merged[col] = merged[col].fillna(medians)

    return merged


# ---------------------------------------------------------------------------
# Red zone features
# ---------------------------------------------------------------------------

def add_rz_features(
    df: pd.DataFrame,
    rz_stats: pd.DataFrame,
    position: str,
) -> pd.DataFrame:
    """
    Merges red zone stats (targets, carries, share) onto player-week rows,
    then adds rolling averages and trend features for those columns.

    Merge key: (season, week, player_id)
    """
    rz_cols = ["season", "week", "player_id", "rz_targets", "rz_carries",
               "rz_target_share", "rz_carry_share"]
    available = [c for c in rz_cols if c in rz_stats.columns]

    df = df.merge(rz_stats[available], on=["season", "week", "player_id"], how="left")

    for col in ["rz_targets", "rz_carries", "rz_target_share", "rz_carry_share"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    pos_rz_cols = [c for c in RZ_STAT_COLS.get(position, []) if c in df.columns]
    if pos_rz_cols:
        df = add_rolling_features(df, pos_rz_cols)
        df = add_trend_features(df, pos_rz_cols)

    return df


# ---------------------------------------------------------------------------
# Position-specific feature builders
# ---------------------------------------------------------------------------

def _build_position_features(
    weekly: pd.DataFrame,
    schedules: pd.DataFrame,
    snaps: pd.DataFrame,
    def_rank: pd.DataFrame,
    rz_stats: pd.DataFrame,
    position: str,
) -> pd.DataFrame:
    """
    Generic builder used by all position-specific functions.
    """
    stat_cols = POSITION_STAT_COLS[position]

    df = weekly[weekly["position"] == position].copy()

    df = add_vegas_features(df, schedules)
    df = add_snap_features(df, snaps)
    df = add_rolling_features(df, stat_cols)
    df = add_trend_features(df, stat_cols)
    df = add_rz_features(df, rz_stats, position)

    # Merge defensive rank
    pos_def = def_rank[def_rank["position"] == position].copy()
    pos_def = pos_def.rename(columns={
        "def_team": "opponent_team",
        "cum_pts_allowed": "cum_pts_allowed",
        "def_rank": f"def_rank_vs_{position.lower()}",
    })
    df = df.merge(
        pos_def[["season", "week", "opponent_team", "cum_pts_allowed", f"def_rank_vs_{position.lower()}"]],
        on=["season", "week", "opponent_team"],
        how="left",
    )

    # Drop week 1 (no prior rolling data, all def_rank NaN due to shift)
    df = df[df["week"] > 1]

    return df


def build_qb_features(weekly, schedules, snaps, def_rank, rz_stats) -> pd.DataFrame:
    return _build_position_features(weekly, schedules, snaps, def_rank, rz_stats, "QB")


def build_rb_features(weekly, schedules, snaps, def_rank, rz_stats) -> pd.DataFrame:
    return _build_position_features(weekly, schedules, snaps, def_rank, rz_stats, "RB")


def build_wr_features(weekly, schedules, snaps, def_rank, rz_stats) -> pd.DataFrame:
    return _build_position_features(weekly, schedules, snaps, def_rank, rz_stats, "WR")


def build_te_features(weekly, schedules, snaps, def_rank, rz_stats) -> pd.DataFrame:
    return _build_position_features(weekly, schedules, snaps, def_rank, rz_stats, "TE")


POSITION_BUILDERS = {
    "QB": build_qb_features,
    "RB": build_rb_features,
    "WR": build_wr_features,
    "TE": build_te_features,
}
