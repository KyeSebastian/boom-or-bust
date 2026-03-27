"""
ml/reasoning.py — Step 5a: Plain-English reasoning generator.

Converts raw feature values into 3–4 readable sentences a fantasy manager
can absorb in seconds. No stats jargon — just the key factors.

Sentence structure (position-aware):
    1. Matchup quality       — defensive rank + points allowed
    2. Recent form           — rolling averages + trend direction
    3. Vegas game context    — implied team total + game script
    4. Usage / red zone      — snap rate or red zone looks (conditional)
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(row: dict, key: str, default: float = float("nan")) -> float:
    val = row.get(key, default)
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _ordinal(n: int) -> str:
    n = int(n)
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    if 11 <= n % 100 <= 13:
        suffix = "th"
    return f"{n}{suffix}"


# ---------------------------------------------------------------------------
# Sentence builders
# ---------------------------------------------------------------------------

def _sentence_matchup(player_name: str, position: str, row: dict) -> str:
    pos = position.upper()
    rank_key = f"def_rank_vs_{pos.lower()}"
    rank = _safe(row, rank_key)
    pts_allowed = _safe(row, "cum_pts_allowed")

    if math.isnan(rank):
        return f"{player_name} has an uncharted matchup this week with limited defensive data available."

    rank_int = int(round(rank))
    ord_rank = _ordinal(rank_int)

    if rank_int <= 8:
        quality = "favorable"
        tone = "one of the most generous defenses in the league"
    elif rank_int <= 16:
        quality = "slightly favorable"
        tone = "a defense that has been giving up points this season"
    elif rank_int <= 24:
        quality = "tough"
        tone = "a defense that has been stingy against this position"
    else:
        quality = "very tough"
        tone = "one of the stingiest defenses in the league"

    pts_str = f", allowing an average of {pts_allowed:.1f} fantasy points per game to {pos}s" if not math.isnan(pts_allowed) else ""

    return (
        f"{player_name} draws a {quality} matchup this week against {tone}{pts_str}."
    )


def _sentence_form(player_name: str, position: str, row: dict) -> str:
    pos = position.upper()

    if pos == "QB":
        primary_stat = "passing_yards"
        primary_label = "passing yards"
        secondary_stat = "rushing_yards"
        secondary_label = "rushing yards"
    elif pos == "RB":
        primary_stat = "rushing_yards"
        primary_label = "rushing yards"
        secondary_stat = "carries"
        secondary_label = "carries"
    else:  # WR / TE
        primary_stat = "receiving_yards"
        primary_label = "receiving yards"
        secondary_stat = "targets"
        secondary_label = "targets"

    roll3 = _safe(row, f"{primary_stat}_roll3")
    roll4 = _safe(row, f"{primary_stat}_roll4")
    trend = _safe(row, f"{primary_stat}_trend")
    sec_roll3 = _safe(row, f"{secondary_stat}_roll3")

    if math.isnan(roll3):
        return f"Recent performance data for {player_name} is limited, so treat this projection with some caution."

    # Trend direction
    if not math.isnan(trend):
        if trend > 5:
            trend_phrase = "and is trending upward"
        elif trend < -5:
            trend_phrase = "but has been trending downward recently"
        else:
            trend_phrase = "and has been fairly consistent"
    else:
        trend_phrase = ""

    # Secondary stat
    if not math.isnan(sec_roll3) and sec_roll3 > 0:
        sec_phrase = f" on {sec_roll3:.1f} {secondary_label}" if pos == "RB" else f" on {sec_roll3:.1f} {secondary_label}"
    else:
        sec_phrase = ""

    return (
        f"Over the last three games, {player_name} has averaged "
        f"{roll3:.1f} {primary_label}{sec_phrase} {trend_phrase}.".strip()
    )


def _sentence_vegas(player_name: str, team: str, position: str, row: dict) -> str:
    implied = _safe(row, "implied_team_total")
    spread = _safe(row, "spread_line")

    if math.isnan(implied):
        return f"Vegas data isn't available for this game, so game-script factors are unknown."

    if implied > 27:
        game_tone = "a high-scoring game is expected"
        script = "a positive game script"
    elif implied > 23:
        game_tone = "a moderately-scoring game is projected"
        script = "a decent game script"
    else:
        game_tone = "a lower-scoring game is projected"
        script = "a tighter game script"

    # Spread context
    if not math.isnan(spread):
        if spread < -4:
            script_detail = ", and they are favored to win comfortably"
        elif spread > 4:
            script_detail = ", though they are underdogs and may need to play from behind"
        else:
            script_detail = ""
    else:
        script_detail = ""

    return (
        f"{team} is projected to score {implied:.1f} points ({game_tone}), "
        f"suggesting {script} for {player_name}{script_detail}."
    )


def _sentence_usage(player_name: str, position: str, row: dict) -> str | None:
    """Returns a usage sentence only when the data is meaningfully notable."""
    pos = position.upper()
    snap_pct = _safe(row, "offense_pct")

    if pos == "RB":
        rz_stat = _safe(row, "rz_carries_roll3")
        rz_label = "red zone carries"
    else:
        rz_stat = _safe(row, "rz_targets_roll3")
        rz_label = "red zone targets"

    # Priority: low snap rate is the strongest warning, red zone is the best upside signal.
    # Only surface ONE factor — the most impactful one.
    if not math.isnan(snap_pct) and snap_pct < 0.55:
        return (
            f"One thing to watch: {player_name} has been on the field for only "
            f"{snap_pct*100:.0f}% of offensive snaps, which limits the overall ceiling."
        )

    if not math.isnan(rz_stat) and rz_stat >= 2.0:
        return (
            f"{player_name} is averaging {rz_stat:.1f} {rz_label} per game, "
            f"giving a real chance at a touchdown even in a tough week."
        )

    if not math.isnan(snap_pct) and snap_pct >= 0.85:
        return (
            f"{player_name} is playing {snap_pct*100:.0f}% of offensive snaps, "
            f"confirming a locked-in featured role."
        )

    return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_reasoning(
    player_name: str,
    team: str,
    position: str,
    feature_row: dict,
    predicted_pts: float,
    recommendation: str,
) -> str:
    """
    Generate 3–4 plain-English sentences explaining a start/sit prediction.

    Parameters
    ----------
    player_name  : Display name (e.g. "Josh Jacobs")
    team         : Team abbreviation (e.g. "GB")
    position     : QB / RB / WR / TE
    feature_row  : Dict of feature values for this player-week
    predicted_pts: Model's predicted PPR fantasy points
    recommendation: "START" or "SIT"

    Returns
    -------
    str — 3 or 4 sentences joined by spaces.
    """
    sentences = [
        _sentence_matchup(player_name, position, feature_row),
        _sentence_form(player_name, position, feature_row),
        _sentence_vegas(player_name, team, position, feature_row),
    ]

    usage = _sentence_usage(player_name, position, feature_row)
    if usage:
        sentences.append(usage)

    return " ".join(sentences)
