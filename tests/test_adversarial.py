"""
Adversarial tests for boom-or-bust.

These tests are written to break the code, not validate it.
Each test corresponds to a specific failure mode or silent incorrectness.
"""

import re
import math
import pytest
import numpy as np
import pandas as pd

from ml.features import add_rolling_features, add_trend_features, compute_def_rank_allowed
from ml.evaluate import start_sit_accuracy, boom_bust_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_player_df(player_id, season, weeks, yards):
    return pd.DataFrame({
        "player_id": [player_id] * len(weeks),
        "season":    [season] * len(weeks),
        "week":      weeks,
        "rushing_yards": yards,
    })


def make_eval_inputs(actuals, preds, season=2024, week=1):
    meta = pd.DataFrame({
        "player_display_name": [f"P{i}" for i in range(len(actuals))],
        "team":   ["A"] * len(actuals),
        "season": [season] * len(actuals),
        "week":   [week] * len(actuals),
    })
    y_true = pd.Series(actuals)
    y_pred = np.array(preds)
    return y_true, y_pred, meta


# ─────────────────────────────────────────────────────────────────────────────
# ATTACK 1: roll3 in week 2 is not a 3-game average — it is a 1-game average
# ─────────────────────────────────────────────────────────────────────────────

def test_roll3_week2_is_actually_one_game():
    df = make_player_df("P1", 2024, [1, 2, 3, 4, 5], [100.0, 200.0, 150.0, 180.0, 160.0])
    result = add_rolling_features(df, ["rushing_yards"])

    week2 = result[result["week"] == 2]
    roll3 = week2["rushing_yards_roll3"].values[0]

    # min_periods=1 means the "3-game" window uses only 1 prior game.
    # roll3 should be 100.0 (week 1 only), not an average of 3 games.
    assert roll3 == 100.0, (
        f"Expected roll3=100.0 (1-game average) in week 2, got {roll3}. "
        "The column is labeled '3-game average' but min_periods=1 means "
        "week 2 uses only 1 prior game."
    )


def test_roll3_and_roll4_are_identical_in_week_2():
    df = make_player_df("P1", 2024, [1, 2, 3, 4, 5], [100.0, 200.0, 150.0, 180.0, 160.0])
    result = add_rolling_features(df, ["rushing_yards"])

    week2 = result[result["week"] == 2]
    roll3 = week2["rushing_yards_roll3"].values[0]
    roll4 = week2["rushing_yards_roll4"].values[0]

    # Both windows collapse to the same single prior game.
    # A "3-game average" and a "4-game average" that are identical have no
    # information difference — any trend computed from them is meaningless.
    assert roll3 == roll4, (
        f"Week 2 roll3={roll3} != roll4={roll4}. "
        "Both windows should collapse to 1 game (week 1 only) due to min_periods=1."
    )


def test_trend_is_always_zero_in_week_2_regardless_of_player_performance():
    df = make_player_df("P1", 2024, [1, 2, 3, 4, 5], [100.0, 200.0, 150.0, 180.0, 160.0])
    result = add_rolling_features(df, ["rushing_yards"])
    result = add_trend_features(result, ["rushing_yards"])

    week2_trend = result[result["week"] == 2]["rushing_yards_trend"].values[0]

    # roll3 == roll4 in week 2, so trend = 0.
    # Every player in week 2 gets a trend of 0 regardless of their actual trajectory.
    # The model cannot distinguish a hot player from a cold one in week 2.
    assert week2_trend == 0.0, (
        f"Week 2 trend={week2_trend}. "
        "All players have trend=0 in week 2 because roll3=roll4 when min_periods=1."
    )


def test_roll3_week3_is_two_game_average_not_three():
    df = make_player_df("P1", 2024, [1, 2, 3, 4, 5], [100.0, 200.0, 150.0, 180.0, 160.0])
    result = add_rolling_features(df, ["rushing_yards"])

    week3 = result[result["week"] == 3]
    roll3 = week3["rushing_yards_roll3"].values[0]

    expected = (100.0 + 200.0) / 2  # 2-game average, not 3-game
    assert roll3 == expected, (
        f"Week 3 roll3={roll3}, expected 2-game average={expected}. "
        "A 3-game window in week 3 can only contain 2 prior games."
    )


def test_roll3_is_correct_3_game_average_only_from_week_4_onward():
    df = make_player_df("P1", 2024, [1, 2, 3, 4, 5, 6], [100.0, 200.0, 150.0, 180.0, 160.0, 170.0])
    result = add_rolling_features(df, ["rushing_yards"])

    week4 = result[result["week"] == 4]
    roll3 = week4["rushing_yards_roll3"].values[0]

    expected = (100.0 + 200.0 + 150.0) / 3
    assert abs(roll3 - expected) < 0.001, (
        f"Week 4 roll3={roll3}, expected true 3-game average={expected:.3f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ATTACK 2: add_trend_features mutates its input DataFrame in place
# ─────────────────────────────────────────────────────────────────────────────

def test_trend_features_mutates_input_dataframe():
    df = make_player_df("P1", 2024, [1, 2, 3, 4, 5], [100.0, 200.0, 150.0, 180.0, 160.0])
    df = add_rolling_features(df, ["rushing_yards"])

    original_cols = set(df.columns)
    original_id = id(df)

    returned = add_trend_features(df, ["rushing_yards"])

    # The function returns df and also modifies df in place.
    # The caller's reference to df now has extra columns they didn't ask for.
    assert set(df.columns) != original_cols, (
        "add_trend_features modified the input df in place. "
        "A caller who did not reassign df now sees unexpected columns. "
        "The function should copy before mutating."
    )
    # Confirm it's the same object, not a copy
    assert id(returned) == original_id, (
        "add_trend_features returns the same object it mutated, not a copy."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ATTACK 3: player name regex injection in predict.py pattern builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_player_pattern(players):
    """Reproduces predict.py line 161 exactly."""
    return "|".join(p.lower() for p in players)


def test_unbalanced_paren_in_player_name_raises_re_error():
    name = "Ja'Marr (Chase"
    pattern = _build_player_pattern([name])
    with pytest.raises(re.error):
        re.compile(pattern)


def test_dot_in_player_name_over_matches():
    # "D.J. Moore" as a raw regex: '.' matches any character.
    # The pattern "d.j. moore" matches any string of the form d[x]j[y] moore
    # where [x] and [y] are ANY character — including "dfj5 moore".
    name = "D.J. Moore"
    pattern = _build_player_pattern([name])
    compiled = re.compile(pattern)

    # "dfj5 moore" should NOT match D.J. Moore, but it does
    assert compiled.search("dfj5 moore") is not None, (
        "D.J. Moore's name as a raw regex matches 'dfj5 moore' "
        "because '.' is not escaped. A typo or different player could be returned."
    )


def test_metachar_in_player_name_breaks_regex():
    # A name containing an unbalanced parenthesis is a broken regex.
    # This is a real name format: some data sources include nicknames in parens.
    broken_name = "Player (nickname"
    pattern = _build_player_pattern([broken_name])
    with pytest.raises(re.error):
        re.compile(pattern)


# ─────────────────────────────────────────────────────────────────────────────
# GAP ANALYSIS: start_sit_accuracy returns 0.0 for ambiguous inputs
# ─────────────────────────────────────────────────────────────────────────────

def test_start_sit_accuracy_returns_zero_when_all_actuals_are_tied():
    # Every player scores the same — all pairs skipped, total=0, returns 0.0.
    # 0.0 is indistinguishable from "model got everything wrong."
    # Should return nan or raise, not a number that looks like a real metric.
    y_true, y_pred, meta = make_eval_inputs(
        actuals=[10.0, 10.0, 10.0, 10.0],
        preds=[15.0, 12.0, 9.0, 7.0],
    )
    result = start_sit_accuracy(y_true, y_pred, meta)
    assert result == 0.0, "Returns 0.0 when all actuals are tied (total=0)"
    # Flag: caller cannot distinguish "0% accuracy" from "no evaluable pairs"


def test_start_sit_accuracy_with_single_player_returns_zero():
    y_true, y_pred, meta = make_eval_inputs(actuals=[20.0], preds=[18.0])
    result = start_sit_accuracy(y_true, y_pred, meta)
    assert result == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GAP ANALYSIS: boom/bust labels can overlap in a tiny player pool
# ─────────────────────────────────────────────────────────────────────────────

def test_boom_and_bust_labels_overlap_in_small_pool():
    # With 2 players, p75 and p25 both resolve to values within the 2-player range.
    # np.percentile with 2 values:
    #   p75 of [10, 20] = 17.5 (linear interpolation)
    #   p25 of [10, 20] = 12.5
    # So is_boom = actual >= 17.5, is_bust = actual <= 12.5
    # Player scoring 10: is_bust=True, is_boom=False
    # Player scoring 20: is_bust=False, is_boom=True
    # No overlap here. But what about 3 players where middle player is at both thresholds?
    # p75 of [10, 15, 20] = 17.5, p25 = 12.5
    # Player scoring 15: neither boom nor bust — this is fine.

    # The real overlap: when pool is 1 player (one team), both thresholds equal the score.
    # is_boom = score >= score = True, is_bust = score <= score = True.
    # A single player is simultaneously flagged as BOOM and BUST.
    y_true, y_pred, meta = make_eval_inputs(actuals=[15.0], preds=[14.0])
    result = boom_bust_metrics(y_true, y_pred, meta)

    # With 1 player: p75 = p25 = 15.0
    # is_boom = 15.0 >= 15.0 = True
    # is_bust = 15.0 <= 15.0 = True
    # The single player is counted as both BOOM and BUST simultaneously
    assert isinstance(result, dict)
    # Document the behavior: with a pool of 1, the same player contributes to
    # both boom and bust precision/recall which makes neither metric meaningful.


def test_boom_bust_with_two_players_in_week():
    # Boundary: exactly 2 players. np.percentile behavior with 2 values.
    y_true, y_pred, meta = make_eval_inputs(
        actuals=[5.0, 30.0],
        preds=[4.0, 28.0],
    )
    result = boom_bust_metrics(y_true, y_pred, meta)
    # Model correctly predicts ordering, both boom and bust precision should be high
    assert result["boom_precision"] > 0, "Model correctly flagged the boom player"
    assert result["bust_precision"] > 0, "Model correctly flagged the bust player"


# ─────────────────────────────────────────────────────────────────────────────
# GAP FILL: cross-season rolling bleed
# ─────────────────────────────────────────────────────────────────────────────

def test_rolling_does_not_bleed_across_seasons():
    # A player with a great 2023 season should NOT carry those stats
    # into week 2 of 2024 via the rolling window.
    df = pd.DataFrame({
        "player_id": ["P1"] * 6,
        "season":    [2023, 2023, 2023, 2024, 2024, 2024],
        "week":      [13, 14, 17, 2, 3, 4],
        "rushing_yards": [200.0, 180.0, 175.0, 5.0, 8.0, 10.0],
    })
    result = add_rolling_features(df, ["rushing_yards"])

    # Week 2 of 2024 should be NaN or based only on 2024 data (no prior 2024 games).
    # shift(1) on a new season group starts with NaN, so roll2_2024_w2 should be NaN.
    s2024_w2 = result[(result["season"] == 2024) & (result["week"] == 2)]
    roll3_val = s2024_w2["rushing_yards_roll3"].values[0]

    assert math.isnan(roll3_val), (
        f"Week 2 of 2024 roll3={roll3_val}. "
        "Expected NaN because there are no prior 2024 games. "
        "The rolling window should not bleed across season groups."
    )


# ─────────────────────────────────────────────────────────────────────────────
# GAP FILL: def_rank week 1 is NaN, confirm it doesn't bleed into training rows
# ─────────────────────────────────────────────────────────────────────────────

def test_def_rank_week1_is_nan_for_all_defenses():
    weekly = pd.DataFrame({
        "season":            [2024] * 6,
        "week":              [1, 1, 1, 2, 2, 2],
        "opponent_team":     ["NYG", "DAL", "PHI"] * 2,
        "position":          ["RB"] * 6,
        "fantasy_points_ppr": [20.0, 15.0, 10.0, 18.0, 12.0, 8.0],
    })
    result = compute_def_rank_allowed(weekly)

    week1 = result[result["week"] == 1]
    # shift(1) makes the first row in each group NaN.
    # Ranking NaN values returns NaN rank. Week 1 def_rank should be all NaN.
    assert week1["def_rank"].isna().all(), (
        "Week 1 def_rank should be NaN for all defenses. "
        "If any week-1 rows survive into training data, they carry a NaN "
        "def_rank that XGBoost handles differently from inference (fill=0.0)."
    )


def test_def_rank_week2_has_valid_ranks():
    weekly = pd.DataFrame({
        "season":            [2024] * 6,
        "week":              [1, 1, 1, 2, 2, 2],
        "opponent_team":     ["NYG", "DAL", "PHI"] * 2,
        "position":          ["RB"] * 6,
        "fantasy_points_ppr": [20.0, 15.0, 10.0, 18.0, 12.0, 8.0],
    })
    result = compute_def_rank_allowed(weekly)

    week2 = result[result["week"] == 2]
    assert not week2["def_rank"].isna().any(), "Week 2 def_rank should have valid ranks"
    # NYG allowed 20pts to RBs in week 1, so should be rank 1 (easiest matchup)
    nyg_rank = week2[week2["def_team"] == "NYG"]["def_rank"].values[0]
    assert nyg_rank == 1, f"NYG allowed most points, should be def_rank=1, got {nyg_rank}"


def test_def_rank_ascending_false_means_rank1_is_most_points_allowed():
    # Confirm the rank direction. def_rank=1 should mean EASIEST matchup.
    weekly = pd.DataFrame({
        "season":            [2024] * 4,
        "week":              [1, 1, 2, 2],
        "opponent_team":     ["EASY", "TOUGH", "EASY", "TOUGH"],
        "position":          ["WR"] * 4,
        "fantasy_points_ppr": [40.0, 5.0, 35.0, 6.0],
    })
    result = compute_def_rank_allowed(weekly)

    week2 = result[result["week"] == 2]
    easy_rank = week2[week2["def_team"] == "EASY"]["def_rank"].values[0]
    tough_rank = week2[week2["def_team"] == "TOUGH"]["def_rank"].values[0]

    assert easy_rank < tough_rank, (
        f"EASY defense (rank={easy_rank}) should rank lower number than "
        f"TOUGH defense (rank={tough_rank}). Rank 1 = easiest matchup."
    )
