"""
app/main.py: FastAPI application for the fantasy football start/sit tool.

Endpoints:
    GET  /               serves the frontend
    GET  /health         liveness check
    GET  /latest-week    returns the most recent season/week in the data
    GET  /players/search autocomplete player search
    POST /compare        head-to-head start/sit comparison with reasoning
"""

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ml.predict import Predictor
from ml.reasoning import generate_reasoning

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROC_DIR   = Path("data/processed")
STATIC_DIR = Path("app/static")

# ---------------------------------------------------------------------------
# App lifespan: load models once at startup so they are ready for every request
# ---------------------------------------------------------------------------

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = Predictor()
    yield


app = FastAPI(title="Start or Sit", version="1.0.0", lifespan=lifespan)

# Serve static files under /static/*
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_features(position: str) -> pd.DataFrame:
    path = PROC_DIR / f"{position.lower()}_features.csv"
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Feature file missing for {position}. Run ml.pipeline first.",
        )
    return pd.read_csv(path)


def _get_feature_row(
    position: str, player_name: str, season: int, week: int
) -> dict | None:
    """Return the feature dict for one player-week, or None if not found."""
    df = _read_features(position)
    mask = (
        (df["season"] == season)
        & (df["week"] == week)
        & df["player_display_name"]
        .str.lower()
        .str.contains(player_name.lower(), regex=False, na=False)
    )
    rows = df[mask]
    return rows.iloc[0].to_dict() if not rows.empty else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/latest-week")
async def latest_week():
    """Return the most recent season and week present in the processed data."""
    df = _read_features("rb")
    max_season = int(df["season"].max())
    max_week   = int(df[df["season"] == max_season]["week"].max())
    return {"season": max_season, "week": max_week}


@app.get("/top-scorers")
async def top_scorers(
    season: int = Query(2024),
    week: int   = Query(..., description="Current week, returns leaders from the previous week"),
    top_n: int  = Query(5, description="Top N per position"),
):
    """Return the top fantasy scorers (PPR) from the previous week, grouped by position."""
    prev_week = week - 1
    if prev_week < 1:
        return {"week": None, "players": []}

    results = []
    for pos in ["QB", "RB", "WR", "TE"]:
        try:
            df = _read_features(pos)
        except HTTPException:
            continue
        mask = (df["season"] == season) & (df["week"] == prev_week)
        subset = (
            df[mask]
            .sort_values("fantasy_points_ppr", ascending=False)
            .head(top_n)[["player_display_name", "team", "position", "fantasy_points_ppr", "headshot_url"]]
        )
        results.extend(subset.to_dict(orient="records"))

    return {"week": prev_week, "season": season, "players": results}


@app.get("/players/search")
async def players_search(
    q: str = Query(..., min_length=2, description="Partial player name"),
    position: str | None = Query(None, description="QB / RB / WR / TE (optional, omit to search all positions)"),
    season: int = Query(2024, description="NFL season year"),
):
    positions = ["QB", "RB", "WR", "TE"]
    if position:
        pos = position.upper()
        if pos not in positions:
            raise HTTPException(status_code=400, detail="Position must be QB, RB, WR, or TE.")
        positions = [pos]

    frames = []
    for pos in positions:
        try:
            df = _read_features(pos)
        except HTTPException:
            continue
        season_df = df[df["season"] == season]
        mask = (
            season_df["player_display_name"]
            .str.lower()
            .str.contains(q.lower(), regex=False, na=False)
        )
        matched = season_df[mask][["player_display_name", "team", "headshot_url"]].copy()
        matched["position"] = pos
        frames.append(matched)

    if not frames:
        return []

    combined = (
        pd.concat(frames)
        .drop_duplicates(subset=["player_display_name"])
        .head(8)
    )
    return combined.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Compare endpoint
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    player1_name: str
    player1_position: str
    player2_name: str
    player2_position: str
    season: int = 2024
    week: int


@app.post("/compare")
async def compare(req: CompareRequest):
    """
    Head-to-head start/sit comparison between any two players.

    Both players are predicted independently using their own position models.
    The player with the higher predicted score is the START recommendation.
    Each player's card includes 3-4 sentences of plain-English reasoning.
    """
    players_input = [
        (req.player1_name, req.player1_position),
        (req.player2_name, req.player2_position),
    ]

    results = []

    for name, pos in players_input:
        pos = pos.upper()

        # Run prediction for this player
        player_list = _predictor.predict(pos, req.season, req.week, players=[name])
        if not player_list:
            raise HTTPException(
                status_code=404,
                detail=f'"{name}" wasn\'t found in our Week {req.week} data. Try using their full name or check that they played that week.',
            )

        player = player_list[0]

        # Fetch feature row for reasoning and headshot
        row = _get_feature_row(pos, name, req.season, req.week)
        headshot = (row.get("headshot_url") or "") if row else ""

        reasoning = (
            generate_reasoning(
                player_name=player["player_name"],
                team=player["team"],
                position=pos,
                feature_row=row,
                predicted_pts=player["predicted_pts"],
                recommendation=player["recommendation"],
            )
            if row
            else "Not enough data to generate reasoning for this player."
        )

        results.append({**player, "headshot_url": headshot, "reasoning": reasoning, "opponent_team": player.get("opponent_team", "")})

    # Head-to-head verdict: higher predicted pts = START
    # Override positional-median START/SIT with head-to-head result
    starter = max(results, key=lambda p: p["predicted_pts"])
    for p in results:
        p["recommendation"] = "BOOM" if p["player_name"] == starter["player_name"] else "BUST"

    verdict = f"Start {starter['player_name']}"

    return {
        "season": req.season,
        "week": req.week,
        "players": results,
        "verdict": verdict,
    }
