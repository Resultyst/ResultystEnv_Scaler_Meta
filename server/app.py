"""
ResultystEnv — FastAPI Server
Exposes the OpenEnv interface over HTTP.

Endpoints:
  POST /reset      → Observation
  POST /step       → StepResponse (observation, reward, done, info)
  GET  /state      → EnvState
  GET  /health     → {"status": "ok"}
  GET  /tasks      → list of available tasks
  GET  /grade      → GradeResult for current episode
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .env import ResultystEnv
from .models import (
    Action,
    EnvState,
    Observation,
    Reward,
    ResetRequest,
    StepRequest,
    StepResponse,
)
from .tasks import list_tasks


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

env_instance = ResultystEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up with default task on startup
    env_instance.reset("task_easy")
    yield


app = FastAPI(
    title="ResultystEnv",
    description=(
        "OpenEnv-compatible environment simulating two-stage HR workflows: "
        "job scam detection + interview scheduling. "
        "Built for the OpenEnv Hackathon by Team Resultyst."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health_check():
    """Liveness probe — must return 200 for HF Space auto-ping."""
    return {"status": "ok", "env": "ResultystEnv", "version": "1.0.0"}


@app.get("/tasks", tags=["system"])
def get_tasks():
    """List all available tasks with metadata."""
    return {"tasks": list_tasks()}


@app.get("/", tags=["system"])
def root():
    """Root health probe — HF Space auto-ping and openenv validate land here."""
    return {"status": "ok", "env": "ResultystEnv", "version": "1.0.0"}


@app.post("/reset", response_model=Observation, tags=["openenv"])
def reset(request: Optional[ResetRequest] = None):
    """
    Reset the environment and start a new episode.
    Accepts empty body {} — defaults to task_easy.
    Returns the initial Observation.
    """
    task_id = (request.task_id if request else None) or "task_easy"
    try:
        obs = env_instance.reset(task_id=task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse, tags=["openenv"])
def step(request: StepRequest):
    """
    Execute one action and return the next observation, reward, done flag, and info.
    """
    try:
        obs, reward, done, info = env_instance.step(request.action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/state", tags=["openenv"])
def get_state():
    """
    Return the full internal environment state.
    Used by openenv validate and for debugging.
    Returns EnvState as dict (set is JSON-serialized as list).
    """
    try:
        s = env_instance.state()
        d = s.model_dump()
        # Convert set to list for JSON serialisation
        d["checks_completed"] = list(d.get("checks_completed", []))
        return d
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/grade", tags=["openenv"])
def grade_episode():
    """
    Run the deterministic grader on the current (or just-finished) episode.
    Returns interpretable scoring breakdown.
    """
    try:
        return env_instance.grade_episode()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
