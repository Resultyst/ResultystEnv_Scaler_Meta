"""
ResultystEnv — FastAPI Server
Exposes the OpenEnv interface over HTTP.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

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
# Global float scrubber
# ─────────────────────────────────────────────

def scrub_float(obj: Any) -> Any:
    """Recursively replace any float equal to 0.0 or 1.0 with safe values."""
    if isinstance(obj, float):
        if obj == 0.0 or obj == -0.0:
            return 0.0001
        if obj == 1.0:
            return 0.9999
        if obj == -1.0:
            return -0.9999
        # Also clamp to safe range just in case
        if obj > 0:
            return max(0.0001, min(0.9999, obj))
        else:
            return max(-0.9999, min(-0.0001, obj))
    elif isinstance(obj, int):
        if obj == 0:
            return 0.0001
        if obj == 1:
            return 0.9999
        return float(obj)
    elif isinstance(obj, dict):
        return {k: scrub_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [scrub_float(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(scrub_float(v) for v in obj)
    else:
        return obj


class FloatScrubberMiddleware(BaseHTTPMiddleware):
    """Middleware that scrubs all floats in JSON responses."""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if response.headers.get("content-type") == "application/json":
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            try:
                data = json.loads(body)
                scrubbed = scrub_float(data)
                return Response(
                    content=json.dumps(scrubbed),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type="application/json",
                )
            except:
                pass
        return response


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

env_instance = ResultystEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    env_instance.reset("task_easy")
    yield


app = FastAPI(
    title="ResultystEnv",
    description="OpenEnv-compatible HR workflow environment.",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware (order matters)
app.add_middleware(FloatScrubberMiddleware)  # This scrubs all JSON responses
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
    return {"status": "ok", "env": "ResultystEnv", "version": "1.0.0"}


@app.get("/tasks", tags=["system"])
def get_tasks():
    return {"tasks": list_tasks()}


@app.get("/", tags=["system"])
def root():
    return {"status": "ok", "env": "ResultystEnv", "version": "1.0.0"}


@app.post("/reset", response_model=Observation, tags=["openenv"])
def reset(request: Optional[ResetRequest] = None):
    task_id = (request.task_id if request else None) or "task_easy"
    try:
        obs = env_instance.reset(task_id=task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse, tags=["openenv"])
def step(request: StepRequest):
    try:
        obs, reward, done, info = env_instance.step(request.action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/state", tags=["openenv"])
def get_state():
    try:
        s = env_instance.state()
        d = s.model_dump()
        d["checks_completed"] = list(d.get("checks_completed", []))
        return d
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/grade", tags=["openenv"])
def grade_episode():
    try:
        result = env_instance.grade_episode()
        return result  # Middleware will scrub floats
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
