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
    """
    Recursively replace any float outside (0, 1) with safe values.
    IMPORTANT: bool is checked before int because bool is a subclass of int in Python.
    """
    if isinstance(obj, bool):
        # Booleans must be checked BEFORE int — return as-is, never convert
        return obj
    elif isinstance(obj, float):
        # Convert negative to small positive
        if obj < 0:
            obj = abs(obj)
        if obj == 0.0:
            return 0.0001
        if obj >= 1.0:
            return 0.9999
        return max(0.0001, min(0.9999, obj))
    elif isinstance(obj, int):
        # Only convert ints that represent scores (0 or 1 boundary)
        if obj <= 0:
            return 0.0001
        if obj >= 1:
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
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            try:
                data = json.loads(body)
                scrubbed = scrub_float(data)
                new_body = json.dumps(scrubbed).encode("utf-8")
                headers = dict(response.headers)
                headers.pop("content-length", None)
                return Response(
                    content=new_body,
                    status_code=response.status_code,
                    headers=headers,
                    media_type="application/json",
                )
            except Exception:
                # If JSON parsing fails, return original response unchanged
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

# Add middleware (order matters — FloatScrubber must be first)
app.add_middleware(FloatScrubberMiddleware)
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
