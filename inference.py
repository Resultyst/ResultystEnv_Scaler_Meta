#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any, List, Optional

import httpx
from openai import AsyncOpenAI


# ─────────────────────────────────────────────
# Config (STRICT — do not modify externally)
# ─────────────────────────────────────────────

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]

# ✅ MUST use supported model for proxy tracking
MODEL_NAME = "gpt-4o-mini"

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "ResultystEnv"
TASKS = ["task_easy", "task_medium", "task_hard"]

MAX_STEPS = 15
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5


# ─────────────────────────────────────────────
# Logging (REQUIRED)
# ─────────────────────────────────────────────

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done} error={error}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}",
        flush=True,
    )


# ─────────────────────────────────────────────
# Env Client
# ─────────────────────────────────────────────

class StepResult:
    def __init__(self, data: dict) -> None:
        self.observation = ObsWrapper(data.get("observation", {}))
        self.reward = (data.get("reward") or {}).get("value", 0.0)
        self.done = data.get("done", False)


class ResetResult:
    def __init__(self, data: dict) -> None:
        self.observation = ObsWrapper(data)
        self.done = data.get("done", False)


class ObsWrapper:
    def __init__(self, data: dict) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        return self._data.get(name)

    def as_dict(self) -> dict:
        return self._data


class ResultystEnvClient:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self):
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def reset(self, task_id: str):
        c = await self._get_client()
        r = await c.post(f"{self._base}/reset", json={"task_id": task_id})
        r.raise_for_status()
        return ResetResult(r.json())

    async def step(self, action_type: str, parameters: dict):
        c = await self._get_client()
        payload = {"action": {"action_type": action_type, "parameters": parameters}}
        r = await c.post(f"{self._base}/step", json=payload)
        r.raise_for_status()
        return StepResult(r.json())

    async def grade(self):
        c = await self._get_client()
        r = await c.get(f"{self._base}/grade")
        r.raise_for_status()
        return r.json()

    async def close(self):
        if self._client:
            await self._client.aclose()


# ─────────────────────────────────────────────
# LLM Client (CRITICAL FIXED)
# ─────────────────────────────────────────────

llm = AsyncOpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

SYSTEM_PROMPT = """You are an expert HR compliance agent.

Rules:
- Always investigate before deciding
- Detect typosquats carefully
- Only output JSON

Format:
{"action_type": "...", "parameters": {}}
"""


def _parse_action(text: str):
    try:
        return json.loads(re.search(r"\{.*\}", text, re.DOTALL).group())
    except:
        return None


async def get_model_action(messages):
    print(f"[DEBUG] Calling LLM via {API_BASE_URL}", flush=True)

    # ❌ NO silent fallback allowed
    resp = await llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=100,
        temperature=0.0,
    )

    raw = resp.choices[0].message.content
    parsed = _parse_action(raw)

    if not parsed:
        raise RuntimeError("LLM returned invalid JSON")

    return parsed["action_type"], parsed.get("parameters", {})


# ─────────────────────────────────────────────
# Episode Runner
# ─────────────────────────────────────────────

async def run_episode(task_id: str):
    env = ResultystEnvClient(ENV_BASE_URL)

    rewards = []
    steps_taken = 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        result = await env.reset(task_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            messages.append({
                "role": "user",
                "content": json.dumps(result.observation.as_dict())
            })

            action_type, params = await get_model_action(messages)

            result = await env.step(action_type, params)

            reward = result.reward
            rewards.append(reward)

            log_step(step=step, action=action_type, reward=reward, done=result.done, error=None)

            steps_taken = step

            if result.done:
                break

        score = min(max(sum(rewards), 0.0), 1.0)

        grade = await env.grade()
        grader_score = grade.get("score", score)

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_id,
        "score": score,
        "grader_score": grader_score,
        "steps": steps_taken,
        "success": success,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

async def main():
    print("Running ResultystEnv baseline...\n", flush=True)

    results = []

    for task in TASKS:
        res = await run_episode(task)
        results.append(res)
        await asyncio.sleep(0.5)

    print("\nSUMMARY\n")
    for r in results:
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
