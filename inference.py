#!/usr/bin/env python3
"""
ResultystEnv — Baseline Inference Script
Matches the official OpenEnv hackathon async pattern exactly.
"""

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
# Config — USE INJECTED ENV VARS
# ─────────────────────────────────────────────

API_BASE_URL = os.environ["API_BASE_URL"]               # Required by validator
API_KEY = os.environ["API_KEY"]                         # Required by validator
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "ResultystEnv"
TASKS = ["task_easy", "task_medium", "task_hard"]

MAX_STEPS = 15
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

# ─────────────────────────────────────────────
# Mandatory structured logging
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
# Async Environment Client
# ─────────────────────────────────────────────

class StepResult:
    def __init__(self, data: dict) -> None:
        self._data = data
        self.observation = ObsWrapper(data.get("observation", {}))
        self.reward: float = (data.get("reward") or {}).get("value", 0.0)
        self.done: bool = data.get("done", False)
        self.info: dict = data.get("info", {})


class ResetResult:
    def __init__(self, data: dict) -> None:
        self._data = data
        self.observation = ObsWrapper(data)
        self.done: bool = data.get("done", False)


class ObsWrapper:
    def __init__(self, data: dict) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    def as_dict(self) -> dict:
        return self._data


class ResultystEnvClient:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def reset(self, task_id: str = "task_easy") -> ResetResult:
        c = await self._get_client()
        r = await c.post(f"{self._base}/reset", json={"task_id": task_id})
        r.raise_for_status()
        return ResetResult(r.json())

    async def step(self, action_type: str, parameters: Optional[dict] = None) -> StepResult:
        c = await self._get_client()
        payload = {"action": {"action_type": action_type, "parameters": parameters or {}}}
        r = await c.post(f"{self._base}/step", json=payload)
        r.raise_for_status()
        return StepResult(r.json())

    async def grade(self) -> dict:
        c = await self._get_client()
        r = await c.get(f"{self._base}/grade")
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ─────────────────────────────────────────────
# LLM Client — MUST USE INJECTED CREDENTIALS
# ─────────────────────────────────────────────

llm = AsyncOpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

SYSTEM_PROMPT = """You are an expert HR compliance agent managing job verification and interview scheduling.

STAGES:
  verification → investigate before deciding: check_domain, analyze_email, verify_company, mark_safe, reject_job
  scheduling   → coordinate calendars: propose_time, reschedule, finalize_schedule

CRITICAL RULES:
1. Always run at least 2 investigation actions BEFORE any verdict.
2. Watch for typosquatting: e.g. 'micr0soft.com' uses zero not letter-o.
3. For scheduling, ONLY propose slots that appear in BOTH candidate and interviewer lists.
4. Consider timezone math carefully — IST is UTC+5:30, PST is UTC-8.

Respond with ONLY a JSON object:
  {"action_type": "check_domain", "parameters": {}}
  {"action_type": "propose_time",  "parameters": {"slot": "2025-06-15T06:30:00+05:30"}}
  {"action_type": "finalize_schedule", "parameters": {"slot": "2025-06-15T06:30:00+05:30"}}
  {"action_type": "reschedule",    "parameters": {"reason": "slot conflicts"}}
No markdown, no explanation — only the JSON object."""


def _build_user_message(obs: ObsWrapper, step_num: int, history: List[str]) -> str:
    d = obs.as_dict()
    stage     = d.get("stage", "verification")
    msg       = d.get("message", "")
    job       = d.get("job_post", {})
    ci        = d.get("company_info", {})
    cal       = d.get("calendars", {})
    available = d.get("available_actions", [])

    lines = [
        f"=== Step {step_num} | Stage: {stage.upper()} ===",
        f"Env message: {msg}",
        "",
        "JOB POST:",
        f"  Title:   {job.get('title')}",
        f"  Company: {job.get('company')}",
        f"  Email:   {job.get('email')}   ← check carefully for character substitutions",
        f"  Domain:  {job.get('domain')}",
        f"  Salary:  {job.get('salary')}",
        "",
        "COMPANY INFO (revealed so far):",
        f"  Domain age (days):       {ci.get('domain_age_days', 'NOT CHECKED')}",
        f"  HTTPS:                   {ci.get('has_https', 'NOT CHECKED')}",
        f"  Company registered:      {ci.get('company_registered', 'NOT CHECKED')}",
        f"  Email-domain match:      {ci.get('email_domain_match', 'NOT CHECKED')}",
        f"  Typosquat detected:      {ci.get('typosquat_detected', 'NOT CHECKED')}",
        f"  Glassdoor rating:        {ci.get('glassdoor_reviews', 'NOT CHECKED')}",
        f"  LinkedIn present:        {ci.get('linkedin_present', 'NOT CHECKED')}",
    ]

    if stage == "scheduling":
        cand = [s["slot"] for s in cal.get("candidate_slots", []) if s.get("available")]
        intr = [s["slot"] for s in cal.get("interviewer_slots", []) if s.get("available")]
        lines += [
            "",
            "CALENDARS:",
            f"  Candidate available:  {cand}",
            f"  Interviewer available: {intr}",
            f"  Proposed slot so far: {cal.get('proposed_slot', 'none')}",
            "  NOTE: Find a slot present in BOTH lists. Account for timezone offsets.",
        ]

    if history:
        lines += ["", "RECENT HISTORY (last 3 steps):"]
        for h in history[-3:]:
            lines.append(f"  {h}")

    lines += [
        "",
        f"Available actions: {available}",
        "Respond ONLY with: {\"action_type\": \"...\", \"parameters\": {}}",
    ]
    return "\n".join(lines)


def _parse_action(text: str) -> Optional[dict]:
    text = text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


async def get_model_action(
    messages: list,
    obs: ObsWrapper,
    step_num: int,
    history: List[str],
) -> tuple[str, dict]:
    user_msg = _build_user_message(obs, step_num, history)
    messages.append({"role": "user", "content": user_msg})

    try:
        resp = await llm.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=150,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", file=sys.stderr, flush=True)
        raw = '{"action_type": "check_domain", "parameters": {}}'

    messages.append({"role": "assistant", "content": raw})

    parsed = _parse_action(raw)
    if not parsed or "action_type" not in parsed:
        available = obs.available_actions or ["check_domain"]
        return available[0], {}

    return parsed["action_type"], parsed.get("parameters", {}) or {}


# ─────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────

async def run_episode(task_id: str, episode: int = 1) -> dict:
    env = ResultystEnvClient(ENV_BASE_URL)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    history: List[str] = []

    try:
        result = await env.reset(task_id)
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_type, parameters = await get_model_action(
                messages, result.observation, step, history
            )

            error = None
            try:
                result = await env.step(action_type, parameters)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
            except Exception as e:
                error = str(e)
                reward = 0.0
                done = False
                print(f"[DEBUG] env.step error: {e}", file=sys.stderr, flush=True)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            action_str = action_type if not parameters else f"{action_type}({parameters})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str!r} → reward {reward:+.4f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)

        try:
            grade_result = await env.grade()
            grader_score = grade_result.get("score", score)
        except Exception:
            grader_score = score

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":      task_id,
        "episode":      episode,
        "score":        round(score, 4),
        "grader_score": round(grader_score, 4),
        "steps":        steps_taken,
        "rewards":      rewards,
        "success":      success,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

async def main() -> None:
    # Validate required env vars
    missing = []
    for var in ["API_BASE_URL", "API_KEY"]:
        if var not in os.environ:
            missing.append(var)
    if missing:
        print(f"ERROR: Missing required environment variables: {missing}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60, flush=True)
    print(f"ResultystEnv — Baseline Inference", flush=True)
    print(f"Model   : {MODEL_NAME}", flush=True)
    print(f"API     : {API_BASE_URL}", flush=True)
    print(f"Env URL : {ENV_BASE_URL}", flush=True)
    print("=" * 60, flush=True)

    results = []
    for task_id in TASKS:
        print(f"\n{'─'*60}", flush=True)
        result = await run_episode(task_id, episode=1)
        results.append(result)
        await asyncio.sleep(0.5)

    print(f"\n{'='*60}", flush=True)
    print("BASELINE RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Task':<20} {'Score':>8} {'Grader':>8} {'Steps':>6} {'Success':>8}", flush=True)
    print(f"{'─'*56}", flush=True)
    for r in results:
        print(
            f"{r['task_id']:<20} {r['score']:>8.4f} {r['grader_score']:>8.4f} "
            f"{r['steps']:>6} {str(r['success']):>8}",
            flush=True,
        )
    avg = sum(r["score"] for r in results) / len(results)
    print(f"{'─'*56}", flush=True)
    print(f"{'AVERAGE':<20} {avg:>8.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
