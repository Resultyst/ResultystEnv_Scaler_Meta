#!/usr/bin/env python3
"""
ResultystEnv — Baseline Inference (Fixed LLM Connectivity)
"""

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

import requests
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

# ------------------------------------------------------------------
# Environment Variables (injected by validator)
# ------------------------------------------------------------------
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

BENCHMARK = "ResultystEnv"
TASKS = ["task_easy", "task_medium", "task_hard"]

MAX_STEPS = 15
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

# ------------------------------------------------------------------
# Logging (mandatory format)
# ------------------------------------------------------------------
def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(*, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={done} error={error}", flush=True)

def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)

# ------------------------------------------------------------------
# Environment HTTP Client
# ------------------------------------------------------------------
class EnvClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def reset(self, task_id: str) -> dict:
        r = requests.post(f"{self.base}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, parameters: dict) -> dict:
        payload = {"action": {"action_type": action_type, "parameters": parameters}}
        r = requests.post(f"{self.base}/step", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def grade(self) -> dict:
        r = requests.get(f"{self.base}/grade", timeout=30)
        r.raise_for_status()
        return r.json()

# ------------------------------------------------------------------
# LLM Client (with connection test and robust error handling)
# ------------------------------------------------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
    timeout=30.0,
    max_retries=0,  # we handle retries ourselves
)

def test_llm_connection() -> bool:
    """Quick test to verify the LLM proxy is reachable and authenticated."""
    try:
        # Minimal request – just list models or a simple completion
        _ = client.models.list()
        print(f"[DEBUG] LLM connection test successful to {API_BASE_URL}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[ERROR] LLM connection test failed: {type(e).__name__}: {e}", file=sys.stderr)
        return False

SYSTEM_PROMPT = """You are an HR compliance agent. You must output ONLY valid JSON.

Available actions:
- check_domain
- analyze_email
- verify_company
- mark_safe
- reject_job
- propose_time (parameters: {"slot": "ISO datetime"})
- reschedule (parameters: {"reason": "..."})
- finalize_schedule (parameters: {"slot": "ISO datetime"})

Respond with: {"action_type": "...", "parameters": {...}}"""

def build_prompt(obs: dict, history: List[str]) -> str:
    stage = obs.get("stage", "verification")
    job = obs.get("job_post", {})
    ci = obs.get("company_info", {})
    cal = obs.get("calendars", {})
    available = obs.get("available_actions", [])

    lines = [
        f"Stage: {stage}",
        f"Message: {obs.get('message', '')}",
        f"Job: {job.get('title')} at {job.get('company')}",
        f"Email: {job.get('email')}",
        f"Domain: {job.get('domain')}",
    ]
    if stage == "scheduling":
        lines.append(f"Candidate slots: {cal.get('candidate_slots')}")
        lines.append(f"Interviewer slots: {cal.get('interviewer_slots')}")
        lines.append(f"Valid overlap windows: {cal.get('timezone_overlap_windows')}")
    lines.append(f"Available actions: {available}")
    lines.append("Choose the next action as JSON.")
    return "\n".join(lines)

def parse_action(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None

def get_action(obs: dict, history: List[str]) -> tuple[str, dict]:
    prompt = build_prompt(obs, history)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    print(f"[DEBUG] Calling LLM with model {MODEL_NAME}", file=sys.stderr)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=150,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content
        print(f"[DEBUG] LLM response: {raw}", file=sys.stderr)
        parsed = parse_action(raw)
        if not parsed or "action_type" not in parsed:
            available = obs.get("available_actions", ["check_domain"])
            print(f"[WARN] Invalid LLM response, falling back to {available[0]}", file=sys.stderr)
            return available[0], {}
        return parsed["action_type"], parsed.get("parameters", {})
    except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
        # Log the error clearly – validator will see this
        print(f"[LLM ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        # Fallback to a safe action
        available = obs.get("available_actions", ["check_domain"])
        return available[0], {}
    except Exception as e:
        print(f"[LLM UNEXPECTED ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        available = obs.get("available_actions", ["check_domain"])
        return available[0], {}

# ------------------------------------------------------------------
# Episode Runner
# ------------------------------------------------------------------
def run_episode(task_id: str) -> dict:
    env = EnvClient(ENV_BASE_URL)
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    history = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id)
        for step in range(1, MAX_STEPS + 1):
            if obs.get("done"):
                break

            action_type, params = get_action(obs, history)
            try:
                step_resp = env.step(action_type, params)
            except Exception as e:
                log_step(step=step, action=action_type, reward=0.0, done=False, error=str(e))
                break

            obs = step_resp.get("observation", {})
            reward = step_resp.get("reward", {}).get("value", 0.0)
            done = step_resp.get("done", False)

            rewards.append(reward)
            steps_taken = step
            history.append(f"{action_type}: reward={reward:.3f}")

            log_step(step=step, action=action_type, reward=reward, done=done, error=None)

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)

        try:
            grade = env.grade()
            grader_score = grade.get("score", score)
        except:
            grader_score = score

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", file=sys.stderr)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_id,
        "score": round(score, 4),
        "grader_score": round(grader_score, 4),
        "steps": steps_taken,
        "success": success,
    }

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(f"[DEBUG] API_BASE_URL = {API_BASE_URL}", file=sys.stderr)
    print(f"[DEBUG] API_KEY = {API_KEY[:8]}...", file=sys.stderr)
    print(f"[DEBUG] MODEL_NAME = {MODEL_NAME}", file=sys.stderr)

    # Test LLM connectivity before starting episodes
    if not test_llm_connection():
        print("[FATAL] Cannot reach LLM proxy – aborting.", file=sys.stderr)
        sys.exit(1)

    results = []
    for task in TASKS:
        print(f"\n--- Running {task} ---", flush=True)
        res = run_episode(task)
        results.append(res)
        time.sleep(0.5)

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['task']}: score={r['score']}, grader={r['grader_score']}, steps={r['steps']}")

if __name__ == "__main__":
    main()
