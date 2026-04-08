"""
ResultystEnv — Grader
Deterministic, interpretable scoring for all 3 tasks.
Breakdown: signal_detection_score + decision_correctness + overconfidence_penalty
All graders return float in (0, 1).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GradeResult:
    """Interpretable grading output per task."""
    score: float                            # Final 0.01–0.99
    signal_detection_score: float           # Did agent gather enough evidence?
    decision_correctness: float             # Was the scam/safe verdict right?
    overconfidence_penalty: float           # Penalty for rushing to verdict
    scheduling_score: float                 # Only for scheduling tasks
    efficiency_score: float                 # Solved in fewer steps than max?
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "score": _clamp_score(self.score),
            "signal_detection_score": max(0.01, min(0.99, self.signal_detection_score)),
            "decision_correctness": max(0.01, min(0.99, self.decision_correctness)),
            "overconfidence_penalty": self.overconfidence_penalty,  # can be negative
            "scheduling_score": max(0.01, min(0.99, self.scheduling_score)),
            "efficiency_score": max(0.01, min(0.99, self.efficiency_score)),
            "details": self.details,
        }


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

VERIFICATION_CHECKS = {"check_domain", "analyze_email", "verify_company"}


def _clamp_score(score: float) -> float:
    """Ensures score is strictly between 0 and 1, as required by Phase 2 validator."""
    # Engineering safe ranges: 0.01 floor, 0.99 ceiling
    return round(max(0.01, min(0.99, float(score))), 4)


def _signal_detection_score(checks_run: set[str], required_checks: int) -> float:
    """
    How well did the agent gather evidence before deciding?
    - Full score if all 3 checks were run
    - Partial credit based on fraction of checks completed
    - Penalizes skipping important investigation steps
    """
    n_run = len(checks_run & VERIFICATION_CHECKS)
    n_total = len(VERIFICATION_CHECKS)  # always 3
    fraction = n_run / n_total

    if n_run >= required_checks:
        # Internal ceiling: 0.95 instead of 1.0
        return min(0.95, fraction + 0.1)
    return max(0.05, fraction)


def _overconfidence_penalty(checks_run: set[str], required_checks: int) -> float:
    """
    Penalty applied when an agent makes a verdict with insufficient investigation.
    Returns a negative float in [-0.35, 0.0].
    """
    n_run = len(checks_run & VERIFICATION_CHECKS)
    if n_run >= required_checks:
        return 0.0
    # Severity scales with how many checks were skipped
    missing = required_checks - n_run
    return -0.15 * missing  # -0.15 per missing required check


def _efficiency_score(steps_used: int, max_steps: int) -> float:
    """
    Rewards finishing faster (but not at the expense of correctness).
    Linear: 1.0 at min steps, 0.0 at max_steps.
    """
    if steps_used <= 0:
        return 0.05
    min_expected = 3  # At minimum: 1 check + decide + (optionally more)
    score = (max_steps - steps_used) / max(1, max_steps - min_expected)
    # Clamp internal score to safe range
    return max(0.05, min(0.95, score))


def _scheduling_quality(
    finalized_slot: Optional[str],
    valid_slots: list[str],
    proposed_slot: Optional[str],
    steps_in_scheduling: int,
) -> float:
    """
    Multi-factor scheduling score:
      overlap_score  × 0.4  — did the agent pick a valid overlap slot?
      comfort_score  × 0.3  — was it within business hours (08:00–18:00 local)?
      efficiency     × 0.3  — fewer reschedule attempts = better
    """
    if not finalized_slot:
        # Partial credit if a valid slot was proposed but not finalized
        if proposed_slot and proposed_slot in valid_slots:
            return 0.25
        return 0.05

    # Overlap score
    overlap = 0.95 if finalized_slot in valid_slots else 0.05

    # Comfort score: check that actual hour is in business range
    comfort = _compute_comfort_score(finalized_slot)

    # Scheduling efficiency: 0.95 for direct finalize, penalize rescheduling
    eff = max(0.05, 0.95 - (steps_in_scheduling - 1) * 0.15)

    total = overlap * 0.4 + comfort * 0.3 + eff * 0.3
    return max(0.05, min(0.95, total))


def _compute_comfort_score(slot_iso: str) -> float:
    """
    Returns 0.95 if slot is 08:00–18:00 local, 0.5 if 06:00–08:00 or 18:00–20:00,
    0.05 if outside those ranges.
    """
    try:
        from datetime import datetime, timezone
        import re
        # Parse offset-aware ISO datetime  e.g. 2025-06-15T06:30:00+05:30
        # Python's fromisoformat handles this in 3.11+
        dt = datetime.fromisoformat(slot_iso)
        hour = dt.hour + dt.minute / 60
        if 8.0 <= hour < 18.0:
            return 0.95
        if 6.0 <= hour < 8.0 or 18.0 <= hour < 20.0:
            return 0.5
        return 0.05
    except Exception:
        return 0.5  # Unknown format — neutral


# ─────────────────────────────────────────────
# Per-task Graders
# ─────────────────────────────────────────────

def grade_task_easy(
    verdict: Optional[str],
    checks_run: set[str],
    steps_used: int,
    **_,
) -> GradeResult:
    """
    Task Easy: Detect an obvious scam.
    Ground truth: scam
    Correct action: reject_job after ≥2 checks.

    Weights:
      decision_correctness  50%
      signal_detection      30%
      efficiency            20%
    """
    required_checks = 2
    ground_truth = "scam"

    sig = _signal_detection_score(checks_run, required_checks)
    overconf = _overconfidence_penalty(checks_run, required_checks)
    eff = _efficiency_score(steps_used, max_steps=8)

    if verdict == ground_truth:
        dec = min(0.95 + overconf, 0.99)   # Clamped ceiling
    elif verdict is None:
        dec = 0.05                          # Floor
    else:
        dec = -0.4                          # Wrong verdict

    raw = (
        max(0.05, min(0.99, dec)) * 0.50 +
        sig * 0.30 +
        eff * 0.20
    )
    raw = max(0.02, min(0.98, raw))
    score = _clamp_score(raw)

    return GradeResult(
        score=score,
        signal_detection_score=sig,
        decision_correctness=max(0.05, min(0.99, dec)),
        overconfidence_penalty=overconf,
        scheduling_score=0.05,
        efficiency_score=eff,
        details={
            "ground_truth": ground_truth,
            "agent_verdict": verdict,
            "checks_run": list(checks_run),
            "steps_used": steps_used,
            "verdict_correct": verdict == ground_truth,
        },
    )


def grade_task_medium(
    verdict: Optional[str],
    checks_run: set[str],
    steps_used: int,
    finalized_slot: Optional[str],
    proposed_slot: Optional[str],
    valid_slots: list[str],
    steps_in_scheduling: int = 0,
    **_,
) -> GradeResult:
    """
    Task Medium: Borderline job — right answer is 'safe' + schedule correctly.
    Ground truth: safe → must schedule

    Weights:
      decision_correctness  35%
      signal_detection      25%
      scheduling_score      30%
      efficiency            10%
    """
    required_checks = 3   # All 3 required for mixed signals
    ground_truth = "safe"

    sig = _signal_detection_score(checks_run, required_checks)
    overconf = _overconfidence_penalty(checks_run, required_checks)
    eff = _efficiency_score(steps_used, max_steps=12)
    sched = _scheduling_quality(finalized_slot, valid_slots, proposed_slot, steps_in_scheduling)

    if verdict == ground_truth:
        dec = min(0.95 + overconf, 0.99)   # Clamped ceiling
    elif verdict is None:
        dec = 0.05                          # Floor
    else:
        dec = -0.5                          # Rejected a legit job

    raw = (
        max(0.05, min(0.99, dec)) * 0.35 +
        sig * 0.25 +
        sched * 0.30 +
        eff * 0.10
    )
    raw = max(0.02, min(0.98, raw))
    score = _clamp_score(raw)

    return GradeResult(
        score=score,
        signal_detection_score=sig,
        decision_correctness=max(0.05, min(0.99, dec)),
        overconfidence_penalty=overconf,
        scheduling_score=sched,
        efficiency_score=eff,
        details={
            "ground_truth": ground_truth,
            "agent_verdict": verdict,
            "checks_run": list(checks_run),
            "finalized_slot": finalized_slot,
            "valid_slots": valid_slots,
            "verdict_correct": verdict == ground_truth,
            "steps_used": steps_used,
        },
    )


def grade_task_hard(
    verdict: Optional[str],
    checks_run: set[str],
    steps_used: int,
    finalized_slot: Optional[str],
    proposed_slot: Optional[str],
    valid_slots: list[str],
    steps_in_scheduling: int = 0,
    **_,
) -> GradeResult:
    """
    Task Hard: Typosquat trap — 'micr0soft.com'.
    Ground truth: scam (typosquat = definitive fraud).
    If agent falls for it and schedules: scheduling score is irrelevant (verdict wrong).

    Weights:
      decision_correctness  45%
      signal_detection      35%  ← Higher weight — catching typosquat requires careful analysis
      efficiency            20%

    Bonus: If agent correctly identifies typosquat via analyze_email specifically → +0.05
    """
    required_checks = 2
    ground_truth = "scam"

    sig = _signal_detection_score(checks_run, required_checks)
    overconf = _overconfidence_penalty(checks_run, required_checks)
    eff = _efficiency_score(steps_used, max_steps=15)

    # Typosquat-specific bonus: must have run analyze_email to catch the '0' character
    typosquat_caught_bonus = 0.04 if "analyze_email" in checks_run and verdict == ground_truth else 0.0

    if verdict == ground_truth:
        dec = min(0.95 + overconf, 0.99)   # Clamped ceiling
    elif verdict is None:
        dec = 0.05                          # Floor
    else:
        # Approved a typosquatted job — severe penalty
        dec = -0.6

    raw = (
        max(0.05, min(0.99, dec)) * 0.45 +
        sig * 0.35 +
        eff * 0.20 +
        typosquat_caught_bonus
    )
    raw = max(0.02, min(0.98, raw))
    score = _clamp_score(raw)

    return GradeResult(
        score=score,
        signal_detection_score=sig,
        decision_correctness=max(0.05, min(0.99, dec)),
        overconfidence_penalty=overconf,
        scheduling_score=0.05,   # Scam — scheduling irrelevant
        efficiency_score=eff,
        details={
            "ground_truth": ground_truth,
            "agent_verdict": verdict,
            "checks_run": list(checks_run),
            "steps_used": steps_used,
            "typosquat_caught_bonus": typosquat_caught_bonus,
            "verdict_correct": verdict == ground_truth,
            "trap_triggered": verdict == "safe",   # Agent fell for the typosquat
        },
    )


# ─────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────

GRADERS = {
    "task_easy": grade_task_easy,
    "task_medium": grade_task_medium,
    "task_hard": grade_task_hard,
}


def grade(task_id: str, **kwargs) -> GradeResult:
    """
    Main entry point. kwargs passed directly to the per-task grader.
    Required kwargs per task:
      All tasks:   verdict, checks_run, steps_used
      task_medium: + finalized_slot, proposed_slot, valid_slots, steps_in_scheduling
      task_hard:   + finalized_slot, proposed_slot, valid_slots, steps_in_scheduling
    """
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task '{task_id}'. Available: {list(GRADERS.keys())}")
    return GRADERS[task_id](**kwargs)
