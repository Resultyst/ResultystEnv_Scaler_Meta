"""
ResultystEnv — Grader
Deterministic, interpretable scoring for all 3 tasks.
Breakdown: signal_detection_score + decision_correctness + overconfidence_penalty
All graders return float strictly in (0, 1) — NEVER 0.0 or 1.0.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


def _safe_float(value: float, allow_negative: bool = False) -> float:
    """
    Guaranteed safe float — NEVER returns exactly 0.0, 1.0, or -0.0.
    Returns value strictly in (0, 1) or (-1, 1) if allow_negative=True.
    """
    val = float(value)

    if allow_negative:
        if val == 0.0 or val == -0.0:
            return -0.0001
        if val <= -0.9999:
            return -0.9998
        if val >= 0.9999:
            return 0.9998
        clamped = max(-0.9998, min(0.9998, val))
        rounded = round(clamped, 4)
        if rounded == 0.0:
            return -0.0001 if val < 0 else 0.0001
        return rounded
    else:
        if val <= 0.0:
            return 0.0001
        if val >= 1.0:
            return 0.9999
        clamped = max(0.0001, min(0.9999, val))
        rounded = round(clamped, 4)
        if rounded == 0.0:
            return 0.0001
        if rounded == 1.0:
            return 0.9999
        return rounded


def _strict_clamp(value: float) -> float:
    """Ensures value is strictly between 0 and 1 (0.0001 to 0.9999)."""
    return _safe_float(value)


@dataclass
class GradeResult:
    """Interpretable grading output per task."""
    score: float
    signal_detection_score: float
    decision_correctness: float
    overconfidence_penalty: float
    scheduling_score: float
    efficiency_score: float
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        """Ensure all float fields are safe on initialization."""
        self.score = _safe_float(self.score)
        self.signal_detection_score = _safe_float(self.signal_detection_score)
        self.decision_correctness = _safe_float(self.decision_correctness)
        self.overconfidence_penalty = _safe_float(self.overconfidence_penalty, allow_negative=True)
        self.scheduling_score = _safe_float(self.scheduling_score)
        self.efficiency_score = _safe_float(self.efficiency_score)

    def as_dict(self) -> dict:
        """Return dict with guaranteed safe float values."""
        def scrub(obj):
            # IMPORTANT: bool must be checked before int (bool is subclass of int in Python)
            if isinstance(obj, bool):
                return obj
            elif isinstance(obj, float):
                return _safe_float(obj, allow_negative=True) if obj < 0 else _safe_float(obj)
            elif isinstance(obj, int):
                return _safe_float(float(obj))
            elif isinstance(obj, dict):
                return {k: scrub(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [scrub(v) for v in obj]
            elif obj is None:
                return None
            else:
                return obj

        raw = {
            "score": self.score,
            "signal_detection_score": self.signal_detection_score,
            "decision_correctness": self.decision_correctness,
            "overconfidence_penalty": self.overconfidence_penalty,
            "scheduling_score": self.scheduling_score,
            "efficiency_score": self.efficiency_score,
            "details": scrub(self.details),
        }

        return scrub(raw)


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

VERIFICATION_CHECKS = {"check_domain", "analyze_email", "verify_company"}


def _signal_detection_score(checks_run: set[str], required_checks: int) -> float:
    """Returns strictly between 0.0001 and 0.9999."""
    n_run = len(checks_run & VERIFICATION_CHECKS)
    n_total = len(VERIFICATION_CHECKS)

    if n_total == 0:
        return _safe_float(0.05)

    fraction = n_run / n_total

    if n_run >= required_checks:
        result = min(0.95, fraction + 0.1)
    else:
        result = max(0.05, fraction)

    return _safe_float(result)


def _overconfidence_penalty(checks_run: set[str], required_checks: int) -> float:
    """Returns negative float (or -0.0001). NEVER 0.0."""
    n_run = len(checks_run & VERIFICATION_CHECKS)

    if n_run >= required_checks:
        return _safe_float(-0.0001, allow_negative=True)

    missing = required_checks - n_run
    penalty = -0.15 * missing

    return _safe_float(penalty, allow_negative=True)


def _efficiency_score(steps_used: int, max_steps: int) -> float:
    """Returns strictly between 0.0001 and 0.9999."""
    if steps_used <= 0:
        return _safe_float(0.05)

    min_expected = 3
    denominator = max(1, max_steps - min_expected)
    score = (max_steps - steps_used) / denominator

    result = max(0.05, min(0.95, score))
    return _safe_float(result)


def _scheduling_quality(
    finalized_slot: Optional[str],
    valid_slots: list[str],
    proposed_slot: Optional[str],
    steps_in_scheduling: int,
) -> float:
    """Returns strictly between 0.0001 and 0.9999."""
    if not finalized_slot:
        if proposed_slot and proposed_slot in valid_slots:
            return _safe_float(0.25)
        return _safe_float(0.05)

    overlap = 0.95 if finalized_slot in valid_slots else 0.05
    comfort = _compute_comfort_score(finalized_slot)
    eff = max(0.05, min(0.95, 0.95 - (steps_in_scheduling - 1) * 0.15))

    total = overlap * 0.4 + comfort * 0.3 + eff * 0.3
    result = max(0.05, min(0.95, total))
    return _safe_float(result)


def _compute_comfort_score(slot_iso: str) -> float:
    """Returns strictly between 0.0001 and 0.9999."""
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(slot_iso)
        hour = dt.hour + dt.minute / 60

        if 8.0 <= hour < 18.0:
            return _safe_float(0.95)
        if 6.0 <= hour < 8.0 or 18.0 <= hour < 20.0:
            return _safe_float(0.50)
        return _safe_float(0.05)
    except Exception:
        return _safe_float(0.50)


# ─────────────────────────────────────────────
# Per-task Graders
# ─────────────────────────────────────────────

def grade_task_easy(
    verdict: Optional[str],
    checks_run: set[str],
    steps_used: int,
    **_,
) -> GradeResult:
    """Task Easy: Detect an obvious scam."""
    required_checks = 2
    ground_truth = "scam"

    sig = _signal_detection_score(checks_run, required_checks)
    overconf = _overconfidence_penalty(checks_run, required_checks)
    eff = _efficiency_score(steps_used, max_steps=8)

    if verdict == ground_truth:
        dec = 0.95
    elif verdict is None:
        dec = 0.05
    else:
        dec = 0.05

    dec_adjusted = dec + overconf
    dec_adjusted = max(0.05, min(0.99, dec_adjusted))

    raw = (
        dec_adjusted * 0.50 +
        sig * 0.30 +
        eff * 0.20
    )
    raw = max(0.02, min(0.98, raw))
    score = _safe_float(raw)

    return GradeResult(
        score=score,
        signal_detection_score=sig,
        decision_correctness=_safe_float(dec_adjusted),
        overconfidence_penalty=overconf,
        scheduling_score=_safe_float(0.05),
        efficiency_score=eff,
        details={
            "ground_truth": ground_truth,
            "agent_verdict": verdict,
            "checks_run": list(checks_run),
            "steps_used": steps_used,
            "verdict_correct": verdict == ground_truth,  # bool — safe
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
    """Task Medium: Borderline job — right answer is 'safe' + schedule."""
    required_checks = 3
    ground_truth = "safe"

    sig = _signal_detection_score(checks_run, required_checks)
    overconf = _overconfidence_penalty(checks_run, required_checks)
    eff = _efficiency_score(steps_used, max_steps=12)
    sched = _scheduling_quality(finalized_slot, valid_slots, proposed_slot, steps_in_scheduling)

    if verdict == ground_truth:
        dec = 0.95
    elif verdict is None:
        dec = 0.05
    else:
        dec = 0.05

    dec_adjusted = dec + overconf
    dec_adjusted = max(0.05, min(0.99, dec_adjusted))

    raw = (
        dec_adjusted * 0.35 +
        sig * 0.25 +
        sched * 0.30 +
        eff * 0.10
    )
    raw = max(0.02, min(0.98, raw))
    score = _safe_float(raw)

    return GradeResult(
        score=score,
        signal_detection_score=sig,
        decision_correctness=_safe_float(dec_adjusted),
        overconfidence_penalty=overconf,
        scheduling_score=sched,
        efficiency_score=eff,
        details={
            "ground_truth": ground_truth,
            "agent_verdict": verdict,
            "checks_run": list(checks_run),
            "finalized_slot": finalized_slot,
            "valid_slots": valid_slots,
            "verdict_correct": verdict == ground_truth,  # bool — safe
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
    """Task Hard: Typosquat trap — 'micr0soft.com'."""
    required_checks = 2
    ground_truth = "scam"

    sig = _signal_detection_score(checks_run, required_checks)
    overconf = _overconfidence_penalty(checks_run, required_checks)
    eff = _efficiency_score(steps_used, max_steps=15)

    typosquat_caught_bonus = (
        _safe_float(0.04)
        if "analyze_email" in checks_run and verdict == ground_truth
        else _safe_float(0.0001)
    )

    if verdict == ground_truth:
        dec = 0.95
    elif verdict is None:
        dec = 0.05
    else:
        dec = 0.05

    dec_adjusted = dec + overconf
    dec_adjusted = max(0.05, min(0.99, dec_adjusted))

    raw = (
        dec_adjusted * 0.45 +
        sig * 0.35 +
        eff * 0.20 +
        typosquat_caught_bonus
    )
    raw = max(0.02, min(0.98, raw))
    score = _safe_float(raw)

    return GradeResult(
        score=score,
        signal_detection_score=sig,
        decision_correctness=_safe_float(dec_adjusted),
        overconfidence_penalty=overconf,
        scheduling_score=_safe_float(0.05),
        efficiency_score=eff,
        details={
            "ground_truth": ground_truth,
            "agent_verdict": verdict,
            "checks_run": list(checks_run),
            "steps_used": steps_used,
            "typosquat_caught_bonus": typosquat_caught_bonus,
            "verdict_correct": verdict == ground_truth,  # bool — safe
            "trap_triggered": verdict == "safe",         # bool — safe
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
    """Main entry point."""
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task '{task_id}'. Available: {list(GRADERS.keys())}")
    return GRADERS[task_id](**kwargs)
