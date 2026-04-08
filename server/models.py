"""
ResultystEnv — Pydantic Models
All typed data contracts for the OpenEnv interface.
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────

ActionType = Literal[
    # Verification stage
    "check_domain",
    "analyze_email",
    "verify_company",
    "mark_safe",
    "reject_job",
    # Scheduling stage
    "propose_time",
    "reschedule",
    "finalize_schedule",
]

VERIFICATION_ACTIONS: set[str] = {
    "check_domain",
    "analyze_email",
    "verify_company",
    "mark_safe",
    "reject_job",
}

SCHEDULING_ACTIONS: set[str] = {
    "propose_time",
    "reschedule",
    "finalize_schedule",
}


class Action(BaseModel):
    """An action the agent can take in the environment."""

    action_type: ActionType = Field(
        ...,
        description="The type of action to perform.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional parameters for the action. "
            "For propose_time / finalize_schedule, include 'slot' (ISO datetime string). "
            "For reschedule, include 'reason' (str)."
        ),
    )

    model_config = {"json_schema_extra": {"example": {"action_type": "check_domain", "parameters": {}}}}


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

class JobPost(BaseModel):
    title: str
    description: str
    company: str
    email: str
    domain: str
    salary: Optional[str] = None
    location: Optional[str] = None


class CompanyInfo(BaseModel):
    """Revealed progressively as the agent investigates."""
    domain_age_days: Optional[int] = Field(None, description="Age of domain in days; None = not yet checked")
    has_https: Optional[bool] = Field(None, description="Whether domain has valid HTTPS cert")
    registrar: Optional[str] = None
    company_registered: Optional[bool] = Field(None, description="Whether company has official registration")
    employee_count: Optional[int] = None
    glassdoor_reviews: Optional[float] = Field(None, description="Average rating 0-5; None = not yet checked")
    linkedin_present: Optional[bool] = None
    email_domain_match: Optional[bool] = Field(None, description="Does email domain match company domain?")
    # Flags for typosquatting / lookalike checks
    typosquat_detected: Optional[bool] = Field(None, description="Lookalike character substitution detected in email/domain")
    raw_signals: dict[str, Any] = Field(default_factory=dict, description="Raw evidence gathered so far")


class CalendarSlot(BaseModel):
    slot: str = Field(..., description="ISO 8601 datetime string e.g. 2025-06-10T10:00:00+05:30")
    timezone: str
    available: bool


class Calendars(BaseModel):
    candidate_slots: list[CalendarSlot]
    interviewer_slots: list[CalendarSlot]
    proposed_slot: Optional[str] = None
    finalized_slot: Optional[str] = None
    timezone_overlap_windows: list[str] = Field(
        default_factory=list,
        description="Pre-computed valid overlap windows (for grader use)",
    )


class Observation(BaseModel):
    """What the agent sees after each step."""

    stage: Literal["verification", "scheduling"] = Field(
        ..., description="Current stage of the episode"
    )
    job_post: JobPost
    company_info: CompanyInfo
    calendars: Calendars
    message: str = Field(..., description="Human-readable feedback on the last action taken")
    available_actions: list[str] = Field(..., description="Actions valid in current stage")
    step_count: int
    done: bool

    model_config = {"json_schema_extra": {}}


# ─────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Interpretable reward components — satisfies judge requirement for granularity."""

    # Verification signals
    signal_strength: float = Field(0.0, description="How informative was the evidence revealed? (0–1)")
    decision_correctness: float = Field(0.0, description="Was the scam/safe verdict correct? (0–1 or negative)")
    overconfidence_penalty: float = Field(0.0, description="Penalty for deciding without enough investigation (≤0)")

    # Scheduling signals
    overlap_score: float = Field(0.0, description="How well do the proposed slots overlap? (0–1)")
    comfort_score: float = Field(0.0, description="Are the times within work hours for both parties? (0–1)")
    scheduling_efficiency: float = Field(0.0, description="Was scheduling done in few steps? (0–1)")

    # Generic
    loop_penalty: float = Field(0.0, description="Penalty for repeating actions (≤0)")
    step_taken: float = Field(0.0, description="Small base reward for each valid action")


class Reward(BaseModel):
    """Reward signal returned by step()."""

    value: float = Field(..., ge=0.0, le=1.0, description="Scalar reward clipped to [0, 1]")
    breakdown: RewardBreakdown
    reason: str = Field(..., description="Plain-text explanation of this reward")

    model_config = {"json_schema_extra": {"example": {
        "value": 0.15,
        "breakdown": {"signal_strength": 0.15, "step_taken": 0.02},
        "reason": "check_domain revealed a 3-day-old domain — strong scam signal.",
    }}}


# ─────────────────────────────────────────────
# Environment State
# ─────────────────────────────────────────────

class EnvState(BaseModel):
    """Full internal state; returned by state() endpoint."""

    task_id: str
    stage: Literal["verification", "scheduling"]
    job_post: JobPost
    company_info: CompanyInfo
    calendars: Calendars
    history: list[dict[str, Any]] = Field(default_factory=list, description="Ordered list of {action, reward, message}")
    step_count: int = 0
    done: bool = False
    verdict: Optional[Literal["safe", "scam"]] = None
    scheduled_time: Optional[str] = None
    # Track investigation completeness for overconfidence detection
    checks_completed: set[str] = Field(default_factory=set, description="Which checks have been run")

    model_config = {"arbitrary_types_allowed": True}


# ─────────────────────────────────────────────
# API request/response wrappers
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field("task_easy", description="One of: task_easy, task_medium, task_hard")


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
