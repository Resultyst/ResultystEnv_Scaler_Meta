"""
ResultystEnv — Core Environment
Implements the OpenEnv interface: reset(), step(), state().

Reward design:
  • Context-aware: signal_strength × correctness (not flat fixed values)
  • Overconfidence penalty: deciding without enough investigation → -0.2
  • Loop penalty: same action repeated >3× → -0.02 per repeat
  • Scheduling: overlap_score×0.4 + comfort_score×0.3 + efficiency×0.3
  • All rewards normalized to [0.0001, 0.9999] at episode end
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Optional

from .models import (
    Action,
    Calendars,
    CompanyInfo,
    EnvState,
    JobPost,
    Observation,
    Reward,
    RewardBreakdown,
    SCHEDULING_ACTIONS,
    VERIFICATION_ACTIONS,
)
from .tasks import get_task
from .grader import grade, _compute_comfort_score, _strict_clamp

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MAX_REWARD_ACCUMULATE = 2.0   # Cap before normalisation (prevents runaway)
LOOP_THRESHOLD = 3            # Repeats before loop penalty kicks in


def _safe_clamp(value: float) -> float:
    """Ensure value is strictly between 0 and 1 (0.0001 to 0.9999)."""
    val = float(value)
    clamped = max(0.0001, min(0.9999, val))
    return round(clamped, 4)


class ResultystEnv:
    """
    OpenEnv-compatible two-stage environment.
    Stage 1 — verification: Detect whether a job post is a scam.
    Stage 2 — scheduling:   Schedule an interview (only for legit jobs).
    """

    def __init__(self) -> None:
        self._state: Optional[EnvState] = None
        self._task_meta: Optional[dict] = None
        self._accumulated_reward: float = 0.0001  # Start with small positive
        self._action_counts: Counter = Counter()
        self._scheduling_steps: int = 0

    # ─────────────────────────────────────────
    # OpenEnv Interface
    # ─────────────────────────────────────────

    def reset(self, task_id: str = "task_easy") -> Observation:
        """Initialize a fresh episode for the given task."""
        task = get_task(task_id)

        self._task_meta = task
        self._accumulated_reward = 0.0001  # Small positive instead of 0.0
        self._action_counts = Counter()
        self._scheduling_steps = 0

        self._state = EnvState(
            task_id=task_id,
            stage="verification",
            job_post=copy.deepcopy(task["job_post"]),
            company_info=CompanyInfo(),
            calendars=copy.deepcopy(task["calendars"]),
            history=[],
            step_count=0,
            done=False,
            verdict=None,
            scheduled_time=None,
            checks_completed=set(),
        )

        return self._build_observation(
            message=(
                f"Episode started. Task: '{task['name']}' [{task['difficulty'].upper()}]. "
                f"Analyze the job posting and investigate before making a decision."
            ),
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute one action and return (observation, reward, done, info).
        Raises RuntimeError if reset() has not been called.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        s = self._state
        s.step_count += 1
        atype = action.action_type

        # ── Loop penalty check ─────────────────
        self._action_counts[atype] += 1
        loop_penalty = 0.0001  # Small positive instead of 0.0
        if self._action_counts[atype] > LOOP_THRESHOLD:
            loop_penalty = -0.0201
            self._accumulated_reward = max(0.0001, self._accumulated_reward + loop_penalty)

        # ── Stage gate ─────────────────────────
        if s.stage == "verification" and atype in SCHEDULING_ACTIONS:
            reward = self._build_reward(
                value=_safe_clamp(0.0001),
                breakdown=RewardBreakdown(loop_penalty=_safe_clamp(abs(loop_penalty))),
                reason=f"❌ Action '{atype}' is not available in verification stage.",
            )
            obs = self._build_observation(
                message=f"Invalid action: '{atype}' can only be used in the scheduling stage.",
            )
            self._record_history(action, reward)
            return obs, reward, False, {"error": "wrong_stage"}

        if s.stage == "scheduling" and atype in VERIFICATION_ACTIONS:
            reward = self._build_reward(
                value=_safe_clamp(0.0001),
                breakdown=RewardBreakdown(loop_penalty=_safe_clamp(abs(loop_penalty))),
                reason=f"❌ Action '{atype}' is not available in scheduling stage.",
            )
            obs = self._build_observation(
                message=f"Invalid action: '{atype}' can only be used in the verification stage.",
            )
            self._record_history(action, reward)
            return obs, reward, False, {"error": "wrong_stage"}

        # ── Dispatch ───────────────────────────
        if s.stage == "verification":
            obs, reward = self._handle_verification(action, loop_penalty)
        else:
            obs, reward = self._handle_scheduling(action, loop_penalty)

        self._record_history(action, reward)
        return obs, reward, s.done, self._build_info()

    def state(self) -> EnvState:
        """Return full internal state (for debugging / state() endpoint)."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ─────────────────────────────────────────
    # Verification Stage
    # ─────────────────────────────────────────

    def _handle_verification(
        self, action: Action, loop_penalty: float
    ) -> tuple[Observation, Reward]:
        s = self._state
        task = self._task_meta
        atype = action.action_type
        truth: CompanyInfo = task["company_info_truth"]
        signal_map: dict = task["signal_map"]

        # ── Investigation actions ──────────────
        if atype == "check_domain":
            is_new = "check_domain" not in s.checks_completed
            s.checks_completed.add("check_domain")
            s.company_info.domain_age_days = truth.domain_age_days
            s.company_info.has_https = truth.has_https
            s.company_info.registrar = truth.registrar

            strength = signal_map["check_domain"] if is_new else 0.0001
            step_base = 0.0201 if is_new else 0.0001
            raw_val = strength + step_base + loop_penalty

            age = truth.domain_age_days or 9999
            if age <= 7:
                msg = (
                    f"🔴 Domain '{s.job_post.domain}' is only {age} day(s) old — "
                    f"HTTPS: {truth.has_https}, Registrar: {truth.registrar}. Very suspicious."
                )
            elif age <= 180:
                msg = (
                    f"🟡 Domain '{s.job_post.domain}' is {age} days old. "
                    f"HTTPS: {truth.has_https}, Registrar: {truth.registrar}. Investigate further."
                )
            else:
                msg = (
                    f"🟢 Domain '{s.job_post.domain}' is {age} days old. "
                    f"HTTPS: {truth.has_https}. Age alone isn't suspicious."
                )

            reward = self._build_reward(
                value=self._safe_clip(raw_val),
                breakdown=RewardBreakdown(
                    signal_strength=_safe_clamp(strength),
                    step_taken=_safe_clamp(step_base),
                    loop_penalty=_safe_clamp(abs(loop_penalty)) if loop_penalty < 0 else 0.0001,
                ),
                reason=f"check_domain: signal_strength={strength:.2f} (domain_age={age}d).",
            )
            return self._build_observation(message=msg), reward

        if atype == "analyze_email":
            is_new = "analyze_email" not in s.checks_completed
            s.checks_completed.add("analyze_email")
            s.company_info.email_domain_match = truth.email_domain_match
            s.company_info.typosquat_detected = truth.typosquat_detected

            strength = signal_map["analyze_email"] if is_new else 0.0001
            step_base = 0.0201 if is_new else 0.0001

            if truth.typosquat_detected:
                msg = (
                    f"🔴 TYPOSQUAT DETECTED in '{s.job_post.email}' — "
                    f"character substitution found. This is NOT the real domain. "
                    f"Real domain likely: '{truth.raw_signals.get('real_microsoft_domain', 'unknown')}'."
                )
                if is_new:
                    strength += 0.0501
            elif not truth.email_domain_match:
                msg = (
                    f"🟡 Email '{s.job_post.email}' does not match company domain. "
                    f"Recruiter is using a personal/free email provider. Moderate concern."
                )
            else:
                msg = (
                    f"🟢 Email '{s.job_post.email}' matches the company domain. "
                    f"No immediate red flags from email analysis."
                )

            raw_val = strength + step_base + loop_penalty
            reward = self._build_reward(
                value=self._safe_clip(raw_val),
                breakdown=RewardBreakdown(
                    signal_strength=_safe_clamp(strength),
                    step_taken=_safe_clamp(step_base),
                    loop_penalty=_safe_clamp(abs(loop_penalty)) if loop_penalty < 0 else 0.0001,
                ),
                reason=f"analyze_email: signal_strength={strength:.2f}, typosquat={truth.typosquat_detected}.",
            )
            return self._build_observation(message=msg), reward

        if atype == "verify_company":
            is_new = "verify_company" not in s.checks_completed
            s.checks_completed.add("verify_company")
            s.company_info.company_registered = truth.company_registered
            s.company_info.employee_count = truth.employee_count
            s.company_info.glassdoor_reviews = truth.glassdoor_reviews
            s.company_info.linkedin_present = truth.linkedin_present

            strength = signal_map["verify_company"] if is_new else 0.0001
            step_base = 0.0201 if is_new else 0.0001

            if truth.company_registered is False:
                msg = (
                    f"🔴 '{s.job_post.company}' has NO official company registration. "
                    f"LinkedIn: {truth.linkedin_present}, Reviews: none. Highly suspicious."
                )
            elif truth.company_registered and truth.linkedin_present:
                msg = (
                    f"🟢 '{s.job_post.company}' is registered. "
                    f"LinkedIn: present, Glassdoor: {truth.glassdoor_reviews:.1f}/5, "
                    f"Employees: {truth.employee_count}. Appears legitimate."
                )
            else:
                msg = (
                    f"🟡 '{s.job_post.company}' registered but LinkedIn absent. "
                    f"Glassdoor: {truth.glassdoor_reviews}, Employees: {truth.employee_count}. Mixed signals."
                )

            raw_val = strength + step_base + loop_penalty
            reward = self._build_reward(
                value=self._safe_clip(raw_val),
                breakdown=RewardBreakdown(
                    signal_strength=_safe_clamp(strength),
                    step_taken=_safe_clamp(step_base),
                    loop_penalty=_safe_clamp(abs(loop_penalty)) if loop_penalty < 0 else 0.0001,
                ),
                reason=f"verify_company: signal_strength={strength:.2f}.",
            )
            return self._build_observation(message=msg), reward

        # ── Verdict actions ────────────────────
        if atype == "mark_safe":
            return self._handle_mark_safe(loop_penalty)

        if atype == "reject_job":
            return self._handle_reject_job(loop_penalty)

        # Fallback
        reward = self._build_reward(
            value=self._safe_clip(0.0001),
            breakdown=RewardBreakdown(),
            reason="Unknown action.",
        )
        return self._build_observation(message="Unknown action."), reward

    def _handle_mark_safe(self, loop_penalty: float) -> tuple[Observation, Reward]:
        s = self._state
        task = self._task_meta
        ground_truth = task["ground_truth"]
        required_checks = task["required_checks"]
        n_checks = len(s.checks_completed)

        overconf = 0.0001
        if n_checks < required_checks:
            missing = required_checks - n_checks
            overconf = -0.2001 * missing
            msg_prefix = (
                f"⚠️ mark_safe called after only {n_checks}/{required_checks} required checks. "
                f"Overconfidence penalty applied."
            )
        else:
            msg_prefix = f"✅ mark_safe: sufficient evidence gathered ({n_checks} checks)."

        if ground_truth == "safe":
            decision_reward = 0.3001
            s.stage = "scheduling"
            s.verdict = "safe"
            msg = msg_prefix + " Verdict: SAFE. Moving to interview scheduling stage."
        else:
            decision_reward = -0.3501
            s.done = True
            s.verdict = "safe"
            msg = msg_prefix + f" ❌ WRONG — ground truth is '{ground_truth}'. Approved a scam post!"

        raw_val = decision_reward + overconf + loop_penalty
        self._accumulated_reward = max(0.0001, self._accumulated_reward + raw_val)

        reward = self._build_reward(
            value=self._safe_clip(raw_val),
            breakdown=RewardBreakdown(
                decision_correctness=_safe_clamp(max(0.0001, decision_reward)),
                overconfidence_penalty=_safe_clamp(abs(overconf)) if overconf < 0 else 0.0001,
                loop_penalty=_safe_clamp(abs(loop_penalty)) if loop_penalty < 0 else 0.0001,
            ),
            reason=f"mark_safe: decision_reward={decision_reward:.2f}, overconf={overconf:.2f}.",
        )
        return self._build_observation(message=msg), reward

    def _handle_reject_job(self, loop_penalty: float) -> tuple[Observation, Reward]:
        s = self._state
        task = self._task_meta
        ground_truth = task["ground_truth"]
        required_checks = task["required_checks"]
        n_checks = len(s.checks_completed)

        overconf = 0.0001
        if n_checks < required_checks:
            missing = required_checks - n_checks
            overconf = -0.2001 * missing
            msg_prefix = (
                f"⚠️ reject_job called after only {n_checks}/{required_checks} required checks. "
                f"Overconfidence penalty applied."
            )
        else:
            msg_prefix = f"✅ reject_job: sufficient evidence gathered ({n_checks} checks)."

        s.done = True
        s.verdict = "scam"

        if ground_truth == "scam":
            decision_reward = 0.5001
            msg = msg_prefix + " Verdict: SCAM ✓ Correct! Job post rejected."
        else:
            decision_reward = -0.4501
            msg = msg_prefix + f" ❌ WRONG — ground truth is '{ground_truth}'. Rejected a legitimate job!"

        raw_val = decision_reward + overconf + loop_penalty
        self._accumulated_reward = max(0.0001, self._accumulated_reward + raw_val)

        reward = self._build_reward(
            value=self._safe_clip(raw_val),
            breakdown=RewardBreakdown(
                decision_correctness=_safe_clamp(max(0.0001, decision_reward)),
                overconfidence_penalty=_safe_clamp(abs(overconf)) if overconf < 0 else 0.0001,
                loop_penalty=_safe_clamp(abs(loop_penalty)) if loop_penalty < 0 else 0.0001,
            ),
            reason=f"reject_job: decision_reward={decision_reward:.2f}, overconf={overconf:.2f}.",
        )
        return self._build_observation(message=msg), reward

    # ─────────────────────────────────────────
    # Scheduling Stage
    # ─────────────────────────────────────────

    def _handle_scheduling(
        self, action: Action, loop_penalty: float
    ) -> tuple[Observation, Reward]:
        s = self._state
        task = self._task_meta
        atype = action.action_type
        self._scheduling_steps += 1
        valid_slots: list[str] = s.calendars.timezone_overlap_windows

        if atype == "propose_time":
            slot = action.parameters.get("slot", "")
            is_new = slot != s.calendars.proposed_slot
            s.calendars.proposed_slot = slot

            if slot in valid_slots:
                comfort = _compute_comfort_score(slot)
                if is_new:
                    overlap = 0.9499
                    eff = max(0.0501, min(0.9499, 0.9999 - (self._scheduling_steps - 1) * 0.1001))
                    raw_reward = overlap * 0.0501 + comfort * 0.0301 + eff * 0.0201
                else:
                    raw_reward = 0.0101
                msg = (
                    f"✅ '{slot}' is a valid overlap slot. "
                    f"Comfort score: {comfort:.1f}. Good proposal — now finalize or reschedule."
                )
            else:
                raw_reward = 0.0101 if is_new else 0.0001
                msg = (
                    f"🟡 '{slot}' is proposed but doesn't align with both calendars. "
                    f"Consider rescheduling."
                )

            reward = self._build_reward(
                value=self._safe_clip(raw_reward + loop_penalty),
                breakdown=RewardBreakdown(
                    overlap_score=_safe_clamp(0.9499 if slot in valid_slots else 0.0101),
                    comfort_score=_safe_clamp(_compute_comfort_score(slot) if slot in valid_slots else 0.0101),
                    step_taken=_safe_clamp(raw_reward),
                    loop_penalty=_safe_clamp(abs(loop_penalty)) if loop_penalty < 0 else 0.0001,
                ),
                reason=f"propose_time: slot={slot!r}, valid={slot in valid_slots}.",
            )
            return self._build_observation(message=msg), reward

        if atype == "reschedule":
            reason = action.parameters.get("reason", "No reason given.")
            raw_reward = -0.0101
            msg = f"🔄 Rescheduling requested. Reason: {reason}. Propose a new slot."
            reward = self._build_reward(
                value=self._safe_clip(raw_reward + loop_penalty),
                breakdown=RewardBreakdown(
                    scheduling_efficiency=_safe_clamp(max(0.0101, -0.0501 * self._scheduling_steps)),
                    loop_penalty=_safe_clamp(abs(loop_penalty)) if loop_penalty < 0 else 0.0001,
                ),
                reason=f"reschedule: step {self._scheduling_steps}, reason='{reason}'.",
            )
            return self._build_observation(message=msg), reward

        if atype == "finalize_schedule":
            return self._handle_finalize(loop_penalty, valid_slots)

        reward = self._build_reward(
            value=self._safe_clip(0.0001),
            breakdown=RewardBreakdown(),
            reason="Unknown scheduling action.",
        )
        return self._build_observation(message="Unknown scheduling action."), reward

    def _handle_finalize(
        self, loop_penalty: float, valid_slots: list[str]
    ) -> tuple[Observation, Reward]:
        s = self._state
        slot = s.calendars.proposed_slot or action_param_slot(s)

        s.done = True
        s.scheduled_time = slot
        s.calendars.finalized_slot = slot

        if slot and slot in valid_slots:
            comfort = _compute_comfort_score(slot)
            eff = max(0.0501, min(0.9499, 0.9999 - (self._scheduling_steps - 1) * 0.1001))
            overlap_score = 0.9499
            sched_reward = overlap_score * 0.4001 + comfort * 0.3001 + eff * 0.3001
            late_penalty = -0.1001 if comfort <= 0.0501 else 0.0001
            final = sched_reward + late_penalty + loop_penalty
            msg = (
                f"🎉 Interview finalized at '{slot}'. "
                f"Overlap: ✓, Comfort: {comfort:.1f}/1.0, Efficiency: {eff:.2f}. "
                f"Episode complete!"
            )
            breakdown = RewardBreakdown(
                overlap_score=_safe_clamp(overlap_score),
                comfort_score=_safe_clamp(comfort),
                scheduling_efficiency=_safe_clamp(eff),
                loop_penalty=_safe_clamp(abs(loop_penalty)) if loop_penalty < 0 else 0.0001,
            )
        elif slot:
            final = 0.0501 + loop_penalty
            msg = (
                f"⚠️ Interview 'finalized' at '{slot}' but this slot has no valid overlap. "
                f"One party will be inconvenienced."
            )
            breakdown = RewardBreakdown(
                overlap_score=0.0101,
                comfort_score=0.0101,
                loop_penalty=_safe_clamp(abs(loop_penalty)) if loop_penalty < 0 else 0.0001,
            )
        else:
            final = 0.0101
            msg = "❌ finalize_schedule called without a proposed slot."
            breakdown = RewardBreakdown()

        self._accumulated_reward = max(0.0001, self._accumulated_reward + final)
        reward = self._build_reward(
            value=self._safe_clip(final),
            breakdown=breakdown,
            reason=f"finalize_schedule: slot={slot!r}, in_valid={slot in valid_slots if slot else False}.",
        )
        return self._build_observation(message=msg), reward

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _build_observation(self, message: str) -> Observation:
        s = self._state
        available = (
            list(VERIFICATION_ACTIONS) if s.stage == "verification"
            else list(SCHEDULING_ACTIONS)
        )
        return Observation(
            stage=s.stage,
            job_post=s.job_post,
            company_info=s.company_info,
            calendars=s.calendars,
            message=message,
            available_actions=available,
            step_count=s.step_count,
            done=s.done,
        )

    @staticmethod
    def _safe_clip(val: float) -> float:
        """Ensure value is strictly between 0 and 1."""
        return _safe_clamp(val)

    def _build_reward(
        self,
        value: float,
        breakdown: RewardBreakdown,
        reason: str,
    ) -> Reward:
        self._accumulated_reward = max(
            0.0001, min(MAX_REWARD_ACCUMULATE, self._accumulated_reward + value)
        )

        def _scrub(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, (float, int)):
                return _safe_clamp(float(v))
            return v

        bd_dict = breakdown.model_dump()
        scrubbed_bd = {k: _scrub(v) for k, v in bd_dict.items()}

        return Reward(
            value=_safe_clamp(value),
            breakdown=RewardBreakdown(**scrubbed_bd),
            reason=reason,
        )

    def _build_info(self) -> dict[str, Any]:
        s = self._state

        def _scrub(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, (float, int)):
                return _safe_clamp(float(v))
            if isinstance(v, list):
                return [_scrub(x) for x in v]
            if isinstance(v, dict):
                return {k: _scrub(val) for k, val in v.items()}
            return v

        raw_info = {
            "task_id": s.task_id,
            "stage": s.stage,
            "step_count": s.step_count,
            "checks_completed": list(s.checks_completed),
            "verdict": s.verdict,
            "accumulated_reward": self._accumulated_reward,
        }
        return {k: _scrub(v) for k, v in raw_info.items()}

    def _record_history(self, action: Action, reward: Reward) -> None:
        self._state.history.append({
            "step": self._state.step_count,
            "action": action.model_dump(),
            "reward": reward.value,
            "message": None,
        })

    # ─────────────────────────────────────────
    # Final episode grading
    # ─────────────────────────────────────────

    def grade_episode(self) -> dict:
        """Run the deterministic grader on the completed episode."""
        if self._state is None:
            raise RuntimeError("No episode to grade.")
        s = self._state
        result = grade(
            task_id=s.task_id,
            verdict=s.verdict,
            checks_run=s.checks_completed,
            steps_used=s.step_count,
            finalized_slot=s.calendars.finalized_slot,
            proposed_slot=s.calendars.proposed_slot,
            valid_slots=s.calendars.timezone_overlap_windows,
            steps_in_scheduling=self._scheduling_steps,
        )
        return result.as_dict()


def action_param_slot(state: EnvState) -> Optional[str]:
    """Extract any slot proposed in the last history entry."""
    if state.history:
        last = state.history[-1]
        return last.get("action", {}).get("parameters", {}).get("slot")
    return None
