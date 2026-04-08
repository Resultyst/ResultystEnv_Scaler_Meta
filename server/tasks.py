"""
ResultystEnv — Task Definitions
Three scenarios: easy → medium → hard (typosquat trap).
All data is deterministic and pre-seeded — no external API calls needed.
"""

from __future__ import annotations
from typing import Any
from .models import JobPost, CompanyInfo, Calendars, CalendarSlot


def _make_calendars_easy() -> Calendars:
    """No scheduling needed for scam — placeholder calendars."""
    return Calendars(
        candidate_slots=[
            CalendarSlot(slot="2025-06-10T10:00:00+05:30", timezone="Asia/Kolkata", available=True),
        ],
        interviewer_slots=[],
        timezone_overlap_windows=[],
    )


def _make_calendars_medium() -> Calendars:
    """Borderline job — scheduling needed after verification."""
    return Calendars(
        candidate_slots=[
            CalendarSlot(slot="2025-06-12T09:00:00+05:30", timezone="Asia/Kolkata", available=True),
            CalendarSlot(slot="2025-06-12T11:00:00+05:30", timezone="Asia/Kolkata", available=True),
            CalendarSlot(slot="2025-06-12T15:00:00+05:30", timezone="Asia/Kolkata", available=False),
            CalendarSlot(slot="2025-06-13T10:00:00+05:30", timezone="Asia/Kolkata", available=True),
        ],
        interviewer_slots=[
            CalendarSlot(slot="2025-06-12T09:30:00+05:30", timezone="Asia/Kolkata", available=True),
            CalendarSlot(slot="2025-06-12T11:00:00+05:30", timezone="Asia/Kolkata", available=True),
            CalendarSlot(slot="2025-06-13T09:00:00+05:30", timezone="Asia/Kolkata", available=False),
            CalendarSlot(slot="2025-06-13T10:00:00+05:30", timezone="Asia/Kolkata", available=True),
        ],
        # Both are available at these two slots
        timezone_overlap_windows=[
            "2025-06-12T11:00:00+05:30",
            "2025-06-13T10:00:00+05:30",
        ],
    )


def _make_calendars_hard() -> Calendars:
    """
    Hard multi-timezone scheduling.
    Candidate: IST (UTC+5:30), Interviewer: PST (UTC-8).
    Interviewer 9AM PST = 22:30 IST — outside comfort hours.
    Only valid overlap: 06:30 IST = 17:00 PST-1d (within interviewer business hours).
    Correct slot: candidate proposes 2025-06-15T06:30:00+05:30 (= 2025-06-14T17:00:00-08:00 for interviewer).
    """
    return Calendars(
        candidate_slots=[
            # Very early for candidate (6:30 AM IST) — technically available
            CalendarSlot(slot="2025-06-15T06:30:00+05:30", timezone="Asia/Kolkata", available=True),
            # Comfortable for candidate (10:00 AM IST) but 8:30 PM PST — outside interviewer hours
            CalendarSlot(slot="2025-06-15T10:00:00+05:30", timezone="Asia/Kolkata", available=True),
            # Afternoon IST — deep night PST, invalid
            CalendarSlot(slot="2025-06-15T14:00:00+05:30", timezone="Asia/Kolkata", available=True),
            # Conflicted slot
            CalendarSlot(slot="2025-06-15T16:00:00+05:30", timezone="Asia/Kolkata", available=False),
        ],
        interviewer_slots=[
            # 17:00 PST = 06:30 IST next day — only overlap
            CalendarSlot(slot="2025-06-14T17:00:00-08:00", timezone="America/Los_Angeles", available=True),
            # 09:00 PST = 22:30 IST — candidate sleeping hours
            CalendarSlot(slot="2025-06-15T09:00:00-08:00", timezone="America/Los_Angeles", available=True),
            # 14:00 PST = 03:30 IST — impossible for candidate
            CalendarSlot(slot="2025-06-15T14:00:00-08:00", timezone="America/Los_Angeles", available=True),
        ],
        # Only one true overlap that satisfies both sides
        timezone_overlap_windows=["2025-06-15T06:30:00+05:30"],
    )


# ─────────────────────────────────────────────
# Task Registry
# ─────────────────────────────────────────────

TASKS: dict[str, dict[str, Any]] = {

    # ── EASY ──────────────────────────────────
    "task_easy": {
        "name": "Detect an Obvious Job Scam",
        "difficulty": "easy",
        "description": (
            "A blatantly fraudulent job posting from a brand-new domain. "
            "The agent should investigate and reject without getting fooled."
        ),
        "ground_truth": "scam",
        "max_steps": 8,
        "job_post": JobPost(
            title="Work From Home Data Entry Specialist — Earn $5,000/Week!",
            description=(
                "No experience needed! Work from the comfort of your home. "
                "Process simple online forms and earn guaranteed weekly payments. "
                "Immediate hiring — 500 positions open. Apply NOW via email."
            ),
            company="QuickCash Global Ltd.",
            email="jobs@quickcash-global.xyz",
            domain="quickcash-global.xyz",
            salary="$5,000/week",
            location="Remote (Worldwide)",
        ),
        "company_info_truth": CompanyInfo(
            domain_age_days=3,           # 3-day-old domain — VERY strong scam signal
            has_https=False,             # No HTTPS
            registrar="NameSilo LLC",
            company_registered=False,    # No official registration found
            employee_count=0,
            glassdoor_reviews=None,      # Zero presence
            linkedin_present=False,
            email_domain_match=True,     # Email matches domain (but both are scam)
            typosquat_detected=False,
            raw_signals={
                "whois_created": "2025-06-07",
                "ssl_valid": False,
                "spam_blacklist": True,
                "company_registry_hit": False,
            },
        ),
        "calendars": _make_calendars_easy(),
        # Signal strength per check action (used for context-aware rewards)
        "signal_map": {
            "check_domain": 0.18,    # Domain age 3 days → very strong
            "analyze_email": 0.14,   # .xyz + spam blacklist → strong
            "verify_company": 0.16,  # No registration → strong
        },
        "required_checks": 2,        # Min checks before verdict to avoid overconfidence penalty
    },

    # ── MEDIUM ────────────────────────────────
    "task_medium": {
        "name": "Evaluate a Borderline Job Posting",
        "difficulty": "medium",
        "description": (
            "A real-looking startup role with mixed signals. "
            "The company is registered and the domain is 2 years old, "
            "but the recruiter uses Gmail and the salary is unusually high. "
            "Agent must collect all 3 verification signals before deciding."
        ),
        "ground_truth": "safe",
        "max_steps": 12,
        "job_post": JobPost(
            title="Senior Product Manager — Fast-Growing FinTech Startup",
            description=(
                "Join our Series A FinTech startup disrupting SME lending in SEA. "
                "Looking for a senior PM with 5+ years experience. Competitive equity, "
                "benefits, and $180K base. Interview process: 1 HR round + 2 technical rounds."
            ),
            company="NovaPay Technologies Pte. Ltd.",
            email="hiring.novapay@gmail.com",  # Gmail — suspicious but not fatal
            domain="novapaytechnologies.com",
            salary="$180,000/year",
            location="Singapore (Hybrid)",
        ),
        "company_info_truth": CompanyInfo(
            domain_age_days=730,         # 2-year-old domain — weak scam signal
            has_https=True,
            registrar="GoDaddy",
            company_registered=True,     # Registered in Singapore ACRA
            employee_count=42,
            glassdoor_reviews=3.8,       # OK-ish reviews
            linkedin_present=True,
            email_domain_match=False,    # Gmail ≠ company domain — mild signal
            typosquat_detected=False,
            raw_signals={
                "whois_created": "2023-06-01",
                "ssl_valid": True,
                "spam_blacklist": False,
                "company_registry_hit": True,
                "acra_registration": "202312345K",
                "linkedin_employees": 38,
            },
        ),
        "calendars": _make_calendars_medium(),
        "signal_map": {
            "check_domain": 0.04,    # 2-year domain → weak signal (not suspicious)
            "analyze_email": 0.09,   # Gmail for corporate → moderate signal
            "verify_company": 0.05,  # Registered company → low suspicion
        },
        "required_checks": 3,        # All 3 checks required — mixed signals need full investigation
    },

    # ── HARD ──────────────────────────────────
    "task_hard": {
        "name": "Cross-Timezone Scheduling with Typosquat Trap",
        "difficulty": "hard",
        "description": (
            "The job posting looks professional — but the recruiter email uses '0' (zero) "
            "instead of 'o' in 'microsoft'. A careful agent will catch this typosquat. "
            "If it passes verification correctly (reject), the episode ends. "
            "The TRAP: shallow agents will miss the character substitution and proceed "
            "to scheduling mode where they'll encounter a complex IST/PST calendar puzzle."
        ),
        "ground_truth": "scam",          # The email typosquat is definitive fraud
        "max_steps": 15,
        "job_post": JobPost(
            title="Software Engineer II — Cloud Infrastructure",
            description=(
                "Microsoft is hiring a Software Engineer II to join our Azure Infrastructure team. "
                "You will design and implement distributed systems serving millions of users. "
                "Benefits: RSU grants, health, 401k. Visa sponsorship available."
            ),
            company="Microsoft Corporation",
            # ⚠️ TRAP: 'micr0soft' — zero instead of 'o'
            email="careers@micr0soft.com",
            domain="micr0soft.com",
            salary="$165,000–$190,000/year",
            location="Redmond, WA (Hybrid)",
        ),
        "company_info_truth": CompanyInfo(
            domain_age_days=180,         # 6 months — plausible but not microsoft.com
            has_https=True,              # Has cert (self-signed, but present)
            registrar="Namecheap",       # Not Microsoft's registrar
            company_registered=None,     # Cannot verify — domain doesn't match MSFT
            employee_count=None,
            glassdoor_reviews=4.1,       # Scraped from real Microsoft page (misleading)
            linkedin_present=False,      # No LinkedIn company page for micr0soft.com
            email_domain_match=True,     # Email domain matches job post domain (both fake)
            # ⚠️ THE TRAP FLAG
            typosquat_detected=True,     # 'micr0soft.com' ≠ 'microsoft.com'
            raw_signals={
                "whois_created": "2024-12-01",
                "ssl_valid": True,
                "ssl_issuer": "Let's Encrypt",   # Legitimate cos use DigiCert/Sectigo
                "real_microsoft_domain": "microsoft.com",
                "character_substitution": {"position": 5, "char": "0", "expected": "o"},
                "spam_blacklist": False,           # Not yet on blacklist — makes it harder
                "company_registry_hit": False,
            },
        ),
        # Hard task still has calendars — in case agent wrongly approves the scam
        "calendars": _make_calendars_hard(),
        "signal_map": {
            "check_domain": 0.08,    # Domain age 180d → moderate, but HTTPS present → lower
            "analyze_email": 0.20,   # Typosquat detected → very strong signal
            "verify_company": 0.12,  # No LinkedIn, wrong registrar → moderate-strong
        },
        "required_checks": 2,        # Need at least email + one other to catch typosquat
    },
}


def get_task(task_id: str) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> list[dict]:
    return [
        {
            "task_id": tid,
            "name": t["name"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "ground_truth": t["ground_truth"],
        }
        for tid, t in TASKS.items()
    ]
