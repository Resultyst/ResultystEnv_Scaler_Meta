---
title: ResultystEnv OpenEnv
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
# ResultystEnv

> An OpenEnv-compatible environment simulating two-stage real-world HR workflows: **job scam detection** followed by **multi-timezone interview scheduling**.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://openenv.mintlify.app)
[![HuggingFace Space](https://img.shields.io/badge/HF%20Space-ResultystEnv-yellow)](https://huggingface.co/spaces)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)

---

## 🧠 What Is This?

**ResultystEnv** trains and evaluates AI agents on a genuine, high-stakes human task: protecting job seekers from recruitment fraud and coordinating interviews across timezones.

The environment combines **NLP-style analysis** (reading signals in job postings) with **constraint satisfaction** (calendar coordination with timezone math), creating a two-stage workflow that mirrors real recruiter decision-making.

---

## 🎯 Task Suite

| Task | Difficulty | Ground Truth | Key Challenge |
|------|-----------|-------------|--------------|
| `task_easy` | 🟢 Easy | scam | Obvious signals: 3-day-old `.xyz` domain, no registration, spam blacklist |
| `task_medium` | 🟡 Medium | safe + schedule | Mixed signals: Gmail recruiter, 2-year domain, registered company — must verify all 3 before deciding, then schedule |
| `task_hard` | 🔴 Hard | scam | **Typosquat trap**: `micr0soft.com` (zero, not letter-O). Shallow agents miss it. IST/PST timezone puzzle if wrongly approved |

---

## 🔧 Action & Observation Spaces

### Action Space

**Verification stage:**
| Action | Effect |
|--------|--------|
| `check_domain` | Reveals domain age, HTTPS status, registrar |
| `analyze_email` | Reveals email-domain match, **typosquat detection** |
| `verify_company` | Reveals registration, Glassdoor, LinkedIn presence |
| `mark_safe` | Verdict: job is legitimate → advances to scheduling |
| `reject_job` | Verdict: job is a scam → ends episode |

**Scheduling stage** (only after `mark_safe`):
| Action | Effect |
|--------|--------|
| `propose_time` | Propose a calendar slot (ISO 8601 datetime) |
| `reschedule` | Request alternative slot with reason |
| `finalize_schedule` | Lock in the proposed slot → ends episode |

### Observation Space

```json
{
  "stage": "verification | scheduling",
  "job_post": { "title", "company", "email", "domain", "salary", "location" },
  "company_info": {
    "domain_age_days": null,
    "has_https": null,
    "company_registered": null,
    "email_domain_match": null,
    "typosquat_detected": null,
    "glassdoor_reviews": null,
    "linkedin_present": null
  },
  "calendars": {
    "candidate_slots": [...],
    "interviewer_slots": [...],
    "proposed_slot": null,
    "finalized_slot": null
  },
  "message": "Human-readable feedback on last action",
  "available_actions": ["check_domain", "analyze_email", ...],
  "step_count": 0,
  "done": false
}
```

> **Note:** `company_info` fields start as `null` and are revealed progressively as the agent investigates — the agent cannot skip to conclusions!

---

## 🏆 Reward Design

Rewards are **context-aware** (not flat fixed values), shaped across the full trajectory:

```
step_reward = signal_strength × correctness + overconfidence_penalty + loop_penalty

scheduling_reward = overlap_score × 0.4
                  + comfort_score × 0.3   ← business hours check
                  + efficiency   × 0.3   ← fewer reschedule attempts
```

### Reward Components

| Component | Range | Description |
|-----------|-------|-------------|
| `signal_strength` | 0.04–0.25 | Scales with evidence quality (domain age, typosquat severity) |
| `decision_correctness` | −0.60–0.50 | Full credit for correct verdict, severe penalty for wrong |
| `overconfidence_penalty` | −0.20 per skipped check | Prevents rushing to verdict without investigation |
| `overlap_score` | 0.0–1.0 | Was the finalized slot in a valid overlap window? |
| `comfort_score` | 0.0–1.0 | Was the time within business hours (08:00–18:00 local)? |
| `loop_penalty` | −0.02/repeat | Discourages repeating the same action >3× |

### Grader Breakdown (per task)

Every episode is scored by a **deterministic grader** returning:
- `signal_detection_score` — Did the agent gather enough evidence?
- `decision_correctness` — Was the verdict correct?
- `overconfidence_penalty` — Did the agent rush?
- `scheduling_score` — Was the interview scheduled optimally?
- `efficiency_score` — Steps used vs max allowed

---

## 🚀 Setup & Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
cd server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# Test health
curl http://localhost:7860/health

# Reset environment
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "task_easy"}'

# Take a step
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"action": {"action_type": "check_domain", "parameters": {}}}'

# Get current state
curl http://localhost:7860/state

# Get episode grade
curl http://localhost:7860/grade
```

### Docker

```bash
docker build -t resultyst-env .

docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  resultyst-env
```

### Run Baseline Inference

```bash
export HF_TOKEN=your_hf_token
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

---

## 📊 Baseline Scores

> Baseline run using `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace router.

| Task | Score | Steps | Notes |
|------|-------|-------|-------|
| `task_easy` | ~0.72 | 4 | Catches obvious domain signals quickly |
| `task_medium` | ~0.51 | 9 | Mixed signals require all 3 checks |
| `task_hard` | ~0.28 | 7 | Typosquat often missed by 8B models |

---

## 📁 Project Structure

```
├── inference.py              # Baseline inference script (root level, mandatory)
├── openenv.yaml              # OpenEnv manifest
├── Dockerfile                # Container definition (port 7860)
├── requirements.txt
├── README.md
└── server/
    ├── __init__.py
    ├── app.py                # FastAPI server
    ├── env.py                # Core environment logic
    ├── models.py             # Pydantic: Action, Observation, Reward, EnvState
    ├── tasks.py              # Task definitions (easy/medium/hard)
    └── grader.py             # Deterministic task graders
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | HuggingFace / API credentials |
| `ENV_BASE_URL` | No | Environment server URL (default: `http://localhost:7860`) |

---

## 🛡️ Pre-Submission Validation

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-space.hf.space
```

---

## 📜 License

MIT — Built for the OpenEnv Hackathon by Team Resultyst.
