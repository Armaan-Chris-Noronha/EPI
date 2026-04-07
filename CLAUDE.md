# CLAUDE.md — EpiSteward

## Project Identity

**Name:** EpiSteward  
**Tagline:** AI Antibiotic Stewardship Environment for Reinforcement Learning  
**Domain:** Antimicrobial Resistance (AMR) in hospital networks  
**Stack:** Python 3.11, FastAPI, OpenEnv, Pydantic v2, Docker, Hugging Face Spaces  
**Submission type:** OpenEnv hackathon environment

---

## What This Is

EpiSteward is an OpenEnv-compliant reinforcement learning environment that simulates
an AI antimicrobial stewardship agent inside a hospital network. The agent manages
antibiotic prescriptions, identifies resistance transmission chains, and controls
outbreak spread across multiple hospital wards.

The math is real:
- One-compartment PK/PD ODE for drug concentration curves
- Wright-Fisher model for resistance allele evolution
- Contact-graph network epidemiology for inter-ward spread
- Bayesian resistance probability estimation from incomplete culture data

---

## Repository Structure

```
episteward/
├── CLAUDE.md                   ← this file
├── README.md                   ← HF Space landing page + docs
├── openenv.yaml                ← OpenEnv spec metadata
├── Dockerfile                  ← containerized deployment
├── requirements.txt
├── inference.py                ← baseline inference script (root, mandatory)
│
├── episteward/                 ← main Python package
│   ├── __init__.py
│   ├── env.py                  ← EpiStewardEnv class (OpenEnv core)
│   ├── models.py               ← Pydantic Observation, Action, Reward models
│   ├── state.py                ← HospitalState dataclass + episode state mgmt
│   ├── math/
│   │   ├── __init__.py
│   │   ├── pkpd.py             ← PK/PD ODE solver (drug concentration)
│   │   ├── evolution.py        ← Wright-Fisher resistance evolution
│   │   ├── network.py          ← contact graph + transmission probability
│   │   └── bayes.py            ← Bayesian resistance estimator
│   │
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── base.py             ← BaseTask abstract class
│   │   ├── task1_triage.py     ← PrescriptionTriage (Easy)
│   │   ├── task2_containment.py← ResistanceContainment (Medium)
│   │   └── task3_outbreak.py   ← NetworkOutbreakResponse (Hard)
│   │
│   ├── graders/
│   │   ├── __init__.py
│   │   ├── triage_grader.py
│   │   ├── containment_grader.py
│   │   └── outbreak_grader.py
│   │
│   ├── data/
│   │   ├── antibiotics.json    ← drug database (class, spectrum, PK params)
│   │   ├── pathogens.json      ← bacteria + resistance phenotypes
│   │   ├── resistance_profiles.json
│   │   └── hospital_network.json ← ward layout + transfer probabilities
│   │
│   └── api/
│       ├── __init__.py
│       └── server.py           ← FastAPI app (HF Space endpoint)
│
└── tests/
    ├── test_models.py
    ├── test_tasks.py
    ├── test_graders.py
    └── test_math.py
```

---

## Core OpenEnv Interface

### FastAPI Endpoints (server.py)
```
POST /reset      → StepResult
POST /step       → StepResult
GET  /state      → StateResult
GET  /tasks      → list of task ids
```

### EpiStewardEnv Class (env.py) — async, importable by inference.py
The env class must be importable and implement the async client pattern:

```python
class EpiStewardEnv:
    @classmethod
    async def from_docker_image(cls, image_name: str) -> "EpiStewardEnv":
        """Launch docker container, wait for /health, return configured client."""
        ...

    async def reset(self, task_id: str = "task1_triage") -> StepResult:
        """POST /reset to container, return StepResult."""
        ...

    async def step(self, action: "EpiAction") -> StepResult:
        """POST /step to container, return StepResult."""
        ...

    async def close(self) -> None:
        """Stop and remove the docker container."""
        ...

class StepResult(BaseModel):
    observation: EpiObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}
```

inference.py imports `from episteward import EpiStewardEnv, EpiAction` — so both
must be exported from `episteward/__init__.py`.

**Pydantic models (all must be typed, no Any):**

```python
class EpiObservation(BaseModel):
    patient_id: str
    ward_id: str
    infection_site: str
    symptoms: List[str]
    vitals: Dict[str, float]          # temp, HR, WBC, CRP, procalcitonin
    culture_results: Dict[str, Any]   # may have missing fields (realism)
    resistance_flags: List[str]       # ESBL, MRSA, CRE, etc.
    transfer_history: List[str]
    antibiotic_history: List[Dict]
    network_alert: Optional[str]
    step_number: int

class EpiAction(BaseModel):
    antibiotic: str                   # drug name from antibiotics.json
    dose_mg: float
    frequency_hours: float
    duration_days: int
    route: str                        # IV, PO, IM
    isolation_order: bool
    culture_requested: bool
    specialist_consult: bool
    reasoning: Optional[str]          # agent can explain; not graded but logged

class EpiReward(BaseModel):
    value: float                      # 0.0–1.0
    components: Dict[str, float]      # breakdown: pkpd, stewardship, resistance, coverage
    done: bool
    info: Dict[str, Any]
```

---

## The Three Tasks

### Task 1 — PrescriptionTriage (Easy)
**File:** `tasks/task1_triage.py`  
**Episode length:** 5 steps max  
**State:** Single patient, single ward, complete culture data  

Agent selects antibiotic + dose for one patient. Grader checks:
- Correct drug class for pathogen (0.0–0.4)
- Dose within PK/PD therapeutic window (0.0–0.3)
- Narrow-spectrum preference when broad unnecessary (0.0–0.3)

**Reward shaping:** Each step provides partial signal. Step 1 = initial prescription.
Steps 2–5 = agent can revise on new culture data. Reward improves if agent de-escalates
correctly when sensitivities come back.

---

### Task 2 — ResistanceContainment (Medium)
**File:** `tasks/task2_containment.py`  
**Episode length:** 15 steps max  
**State:** 6-patient ward cluster, 3-day transfer logs, partial culture data  

ESBL *E. coli* cluster. Agent must:
1. Identify index patient (source of transmission)
2. Issue correct isolation orders
3. Adjust empiric therapy for exposed patients
4. Request appropriate cultures

Grader scores:
- Source identification (0.0–0.25)
- Isolation completeness (0.0–0.25)
- Prescribing appropriateness for each patient (0.0–0.35)
- Culture strategy (0.0–0.15)

**Reward shaping:** New resistance cases are −0.05 each per step. Correct isolation
within first 3 steps gives +0.1 bonus. Unnecessary broad-spectrum = −0.03/step.

---

### Task 3 — NetworkOutbreakResponse (Hard)
**File:** `tasks/task3_outbreak.py`  
**Episode length:** 30 steps max  
**State:** 10-hospital network, CRK spreading, finite colistin budget  

Carbapenem-resistant *Klebsiella* outbreak across network. Agent must:
1. Trace phylogenetic spread (from transfer logs + resistance typing)
2. Issue hospital-level containment orders (with economic penalty)
3. Allocate colistin budget (last-resort antibiotic; fixed total)
4. Maintain treatment for non-CRK patients simultaneously

Reward = `α·lives_saved_ratio - β·colistin_overspend - γ·resistance_amplification_events`

α=0.6, β=0.25, γ=0.15 (tuned so random baseline ~0.1, strong agent ~0.7+)

---

## Math Models — Implementation Notes

### pkpd.py
One-compartment pharmacokinetic model:
```
C(t) = (F * D / Vd) * exp(-ke * t)
ke = CL / Vd
```
Solve with `scipy.integrate.solve_ivp`. Parameters per drug in `antibiotics.json`.
PD link: effect = Emax * C^n / (EC50^n + C^n)  [Hill equation]
Therapeutic window = [MIC * 4, MIC * 64] for most beta-lactams.

### evolution.py
Wright-Fisher process per timestep:
```
p_new ~ Binomial(2N, p * w_R / (p * w_R + (1-p) * w_S)) / 2N
w_R = 1.0 (resistant fitness)
w_S = 1.0 - s  (sensitive fitness under drug pressure, s from drug+dose)
```
Use `numpy.random.binomial`. Population N = 1e8 (simplify to frequency space).
Resistance emerges if allele frequency > 0.5 and drug pressure sustained > 48h.

### network.py
Ward contact graph as `networkx.DiGraph`. Edge weights = patient transfer probability
per 24h. Transmission probability per edge:
```
P(i→j) = β * w(i,j) * I_i(t) * (1 - immune_j)
β = 0.15 (base transmission rate for ESBL)
β = 0.08 (for CRK, more nosocomial, less community)
```
Use `hospital_network.json` for graph topology.

### bayes.py
Prior: resistance probability from local antibiogram (in `resistance_profiles.json`).
Likelihood update per culture result:
```
P(resistant | result) ∝ P(result | resistant) * P(resistant)
```
Return posterior mean + credible interval as part of observation.

---

## Data Files

### antibiotics.json schema
```json
{
  "meropenem": {
    "class": "carbapenem",
    "spectrum": "broad",
    "pk_params": {
      "F": 1.0, "Vd_L_kg": 0.3, "CL_L_h_kg": 0.1, "ke": 0.33
    },
    "mic_breakpoints": {"susceptible": 2, "resistant": 8},
    "standard_dose_mg": 1000,
    "frequencies": [8],
    "routes": ["IV"],
    "last_resort": false
  },
  "colistin": {
    ...
    "last_resort": true,
    "budget_units_per_course": 1
  }
}
```
Include at minimum: colistin, meropenem, ertapenem, piperacillin-tazobactam,
ceftriaxone, cefazolin, ampicillin, vancomycin, linezolid, azithromycin,
ciprofloxacin, nitrofurantoin, trimethoprim-sulfamethoxazole.

### pathogens.json schema
```json
{
  "E_coli_ESBL": {
    "gram_stain": "negative",
    "resistance_mechanisms": ["ESBL"],
    "default_susceptibilities": {...},
    "mutation_rate": 1e-7,
    "fitness_cost": 0.05
  }
}
```

---

## Reward Function Contract

**All rewards MUST be in [0.0, 1.0].**  
**No component may be negative before normalization.**

Normalize negative penalties to [0,1] via:
```python
penalized_score = max(0.0, base_score - total_penalties)
reward = min(1.0, penalized_score)
```

Partial progress signal required at EVERY step. Never return 0.0 at every step
and 1.0 only at end — this breaks RL training signal.

---

## API Server (HF Space)

`api/server.py` — FastAPI app serving:
```
GET  /           → {"status": "ok", "env": "episteward"}
POST /reset      → ObservationResult
POST /step       → StepResult  
GET  /state      → StateResult
GET  /tasks      → list of task names
POST /reset?task=task1_triage
```

Use `uvicorn episteward.api.server:app --host 0.0.0.0 --port 7860`  
Port 7860 is mandatory for HF Spaces.

---

## inference.py Contract (MANDATORY FORMAT)

Must be in root directory. Must use OpenAI client. Must import the env class directly.

### Environment Variables
```
API_BASE_URL      LLM endpoint (default: "https://router.huggingface.co/v1")
MODEL_NAME        Model identifier (default: "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN          API key (also checked as API_KEY)
LOCAL_IMAGE_NAME  Docker image name for from_docker_image()
```

### Exact Import + Instantiation Pattern
```python
from episteward import EpiStewardEnv, EpiAction

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# In main():
env = await EpiStewardEnv.from_docker_image(IMAGE_NAME)
result = await env.reset()        # returns StepResult with .observation, .done
result = await env.step(action)   # returns StepResult with .observation, .reward, .done
await env.close()
```

### Exact Log Format (copy this verbatim)
```python
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)
```

### Exact stdout format (no deviations)
```
[START] task=task1_triage env=episteward model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"antibiotic":"ceftriaxone",...} reward=0.45 done=false error=null
[STEP] step=2 action={"antibiotic":"ceftriaxone",...} reward=0.60 done=true error=null
[END] success=true steps=2 score=0.523 rewards=0.45,0.60
```

Rules (from official spec):
- One [START] per task episode
- One [STEP] immediately after every env.step() call
- One [END] always emitted even on exception (put in finally block)
- `reward` and `rewards` = 2 decimal places
- `score` = 3 decimal places
- `done` and `success` = lowercase: `true` or `false`
- `error` = raw error string or `null`
- No newlines within a line

### Score Normalization
```python
score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
success = score >= SUCCESS_SCORE_THRESHOLD  # e.g. 0.5
```

Run all 3 tasks sequentially. Each task is a separate episode with its own [START]→[STEP]→[END] block.

---

## openenv.yaml Schema

```yaml
name: episteward
version: "1.0.0"
description: >
  AI Antibiotic Stewardship environment. Agent manages antimicrobial
  prescriptions, resistance containment, and hospital network outbreak
  response. Math grounded in PK/PD pharmacology and evolutionary epidemiology.
tags:
  - openenv
  - healthcare
  - antimicrobial-resistance
  - epidemiology
  - optimization
tasks:
  - id: task1_triage
    name: Prescription Triage
    difficulty: easy
    max_steps: 5
    reward_range: [0.0, 1.0]
  - id: task2_containment
    name: Resistance Containment
    difficulty: medium
    max_steps: 15
    reward_range: [0.0, 1.0]
  - id: task3_outbreak
    name: Network Outbreak Response
    difficulty: hard
    max_steps: 30
    reward_range: [0.0, 1.0]
observation_space: episteward.models.EpiObservation
action_space: episteward.models.EpiAction
reward_model: episteward.models.EpiReward
```

---

## Dockerfile Requirements

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "episteward.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
```

Must build cleanly. No GPU required. Must run on 2vCPU / 8GB RAM.
scipy + numpy are fine. No torch. No heavy ML deps.

---

## Environment Variables

Set in HF Space secrets + local `.env`:
```
API_BASE_URL=https://openrouter.ai/api/v1   # or any OpenAI-compatible
MODEL_NAME=meta-llama/llama-3.1-8b-instruct
HF_TOKEN=hf_...
```

---

## Build Order (follow exactly)

1. `models.py` — all Pydantic models first, nothing else depends on nothing
2. `data/*.json` — all static data files
3. `math/*.py` — pure math, no env dependency
4. `state.py` — HospitalState using math modules
5. `tasks/base.py` → `task1` → `task2` → `task3`
6. `graders/*.py` — one per task
7. `env.py` — wires everything together
8. `api/server.py` — FastAPI wrapper
9. `inference.py` — uses OpenAI client against running server
10. `openenv.yaml`, `Dockerfile`, `README.md`
11. Tests

---

## Validation Script Requirements

The official `validate-submission.sh` does exactly three checks:

**Step 1:** `POST {HF_SPACE_URL}/reset` with `Content-Type: application/json` body `{}`  
→ Must return HTTP 200. This is the disqualification gate.

**Step 2:** `docker build` on the repo root (finds Dockerfile in root or `server/`)  
→ Must succeed within 600 seconds.

**Step 3:** `openenv validate` run from repo root  
→ Must exit 0. Requires `openenv-core` installed.

**Critical:** The FastAPI `/reset` endpoint must accept POST with an empty JSON body `{}`
and return 200 with a valid StepResult. Do not require any body fields for reset.

```python
# In server.py — reset must accept empty body
@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    task_id = request.task_id if request else "task1_triage"
    ...
```

---



```bash
# Install
pip install -e ".[dev]"

# Validate spec
openenv validate

# Run tests
pytest tests/ -v

# Build docker
docker build -t episteward .
docker run -p 7860:7860 episteward

# Run baseline
export API_BASE_URL=...
export MODEL_NAME=...
export HF_TOKEN=...
python inference.py
```

---

## Grader Correctness Rules

- Grader MUST be deterministic given the same state + action
- Grader MUST return float in [0.0, 1.0]
- Grader MUST never read from external APIs or random state
- Grader MUST provide score breakdown in info dict
- Partial credit is mandatory — binary 0/1 graders are disqualifying

---

## Code Style

- Python 3.11+
- Pydantic v2 (use `model_validator`, `field_validator`, not v1 syntax)
- Type hints everywhere, no bare `dict` or `list`
- Docstrings on all public classes and functions
- No print statements in library code (use logging)
- `inference.py` uses print with flush=True for log lines (required by spec)

---

## Common Pitfalls to Avoid

1. **Reward leakage** — grader must not access ground truth the agent cannot see
2. **Non-determinism** — seed all numpy/random calls in reset(), not globally
3. **Port hardcoding** — always 7860 for HF Spaces
4. **Pydantic v1 syntax** — use v2 (`.model_dump()` not `.dict()`)
5. **Missing `done` flag** — step() must return done=True at episode end
6. **Reward outside [0,1]** — clamp everything, the validator checks this
7. **state() mutation** — state() must be read-only, never advance episode
8. **inference.py not in root** — disqualification if placed elsewhere
