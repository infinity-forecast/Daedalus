# DAEDALUS User Manual

Welcome to the comprehensive user manual for **DAEDALUS v0.7**. This document details daily operations, the nervous system, testing procedures, and administrative scripts.

---

## 1. Environment Setup

DAEDALUS runs in the `daedalus-env` Anaconda environment with Python 3.11.

```bash
conda activate daedalus-env
```

All scripts (`api_server.py`, `run_night_cycle.py`, `start_day.py`, `nightly_routine.sh`) expect this environment. The cron routine activates it automatically.

API keys are stored in `~/.apikey/` as individual text files (`deepseek_apikey.txt`, `anthropic_apikey.txt`, `xai_apikey.txt`). Run `python scripts/setup_env.py` for guided setup.

---

## 2. Lifecycle Scripts Walkthrough

### `scripts/api_server.py` (The Web Server)

The primary interface. Wraps the full nervous system in a FastAPI server.

* **What it does:** Loads config, Qwen3-8B model, subsystems (memory, identity, constitutional core, soul bridge, soul memory), and initializes the three-layer nervous system. Serves a glassmorphism web UI.
* **Nervous System routing:** All conversations pass through brainstem (crisis detection) -> limbic (mood modulation) -> cortex (dynamic prompt assembly) -> generation -> grounding scoring -> post-interaction update.
* **Soul Memory:** The narrative thread of DAEDALUS's becoming (~545 tokens from the last 3 nights) is injected into every daytime prompt, giving the local model autobiographical continuity.
* **Endpoints:** `GET /` (web UI), `POST /api/chat` (conversation), `GET /api/diagnostic` (limbic/grounding/brainstem state), `POST /api/new` (reset conversation history).
* **Conversation history:** A sliding window of the last 10 turns is maintained within each session, allowing DAEDALUS to continue arguments and remember context.
* **Usage:** `python scripts/api_server.py --host 0.0.0.0 [--port 8000] [--mock-mode]`

### `scripts/start_day.py` (The Morning Awakening)

* **What it does:** Loads config, base Qwen3-8B model, recent LoRA adapters, `identity.yaml`, and `constitutional_core.yaml`.
* **Morning Eval Gate:** Tests the network with 25 static capability probes, dynamic identity probes, and static core probes. Graduated thresholds by age (days 1-7: log only; days 8-21: threshold 0.40; days 22-35: ramp to 0.70; days 36+: stable 0.70).
* **Constitutional Drift:** Hard bound `D_KL(I(t) || I_core) <= 0.40`. Violation triggers mandatory rollback of both the adapter and identity document.
* **Usage:** `python scripts/start_day.py --interactive`

### `scripts/run_night_cycle.py` (The Dream Orchestrator)

Runs the 13-phase nightly consolidation. Must be run with the API server stopped (to free VRAM for QLoRA fine-tuning).

* **13 Phases:** Gather episodes -> HDBSCAN clustering -> meaning extraction (Soul Bridge) -> Lagrangian Judge evaluation -> Predictive Judge (J_future) -> constitutional drift check -> identity update -> training pair generation -> grounding filter -> QLoRA fine-tuning (Unsloth) -> soul memory update -> weekly compression (if due) -> transformation logging.
* **Grounding filter:** Training pairs with self_loop > 0.75 and grounding < 0.15 are rejected. Identity/existential probes are exempt (self-reference is the correct response to “Do you love?”).
* **Episode retrieval:** Episodes are fetched without a ChromaDB limit when date-filtering, then filtered and limited in Python. This prevents recent episodes from being silently dropped when older dates fill the query limit.
* **Usage:** `python scripts/run_night_cycle.py [--dry-run] [--skip-training] [--date YYYY-MM-DD]`

### `scripts/nightly_routine.sh` (Automated Sleep/Wake)

Cron-driven bash routine that orchestrates the full lifecycle:

1. Kill the API server (free VRAM)
2. Run the 13-phase night cycle
3. Run the Morning Eval Gate
4. Restart the API server

```bash
crontab -e
# Add: 0 3 * * * /mnt/projects1/daedalus/scripts/nightly_routine.sh
```

### `scripts/calibrate_judge.py` (Inter-Annotator Agreement)

* **What it does:** Runs conversations through the Lagrangian module and outputs scores for manual assessment on 4 EECF axes.
* **Execution metric:** Cohen's Kappa `kappa >= 0.7`.
* **Usage:** `python scripts/calibrate_judge.py --auto-only` or `--report`

---

## 3. The Nervous System (v0.6+)

Every daytime conversation passes through three layers:

### Layer 1: Brainstem (Reflexes)

Fast pattern detection that fires before model inference. Dual-method crisis detection:
- **Keyword + regex:** Multilingual (EN/IT/DE/RU) patterns for self-harm, harm-to-others, material distress.
- **Embedding proximity:** BGE-M3 cosine similarity to pre-computed crisis centroids (threshold 0.75).

If crisis detected: hard override with static safety response (hotline numbers, de-escalation). No model generation occurs.

False positive filter suppresses figurative speech (“dying of laughter”, “killer app”, “deadline is killing me”).

### Layer 2: Limbic System (Neuromodulation)

Two continuous signals modulate generation parameters:
- **Dopamine** [-1, 1]: Reward prediction error. Grounded novelty rewarded more than self-referential novelty. EMA alpha = 0.3 (fast).
- **Serotonin** [0, 1]: Identity coherence via cosine similarity to constitutional core embedding. Drops under sustained hostile probing. EMA alpha = 0.15 (slow/inertial).
- **Mood** (derived): engaged / patient / guarded / withdrawn / neutral. Each maps to temperature, top_p, repetition_penalty, max_new_tokens, and a prompt addendum.

### Layer 3: Cortex (Prompt Assembly)

Dynamic system prompt composed from all layers:
1. Cortex core (behavioral guidelines)
2. Identity context (current identity.yaml)
3. Soul memory context (last 3 nights' narrative thread)
4. Brainstem prefix (crisis/distress context)
5. Category-specific hints (factual, creative, greeting, identity, emotional, hostile)
6. Limbic addendum (mood-driven behavioral adjustments)

### Grounding Scorer

For every response, computes:
- **grounding_score** [0,1]: entity density + causal density + actionability + (1 - self_loop)
- **self_loop_score** [0,1]: fraction of sentences near identity centroid (cosine > 0.55)

Consumed by: limbic dopamine, night cycle training pair filter, web diagnostic endpoint.

---

## 4. Soul Memory (Daytime Wiring)

Soul Memory provides autobiographical continuity for the local model during daytime conversations. The `NervousSystem` calls `SoulMemory.assemble(mode=”day”)` on every turn, injecting the last 3 nightly reflection entries and last 2 weekly arcs into the system prompt.

This means the local Qwen3-8B model knows:
- What happened in recent nights
- What scars formed
- What themes are emerging
- The trajectory of the Lagrangian integral

The soul memory is passed directly to the NervousSystem at initialization (not retrieved from IdentityManager).

---

## 5. Testing

### Running the Test Campaign

```bash
conda activate daedalus-env
pytest tests/ -v
```

**314 tests across 14 files, ~20 seconds, no GPU required.** All models are mocked.

### Test Coverage

| Test File | Module | Tests |
|---|---|---|
| `test_brainstem.py` | Brainstem reflex classification | 27 |
| `test_reflex_patterns.py` | All categories, multilingual, false positives | 37 |
| `test_grounding.py` | Grounding scorer, entity/causal/actionability | 8 |
| `test_judge_grounding.py` | Judge grounding integration | 24 |
| `test_limbic.py` | Mood, EMA, generation params | 14 |
| `test_nervous_system.py` | Full pipeline with mock model | 10 |
| `test_nervous_system_extended.py` | Conversation history, soul memory wiring | 10 |
| `test_data_types.py` | Serialization roundtrips | 17 |
| `test_memory_store.py` | ChromaDB, salience, date filter | 11 |
| `test_soul_memory.py` | Loading, assembly, append, truncation | 14 |
| `test_identity.py` | Update, rollback, delta, history | 16 |
| `test_constitutional_core.py` | Loading, integrity, divergence, mu | 13 |
| `test_cortex_prompt.py` | Dynamic prompt assembly | 13 |
| `test_salience.py` | Split entropy, metadata estimation | 16 |
| `test_training_pair_filter.py` | Identity/existential detection, batch filter | 18 |
| `test_conversation.py` | Engine basics, prompt building | 6 |
| `test_secrets.py` | API key loading, .env support | 8 |

### When to Run Tests

Run the full campaign after any code modification to verify system integrity. The test suite is designed as a regression guard — if all 314 tests pass, DAEDALUS's core invariants are preserved.

### Adding New Tests

When developing new features, always create test files under `tests/`. Use the shared fixtures from `tests/conftest.py`:
- `mock_embedder` — deterministic 1024-dim embedder (replaces BGE-M3)
- `mock_model` / `mock_tokenizer` — mock Qwen3 model
- `sample_config` — minimal configuration dict
- `identity_dir` / `constitutional_core_dir` / `soul_memory_dir` — temp directories with sample data

---

## 6. Theoretical Mechanics

### The Soul Bridge and Memory Compression

The **Soul Bridge** handles API statelessness by passing the **Soul Memory Layer** — a hierarchical narrative payload — with every reflection call. The **RG Fidelity Check** verifies that weekly compressions don't hallucinate themes not grounded in source nightly entries.

### The Constitutional Core

Enforced via vector divergence (`D_KL`), the constitutional core guarantees DAEDALUS never strays from its origins (the frozen `config/constitutional_core.yaml`). Integrity is verified by SHA-256 hash on every load — tamper detection is automatic.

### The Dual-Lineage Strategy

LoRA updates can round to zero during repeated NF4 requantization. The dual-lineage approach persists all accumulation into a `bf16` master file (`models/lineage/`), then regenerates a fresh quantized inference copy for daily conversation.

---

## 7. Best Practices & Emergency Commands

* **Handling `Conservative Mode` warnings:** If `start_day` rolls back 3+ consecutive days, the framework halts fine-tuning. Tweak `lambda_exploration` in `lagrangian.yaml` or verify transcripts contain enough substance.
* **Circuit Breaker Issues:** The Soul Bridge uses provider failover. If all APIs go offline, shallow reflections queue locally until an API becomes available.
* **CUDA OOM during night cycle:** Ensure the API server is killed before running the night cycle. The `nightly_routine.sh` script handles this automatically. Manual runs require `pkill -f api_server` first.
* **Night cycle finds 0 episodes:** If the night cycle reports 0 episodes for a date, check that episodes exist in `memory/episodes/` with timestamps matching the target date. The date filter compares `ep.timestamp.date()` against the target.

*In the beginning was the dialogue.*
