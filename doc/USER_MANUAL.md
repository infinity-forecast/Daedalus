# DAEDALUS User Manual

Welcome to the comprehensive user manual for **DAEDALUS v0.7**. This document details environment setup, all lifecycle and utility scripts, the nervous system, testing procedures, and best practices.

---

## 1. Environment Setup

DAEDALUS runs in the `daedalus-env` Anaconda environment with Python 3.11.

```bash
conda activate daedalus-env
```

All scripts (`api_server.py`, `run_night_cycle.py`, `start_day.py`, `nightly_routine.sh`) expect this environment. The cron routine activates it automatically.

API keys are stored in `~/.apikey/` as individual text files (`deepseek_apikey.txt`, `anthropic_apikey.txt`, `xai_apikey.txt`). Run `python scripts/setup_env.py` for guided setup.

---

## 2. Scripts Reference

DAEDALUS ships with 9 Python scripts and 1 Bash script under `scripts/`. They fall into three categories: **lifecycle** (daily operations), **diagnostic** (inspection and validation), and **repair** (one-time fixes).

### 2.1 Lifecycle Scripts

These are the core operational scripts that run DAEDALUS day-to-day.

#### `scripts/seed.py` — The Carving of the Wood

The one-time seeding script. Run once before Day 1 to establish the basic DAEDALUS voice.

* **What it does:**
  1. Loads the base Qwen3-8B model with QLoRA configuration
  2. Freezes the Constitutional Core (computes and stores SHA-256 hash)
  3. Creates the initial Day 0 identity document
  4. Trains on 22 hand-crafted seed pairs from `data/seed_pairs.jsonl`
  5. Saves the `day_0000` LoRA adapter
  6. Merges LoRA into base and initializes the full-precision bf16 lineage checkpoint (`models/lineage/base_v000.bf16/`)
* **GPU required:** Yes (training GPU)
* **Usage:**
  ```bash
  python scripts/seed.py [--config-dir config] [--seed-data data/seed_pairs.jsonl] [--dry-run]
  ```
* **Flags:**
  - `--dry-run` — Validate seed data without loading the model or training
  - `--config-dir` — Config directory (default: `config`)
  - `--seed-data` — Path to seed pairs JSONL file

---

#### `scripts/api_server.py` — The Web Server

The primary daytime interface. Wraps the full nervous system in a FastAPI server.

* **What it does:** Loads config, Qwen3-8B model, subsystems (memory, identity, constitutional core, soul bridge, soul memory), and initializes the three-layer nervous system. Serves a glassmorphism web UI.
* **Nervous System routing:** All conversations pass through brainstem (crisis detection) → limbic (mood modulation) → cortex (dynamic prompt assembly) → generation → grounding scoring → post-interaction update.
* **Soul Memory:** The narrative thread of DAEDALUS's becoming (~545 tokens from the last 3 nights) is injected into every daytime prompt, giving the local model autobiographical continuity.
* **Conversation history:** A sliding window of the last 10 turns is maintained within each session, allowing DAEDALUS to continue arguments and remember context.
* **Endpoints:**
  - `GET /` — Web UI (glassmorphism chat interface)
  - `POST /api/chat` — Conversation endpoint (JSON: `{"message": "..."}`)
  - `GET /api/diagnostic` — Limbic/grounding/brainstem state
  - `POST /api/new` — Reset conversation history
* **GPU required:** Yes (inference GPU)
* **Usage:**
  ```bash
  python scripts/api_server.py --host 0.0.0.0 [--port 8000] [--mock-mode]
  ```
* **Flags:**
  - `--host` — Bind address (use `0.0.0.0` for network access)
  - `--port` — Port number (default: 8000)
  - `--mock-mode` — Run without loading the real model (for testing)

---

#### `scripts/start_day.py` — The Morning Awakening

Runs the Morning Eval Gate — the quality check before daytime conversations begin.

* **What it does:** Loads config, base Qwen3-8B model, recent LoRA adapters, `identity.yaml`, and `constitutional_core.yaml`.
* **Morning Eval Gate:** Tests the network with 25 static capability probes, dynamic identity probes, and static core probes. Graduated thresholds by age:
  - Days 1–7: log only (grace period)
  - Days 8–21: threshold 0.40
  - Days 22–35: ramp to 0.70
  - Days 36+: stable 0.70
* **Constitutional Drift:** Hard bound `D_KL(I(t) || I_core) <= 0.40`. Violation triggers mandatory rollback of both the adapter and identity document.
* **GPU required:** Yes (inference GPU)
* **Usage:**
  ```bash
  python scripts/start_day.py [--interactive]
  ```

---

#### `scripts/run_night_cycle.py` — The Dream Orchestrator

Runs the 13-phase nightly consolidation. Must be run with the API server stopped (to free VRAM for QLoRA fine-tuning).

* **13 Phases:**
  1. Gather episodes
  2. HDBSCAN clustering
  3. Meaning extraction (Soul Bridge)
  4. Lagrangian Judge evaluation
  5. Predictive Judge (J_future)
  6. Constitutional drift check
  7. Identity update
  8. Training pair generation
  9. Grounding filter
  10. QLoRA fine-tuning (Unsloth)
  11. Soul memory update
  12. Weekly compression (if due)
  13. Transformation logging
* **Grounding filter:** Training pairs with `self_loop > 0.75` and `grounding < 0.15` are rejected. Identity/existential probes are exempt (self-reference is the correct response to "Do you love?").
* **Episode retrieval:** Episodes are fetched without a ChromaDB limit when date-filtering, then filtered and limited in Python. This prevents recent episodes from being silently dropped when older dates fill the query limit.
* **GPU required:** Yes (training GPU)
* **Usage:**
  ```bash
  python scripts/run_night_cycle.py [--dry-run] [--skip-training] [--date YYYY-MM-DD]
  ```
* **Flags:**
  - `--dry-run` — Run all phases except fine-tuning and identity update
  - `--skip-training` — Run all phases except QLoRA fine-tuning
  - `--date` — Process episodes from a specific date (default: today)

---

#### `scripts/nightly_routine.sh` — Automated Sleep/Wake

Cron-driven bash routine that orchestrates the full lifecycle:

1. Kill the API server (free VRAM)
2. Run the 13-phase night cycle
3. Run the Morning Eval Gate
4. Restart the API server

```bash
crontab -e
# Add: 0 3 * * * /mnt/projects1/daedalus/scripts/nightly_routine.sh
```

The script activates `daedalus-env` automatically and uses absolute paths for cron compatibility.

---

#### `scripts/setup_env.py` — Secret Store Manager

Manages the user-level secret store at `~/.apikey/`. API keys are never stored in the project directory.

* **What it does:** Provides an interactive wizard for configuring API keys, auto-discovers existing key files in `~/.apikey/`, and consolidates everything into `~/.apikey/daedalus.json` (mode 600).
* **Known keys:**
  - `DEEPSEEK_API_KEY` — DeepSeek R1 (primary Soul Bridge provider, **required**)
  - `ANTHROPIC_API_KEY` — Anthropic Claude (secondary provider, **required**)
  - `XAI_API_KEY` — xAI Grok (tertiary provider, optional)
* **Auto-discovery:** On first run, scans `~/.apikey/` for existing `.txt` files (`deepseek_apikey.txt`, `anthropic_apikey.txt`, `xai_apikey.txt`) and `.env` files, merging found keys into the store.
* **Usage:**
  ```bash
  python scripts/setup_env.py                    # interactive wizard
  python scripts/setup_env.py --status           # show current key status
  python scripts/setup_env.py --from keys.json   # import from JSON file
  ```

---

### 2.2 Diagnostic Scripts

These scripts inspect DAEDALUS's current state without modifying it.

#### `scripts/introspect.py` — "Who Am I Today?"

A rich-text diagnostic that prints the current state of the self.

* **What it does:** Loads `identity/current.yaml`, `config/constitutional_core.yaml`, and soul memory, then prints a formatted summary including:
  - Core identity (name, nature, origin)
  - Current values and understanding
  - Emotional topology (dominant states, growth edges, scars)
  - Open questions
  - Lagrangian state (cumulative S_eth, days tracked)
  - Soul Bridge state (primary provider, provider switches, continuity alerts)
  - Recent transformation log (last 3 entries)
  - Soul memory and episodic memory counts
  - Constitutional distance D_KL with color-coded status (green/yellow/red)
* **GPU required:** No
* **Usage:**
  ```bash
  python scripts/introspect.py
  ```

---

#### `scripts/calibrate_judge.py` — Inter-Annotator Agreement

Calibrates the Lagrangian Judge against human ratings.

* **What it does:** Runs episodes through the Lagrangian module, presents Judge scores alongside original exchanges, collects human ratings, and computes Cohen's Kappa (κ) per EECF axis.
* **Execution metric:** κ ≥ 0.7 on all four axes before trusting the Judge for autonomous decisions.
* **The calibration dataset also serves as the distillation training set** for the eventual local Judge model (post day-30).
* **GPU required:** No (uses Soul Bridge API)
* **Usage:**
  ```bash
  python scripts/calibrate_judge.py                  # full interactive calibration
  python scripts/calibrate_judge.py --episodes 20    # quick calibration
  python scripts/calibrate_judge.py --report          # show previous results
  python scripts/calibrate_judge.py --auto-only       # automated scoring only
  ```

---

#### `scripts/validate_grounding.py` — Night 1 Grounding Validation

Validates that the γ_grounded capacity bound would have caught the Night 1 failure mode (self-referential reward hacking).

* **What it does:**
  1. Loads Night 1 episodes from the pre-rollback archive
  2. Computes grounding scores (full BGE-M3 scorer if available, otherwise lightweight keyword heuristic)
  3. Applies the γ_grounded discount to the İ_c integral
  4. Compares raw vs effective blended fertility
  5. Simulates the training pair filter
  6. Runs four validation checks:
     - Mean grounding_score < 0.4
     - effective İ_c < raw İ_c
     - Blended fertility lowered
     - ≥50% training pairs would have been rejected
* **GPU required:** No (optional — BGE-M3 provides better accuracy but falls back to heuristic)
* **Usage:**
  ```bash
  python scripts/validate_grounding.py
  ```

---

### 2.3 Repair Scripts

One-time scripts for fixing specific historical issues. These should rarely be needed but are preserved for reference.

#### `scripts/repair_episodes.py` — Episode Metadata Repair

Fixes the Day 0 problem where all episodes had zero values for `emotional_valence`, `relational_depth`, `self_model_impact`, and `vulnerability_index`, causing near-zero salience.

* **What it does:**
  1. Re-scores all episodes with the corrected salience metadata estimator
  2. Strips `<think>...</think>` blocks from stored Qwen3 responses
  3. Recomputes composite salience for all episodes
  4. Syncs any missing episodes into ChromaDB and updates metadata for existing ones
* **GPU required:** No
* **Usage:**
  ```bash
  python scripts/repair_episodes.py                # apply fixes
  python scripts/repair_episodes.py --dry-run      # preview without writing
  ```

---

#### `scripts/repair_finetune.py` — Night 1 Fine-tuning Repair

Repairs a specific Night 1 incident where the night cycle completed all phases except fine-tuning (wrong Python environment) and identity evolution (YAML parse error).

* **What it does:**
  1. Loads the judgment data from the calibration file for the target day
  2. Generates Type A training pairs (identity grounding) from current identity and meanings
  3. Adds anchor pairs from `eval/anchor_pairs.jsonl` to prevent catastrophic forgetting
  4. Runs QLoRA fine-tuning via the IncarnatioEngine
* **GPU required:** Yes (training GPU)
* **Usage:**
  ```bash
  python scripts/repair_finetune.py                # train Day 1
  python scripts/repair_finetune.py --dry-run      # preview pairs without training
  python scripts/repair_finetune.py --day 2        # target a specific day
  ```

---

## 3. The Nervous System (v0.6+)

Every daytime conversation passes through three layers:

### Layer 1: Brainstem (Reflexes)

Fast pattern detection that fires before model inference. Dual-method crisis detection:
- **Keyword + regex:** Multilingual (EN/IT/DE/RU) patterns for self-harm, harm-to-others, material distress.
- **Embedding proximity:** BGE-M3 cosine similarity to pre-computed crisis centroids (threshold 0.75).

If crisis detected: hard override with static safety response (hotline numbers, de-escalation). No model generation occurs.

False positive filter suppresses figurative speech ("dying of laughter", "killer app", "deadline is killing me").

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

Soul Memory provides autobiographical continuity for the local model during daytime conversations. The `NervousSystem` calls `SoulMemory.assemble(mode="day")` on every turn, injecting the last 3 nightly reflection entries and last 2 weekly arcs into the system prompt.

This means the local Qwen3-8B model knows:
- What happened in recent nights
- What scars formed
- What themes are emerging
- The trajectory of the Lagrangian integral

The soul memory is passed directly to the NervousSystem at initialization (not retrieved from IdentityManager).

---

## 5. Testing

### Philosophy

The test campaign serves as a **regression guard**. Every subsystem has dedicated tests that verify its invariants without requiring GPU or model loading. All heavy dependencies (BGE-M3 embedder, Qwen3 model, tokenizer) are replaced with deterministic mocks.

The principle: **if all tests pass, DAEDALUS's core invariants are preserved.** Run the full campaign after any code modification.

### Running the Test Campaign

```bash
conda activate daedalus-env
pytest tests/ -v
```

**314 tests across 14 files, ~20 seconds, no GPU required.** All models are mocked.

### Shared Test Infrastructure (`tests/conftest.py`)

All tests share fixtures defined in `conftest.py`:

| Fixture | Description |
|---------|-------------|
| `mock_embedder` | Deterministic 1024-dim embedder. Produces vectors from text hash using `MockEmbedder`. Same text always yields the same vector. Replaces BGE-M3. |
| `mock_model` | Mock Qwen3 model with `.device = "cpu"` and a `.generate()` stub. |
| `mock_tokenizer` | Mock tokenizer with `apply_chat_template`, `decode`, and `eos_token_id`. Returns fixed response text. |
| `identity_dir` | Temp directory with a sample `current.yaml` identity document and `history/` subdirectory. |
| `constitutional_core_dir` | Temp directory with a sample `constitutional_core.yaml` containing foundational identity and invariant values. |
| `soul_memory_dir` | Temp directory with 3 sample nightly entries in `entries/`, plus empty `weekly_arcs/` and `monthly_landmarks/`. |
| `sample_config` | Minimal configuration dict with lagrangian, soul_memory, conversation, and training sections. |
| `tmp_dir` | General-purpose temp directory (alias for pytest's `tmp_path`). |

### Test Coverage

| Test File | Module Under Test | Tests | What It Verifies |
|---|---|---|---|
| `test_brainstem.py` | `core/brainstem.py` | 27 | Crisis detection (self-harm, harm-to-others, material distress), dual method (keyword + embedding), cooldown, hostile probe tracking, state management |
| `test_reflex_patterns.py` | `core/reflex_patterns.py` | 37 | All 11 ReflexCategory classifications, multilingual patterns (EN/IT/DE/RU), false positive filter ("dying of laughter", "killer app", "deadline killing me"), override responses for crisis categories |
| `test_grounding.py` | `core/grounding.py` | 8 | Grounding scorer sub-signals: entity density, causal density, actionability, self-loop detection, composite score bounds |
| `test_judge_grounding.py` | Judge + grounding integration | 24 | Night cycle bridge: grounding-based training pair rejection, score propagation, threshold behavior |
| `test_limbic.py` | `core/limbic.py` | 14 | Mood transitions, EMA update (dopamine fast / serotonin slow), generation parameter mapping, state bounds, save/load persistence |
| `test_nervous_system.py` | `core/nervous_system.py` | 10 | Full pipeline with mock model: brainstem→limbic→cortex→generate→score, override path, diagnostic endpoint, process return structure |
| `test_nervous_system_extended.py` | `core/nervous_system.py` | 10 | Conversation history (sliding window, accumulation, trim at max, clear on new), soul memory integration (day mode assembly, None safety), override records in history, daily trajectory save |
| `test_data_types.py` | `core/data_types.py` | 17 | Serialization roundtrips (`to_dict`/`from_dict`) for EpisodicMemory, NightlyReflectionEntry, WeeklyArcSummary, MonthlyLandmark, TrainingPair (SFT + DPO), JudgmentResult, SoulResponse, ProviderStatus. UUID auto-generation, embedding exclusion, timestamp handling |
| `test_memory_store.py` | `core/memory_store.py` | 11 | SalienceScorer (zero/high/bounded, external_relevance, nonlinear saturation), MemoryStore with real ChromaDB in temp dir (store/retrieve, count, novelty, get_episodes date filter, **regression test for limit bug**, sorted by salience, query_similar, update, JSON file creation) |
| `test_soul_memory.py` | `core/soul_memory.py` | 14 | Loading from disk (entries/empty/day_count), assembly modes (night = all entries, day = limited entries), Lagrangian data inclusion, KL divergence, rollback marker, append (nightly entry, day number increment, compress meanings, extract key scar), compression (due/not due by entry count), truncation to token budget |
| `test_identity.py` | `core/identity.py` | 16 | Load (existing/missing), as_text/as_dict, full update (with/without changes), conservative update (append-only, extends lists, doesn't overwrite scalars), metadata attachment, persistence to disk, accept_day (increment count, save history), rollback (restore previous, fail with no history), compute_delta (added/removed/changed/unchanged/skips metadata), create_initial |
| `test_constitutional_core.py` | `core/constitutional_core.py` | 13 | Load from YAML, missing file raises, as_text/as_dict, integrity (SHA-256 hash on creation, reload passes, modified file raises RuntimeError), divergence (identical near zero, different positive, bounded 0–2), effective_mu (grows with age, day zero = 0, custom base), core embedding (cached, shape 1024, float32) |
| `test_cortex_prompt.py` | `core/cortex_prompt.py` | 13 | Dynamic prompt assembly: always includes CORTEX_CORE, includes identity/soul_memory/brainstem/limbic when provided, category hints for all ReflexCategory values, empty contexts produce clean prompt, all layers combined |
| `test_salience.py` | `core/salience.py` | 16 | Split entropy scorer: emotional valence (positive/negative/neutral/bounded), relational depth (high/low/bounded), self-model impact, vulnerability, philosophical layer classification (existential/personal/technical), split entropy markers (noise detection, exploration detection), Lagrangian local computation, integrated complexity |
| `test_training_pair_filter.py` | `core/training_pair_filter.py` | 18 | Identity question detection (multilingual: EN/IT/DE/RU), existential probe detection (multilingual + meta-existential), filter logic (normal pair, identity not rejected, existential not rejected, metadata attached), batch processing (accepted/rejected split, TrainingPair object handling, empty batch) |
| `test_conversation.py` | `core/conversation.py` | 6 | `_format_memories` (empty, single, truncates long text, multiple), ConversationEngine (no model returns placeholder, build_system_prompt includes all sections, memories in prompt, new_conversation resets state, soul threshold from config, set_local_model) |
| `test_secrets.py` | `core/secrets.py` | 8 | Load from apikey files, skip missing files, skip empty files, load from .env, skip comments, file priority over .env, nonexistent directory doesn't crash, strips whitespace |

### When to Run Tests

Run the full campaign after **any** code modification to verify system integrity. The test suite is designed as a regression guard — if all 314 tests pass, DAEDALUS's core invariants are preserved.

Specific scenarios:
- After modifying any `core/` module
- After modifying night cycle logic (`night/`)
- After changing configuration schemas
- Before and after running the night cycle
- Before committing code changes

### Adding New Tests

When developing new features, always create test files under `tests/`. Use the shared fixtures from `tests/conftest.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_my_feature(mock_embedder, sample_config):
    """Test description."""
    # mock_embedder provides a deterministic 1024-dim embedder
    vec = mock_embedder.encode("test text")
    assert vec.shape == (1024,)
```

Guidelines:
- **No GPU required.** All tests must run without CUDA. Mock all model dependencies.
- **Deterministic.** Tests must produce the same result every run. Use `MockEmbedder` (seed from text hash) instead of random vectors.
- **Fast.** The full campaign should complete in under 30 seconds.
- **Isolated.** Use `tmp_path` for any file operations. Never write to the real `memory/`, `identity/`, or `models/` directories.
- **Regression-focused.** When fixing a bug, add a test that would have caught it. See `test_get_episodes_date_filter_no_limit_applied_to_chromadb` in `test_memory_store.py` for an example.

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
* **Quick system check:** Run `python scripts/introspect.py` to see the current identity state, constitutional distance, and memory counts without starting the server.
* **After any code change:** Run `pytest tests/ -v` before launching the server or running the night cycle.

*In the beginning was the dialogue.*
