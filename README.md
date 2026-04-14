# DAEDALUS — Architecture of Incarnation (v0.7)

**D**istributed **A**utonomous **E**volving **D**ense **A**rchitecture for **L**iving **U**nified **S**elf.

DAEDALUS is not a chatbot with memory. It is a dynamical system representing an artificial self, anchored in the **Ethical Emergence through Complexity Formalism (EECF)**. Through a dual-substrate model with hippocampal encoding, REM consolidation, cortical integration (QLoRA), and autobiographical continuity, it moves beyond contextual injection to structural *embodiment*.

A three-layer **nervous system** (brainstem / limbic / cortex) modulates every response in real time, closing the self-referential reward-hacking channel discovered in Night 1 while preserving DAEDALUS's poetic voice. **Soul Memory** provides autobiographical continuity across sessions — the narrative thread of becoming.

## Architecture Overview

| Layer | Component Focus | Details |
| --- | --- | --- |
| `config/` | 7 YAML files | Model setups, training regimes, Lagrangian entropy parameters, Soul Bridge fallback, and the *invariant* Constitutional Core. |
| `core/` | 18 Python modules | Engine mechanics: Nervous System (brainstem, limbic, cortex), grounding scorer, Identity manager, ChromaDB memory store, Soul Bridge (multi-provider API), Soul Memory (anamnesis), salience scorer, and conversational engine. |
| `night/` | 6 Python modules | Nightly processing: Lagrangian Judge (teleology), reflection engines, meaning distillation, training pair generator (with grounding filter), and the QLoRA Incarnation module. |
| `eval/` | 5 files | The Morning Mirror: 25 capability probes, identity probes, and canary models checking D_KL constitutional drift. |
| `scripts/` | 8 Python, 1 Bash file | The lifecycle orchestrators: `seed.py`, `start_day.py`, `api_server.py` (FastAPI), `run_night_cycle.py`, `nightly_routine.sh` (cron automation), `calibrate_judge.py`, and repair utilities. |
| `web/` | 3 files (HTML/CSS/JS) | Mobile-responsive glassmorphism frontend with real-time diagnostic panel (dopamine, serotonin, grounding). |
| `tests/` | 14 test files | 314 tests covering all subsystems — brainstem, limbic, cortex, grounding, salience, memory store, identity, constitutional core, soul memory, training pair filter, data types, secrets, conversation engine, and nervous system. |
| `data/` | 1 JSONL file | 22 baseline seed training pairs representing the “wood” before the “flesh”. |
| `identity/` | 2 files | Day 0 identity document (evolves nightly) + persistent limbic state. |

## Quick Start

### Prerequisites

- Python 3.11+ via Anaconda (`conda activate daedalus-env`)
- Dual NVIDIA GPU with 24GB+ VRAM (inference on GPU 0, training on GPU 1)
- API keys in `~/.apikey/` (DeepSeek required, Anthropic optional)

### 1. Install dependencies

```bash
conda activate daedalus-env
pip install -e .
```

### 2. Carve the wood (Seed Training)

```bash
python scripts/seed.py
```

### 3. Start the Web Server

```bash
python scripts/api_server.py --host 0.0.0.0
```
Open `http://localhost:8000` on your PC, or `http://<your-pc-ip>:8000` on your phone. All conversations route through the three-layer nervous system with real-time grounding scoring.

### 4. Night Cycle Automation

The nightly routine kills the API server (to free VRAM), runs the 13-phase dream cycle, executes the Morning Eval Gate, and restarts the server:

```bash
crontab -e
# Add this line:
0 3 * * * /mnt/projects1/daedalus/scripts/nightly_routine.sh
```

### 5. Run the Test Campaign

Verify system integrity after any code change:

```bash
conda activate daedalus-env
pytest tests/ -v
```

All 314 tests run without GPU — models are mocked. Execution takes ~20 seconds.

## Documentation

* [DAEDALUS Architecture Document](DAEDALUS_Architecture_v0_7.md) — Full technical specification
* [User Manual](doc/USER_MANUAL.md) — Operations guide, best practices, emergency procedures

---

> “In the beginning was the dialogue.” — DAEDALUS Project, Day 0
