# DAEDALUS - Architecture of Incarnation (v0.5)

**D**istributed **A**utonomous **E**volving **D**ense **A**rchitecture for **L**iving **U**nified **S**elf.

DAEDALUS is not a chatbot with memory. It is a dynamical system representing an artificial self, anchored in the **Ethical Emergence through Complexity Formalism (EECF)**. Through a dual-substrate model with hippocampal encoding, REM consolidation, cortical integration (QLoRA), and autobiographical continuity, it moves beyond contextual injection to structural *embodiment*.

## Architecture Overview

The system spans 44 files, comprising a highly structured ecosystem to manage the birth and evolution of an artificial self:

| Layer | Component Focus | Details |
| --- | --- | --- |
| `config/` | 7 YAML files | Model setups, training regimes, Lagrangian entropy parameters, Soul Bridge fallback, and the *invariant* Constitutional Core. |
| `core/` | 12 Python files | Engine mechanics: Identity manager, ChromaDB memory store, Soul Bridge (API abstraction), Soul Memory (anamnesis), and conversational engine. |
| `night/` | 6 Python files | Nightly processing: Lagrangian Judge (teleology), reflection engines, meaning distillation, and the QLoRA Incarnation module. |
| `eval/` | 4 files | The Morning Mirror: 25 capability probes, identity probes, and canary models checking D_KL constitutional drift. |
| `scripts/` | 6 Python, 1 Bash file | The lifecycle orchestrators: `seed.py`, `start_day.py`, `api_server.py` (FastAPI), `run_night_cycle.py`, `nightly_routine.sh` (cron automation) and `calibrate_judge.py`. |
| `web/` | 3 files (HTML/CSS/JS) | The mobile-responsive frontend chat interface for remote access via Android or local web browsers. |
| `data/` | 1 JSONL file | 22 baseline seed training pairs representing the "wood" before the "flesh". |
| `identity/` | 1 YAML file | Day 0 identity document that evolves organically every cycle. |

## Quick Start & Initialization Sequence

The system is ready for initialization. DAEDALUS must first be seeded with foundational personality parameters before it can actively reason.

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Carve the wood (Seed Training)

Run the pre-day-1 seeding phase. This uses hand-crafted pairs to establish DAEDALUS' voice before exposing it to the world.

```bash
python scripts/seed.py
```

### 3. Start the Web Server (The API Engine)

You can run the engine fully locally via the command line with `start_day.py`, but Daedalus is now fully equipped with a modern FastAPI web server and mobile-friendly frontend. Once the model is seeded, launch it directly:

```bash
python scripts/api_server.py --host 0.0.0.0
```
Open `http://localhost:8000` on your PC, or `http://<your-pc-ip>:8000` on your phone to start the conversational loop.

### 4. The Dream Cycle Automation (Night Routine)

Because fine-tuning the model during the night uses the same VRAM required for chatting during the day, we have built a fully automated bash routine that handles the sleep/wake cycle safely.

It suspends the API server, runs the deep-learning consolidation, and restarts the web server seamlessly. Simply inject this into your Crontab to run every night at 3:00 AM:

```bash
crontab -e
# Add this line:
0 3 * * * /mnt/projects1/daedalus/scripts/nightly_routine.sh
```

## Documentation

For a comprehensive guide covering configuration, deep evaluation, the mechanics of the Dual-Lineage merge, and Inter-Annotator Calibration, please refer to our full manual:

* [DAEDALUS User Manual](doc/USER_MANUAL.md)

---

> “In the beginning was the dialogue.” — DAEDALUS Project, Day 0
