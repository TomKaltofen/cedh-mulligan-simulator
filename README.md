[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# cEDH Mulligan Simulator

A Monte Carlo mulligan simulator for **Braids, Arisen Nightmare** (mono-black cEDH), built as an [mloda](https://github.com/mloda-ai/mloda) feature engineering pipeline.

Every card mechanic, mana calculation, mulligan decision, and A/B scenario is a composable, reusable **Feature Group plugin** — demonstrating mloda's "build once, reuse everywhere" principle in a combinatorics-heavy domain.

## What it does

- Simulates 100K+ London Mulligan sequences (7→7→6→5) for a full 99-card deck
- Evaluates T1/T2/T3 castability of Braids for each hand
- Runs A/B scenarios (remove a card, replace with a Swamp) to measure marginal card contribution
- Produces per-scenario statistics, convergence plots, and card delta tables

## Quick start

```bash
source .venv/bin/activate
uv sync --all-extras
python run_simulation.py
```

## Running checks

```bash
tox
```

## Architecture

```
HandGenerator (DataCreator)  →  Turn1 / Turn2 / Turn3  →  MulliganResult
      ↓                               ↓
  hand, sim_id             t1_castable, t2_castable, t3_castable
                                       ↓
                           Proportion / Statistics / Plots
```

Card data lives in `card_database/` and `card_registries/`. Mechanics are in `cedh_mulligan_simulator/feature_groups/mulligan/`. See `todo.md` for the current backlog.

## Related

- **[mloda](https://github.com/mloda-ai/mloda)**: Core library — declarative feature resolution, plugin composability, compute framework abstraction.
- **[mloda-registry](https://github.com/mloda-ai/mloda-registry)**: Plugin guides and patterns used throughout this project.
