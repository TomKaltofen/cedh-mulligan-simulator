# Getting Started

## Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv sync --all-extras
```

## Run the simulation

```bash
python run_simulation.py
```

This runs the full multi-scenario comparison: baseline vs. each A/B scenario (remove a card, add a Swamp). Output includes per-scenario statistics and plots saved to `plots/comparison/`.

## Run checks

```bash
tox
```

Runs pytest, ruff format, ruff check, mypy --strict, and bandit.

## Project layout

```
card_database/          # Card definitions (lands, black spells, colorless artifacts)
card_registries/        # Deck registries (e.g., Braids mono-black 99-card list)
cedh_mulligan_simulator/
  card_registry.py      # Card / Mana / ManaRequirement dataclasses
  mana.py               # Mana line enumeration engine (pure Python)
  deck.py               # Deck shuffle / draw utilities
  feature_groups/
    mulligan/           # HandGenerator, Turn1, Turn2, Turn3, MulliganResult
    statistics/         # Proportion, MeanMulliganDepth, AverageKeptHandSize, CardCooccurrence
    plots/              # ConvergencePlot, HandCompositionPlot
tests/                  # Test suite (mirrors source layout)
run_simulation.py       # Multi-scenario comparison entry point
```

## Adding a new scenario

Edit the `SCENARIOS` list in `run_simulation.py`:

```python
SCENARIOS = [
    {"id": "baseline", "name": "Baseline", "registry": BRAIDS_REGISTRY, "cost": BRAIDS_COST},
    {
        "id": "no_dark_ritual",
        "name": "No Dark Ritual",
        "registry": remove_card(BRAIDS_REGISTRY, "dark_ritual"),
        "cost": BRAIDS_COST,
    },
    # ... add more here
]
```

## Adding a new card

Add card definition to the appropriate `card_database/` module, then include it in a registry in `card_registries/`.
