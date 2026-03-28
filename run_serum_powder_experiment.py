"""Serum Powder Experiment: compare Braids mulligan with and without Serum Powder."""

import logging
from typing import Any, Dict, List

from card_database.colorless import SERUM_POWDER
from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.card_registry import Mana, build_registry, land
from cedh_mulligan_simulator.feature_groups.mulligan import HandGenerator, MulliganResult, Turn1, Turn2
from cedh_mulligan_simulator.feature_groups.statistics import AverageKeptHandSize, MeanMulliganDepth
from run_helpers import ProviderSpec, remove_card, run_experiment

N_SIMULATIONS = 50000

# Serum Powder replaces Dismember (generic removal, weakest synergy with sacrifice themes).
REPLACED_CARD = "dismember"

_BRAIDS_WITHOUT_DISMEMBER = remove_card(BRAIDS_REGISTRY, REPLACED_CARD)
_BRAIDS_WITH_SERUM_POWDER = build_registry(*_BRAIDS_WITHOUT_DISMEMBER.values(), SERUM_POWDER)
_EXTRA_SWAMP = land("swamp_3", t1_mana=Mana(1, black=1), t2_mana=Mana(1, black=1))
_BRAIDS_WITH_SWAMP = build_registry(*_BRAIDS_WITHOUT_DISMEMBER.values(), _EXTRA_SWAMP)

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "baseline",
        "name": "Baseline (with Dismember)",
        "registry": BRAIDS_REGISTRY,
        "cost": BRAIDS_COST,
    },
    {
        "id": "serum_powder",
        "name": "Serum Powder (replaces Dismember)",
        "registry": _BRAIDS_WITH_SERUM_POWDER,
        "cost": BRAIDS_COST,
    },
    {
        "id": "swamp",
        "name": "Swamp (replaces Dismember)",
        "registry": _BRAIDS_WITH_SWAMP,
        "cost": BRAIDS_COST,
    },
]

PROVIDER_SPECS: List[ProviderSpec] = [
    (HandGenerator, {"hand", "simulation_id", "mulligan_count", "scenario_id"}),
    (Turn1, None),
    (Turn2, None),
    (MulliganResult, {"MulliganResult"}),
    (MeanMulliganDepth, {"MeanMulliganDepth"}),
    (AverageKeptHandSize, {"AverageKeptHandSize"}),
]

FEATURE_NAMES = [
    "MulliganResult",
    "hand",
    "simulation_id",
    "mulligan_count",
    "scenario_id",
    "hand__t1",
    "hand__t1__t2",
    "MeanMulliganDepth",
    "AverageKeptHandSize",
]


def main() -> None:
    run_experiment(
        scenarios=SCENARIOS,
        provider_specs=PROVIDER_SPECS,
        feature_names=FEATURE_NAMES,
        n_simulations=N_SIMULATIONS,
        experiment_id="serum_powder",
        title="Serum Powder Experiment",
        delta_precision=2,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    main()
