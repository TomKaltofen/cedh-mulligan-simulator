"""HandGenerator — DataCreator that generates mulligan sequences from a singleton deck."""

import logging
from typing import Any, List, Optional

from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet

from cedh_mulligan_simulator.card_registry import DEFAULT_CARD_REGISTRY, CardRegistry
from cedh_mulligan_simulator.deck import Deck


class HandGenerator(FeatureGroup):
    """Generates mulligan sequences: N_sims × mulligan_steps rows."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"hand", "simulation_id", "mulligan_count", "scenario_id", "remaining_library"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        registry: CardRegistry = features.get_options_key("card_registry") or DEFAULT_CARD_REGISTRY
        n_simulations: int = features.get_options_key("n_simulations") or 100_000
        deck_size: int = features.get_options_key("deck_size") or 99
        mulligan_steps: int = features.get_options_key("mulligan_steps") or 4
        scenario_id: str = features.get_options_key("scenario_id") or "baseline"

        hands: List[List[str]] = []
        sim_ids: List[int] = []
        mull_counts: List[int] = []
        scenario_ids: List[str] = []
        remaining_libraries: List[List[str]] = []

        for sim in range(n_simulations):
            deck = Deck(registry, deck_size)
            for step in range(mulligan_steps):
                deck.mulligan()
                hand_size = 7 if step <= 1 else 7 - (step - 1)  # 7, 7, 6, 5
                deck.draw(hand_size)
                hands.append(deck.hand)
                sim_ids.append(sim)
                mull_counts.append(step)
                scenario_ids.append(scenario_id)
                remaining_libraries.append(list(deck.library))

        logging.debug("Create hands: %d", len(hands))
        return {
            "hand": hands,
            "simulation_id": sim_ids,
            "mulligan_count": mull_counts,
            "scenario_id": scenario_ids,
            "remaining_library": remaining_libraries,
        }
