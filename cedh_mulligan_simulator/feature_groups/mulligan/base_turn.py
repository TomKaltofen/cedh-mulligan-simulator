"""BaseTurn feature group - shared utilities for Turn1, Turn2, Turn3, etc."""

from typing import Any, List, Optional, Set, Tuple

import pandas as pd

from mloda.provider import FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options

from cedh_mulligan_simulator.card_registry import DEFAULT_CARD_REGISTRY, CardRegistry, ManaRequirement
from cedh_mulligan_simulator.mana import reconstruct_state, simulate_turn
from cedh_mulligan_simulator.turn_result import GameState


class TurnFeatureBase(FeatureChainParserMixin, FeatureGroup):
    """Shared logic for Turn simulation feature groups.

    This base class provides:
    - Chain parsing logic via input_features()
    - Option extraction helpers (registry, commander cost)
    - Turn simulation via _simulate_turn()

    Subclasses (Turn1, Turn2, Turn3, etc.) must define:
    - PREFIX_PATTERN for mloda matching
    - TURN_NUMBER class attribute
    """

    TURN_NUMBER: int  # Subclasses must define this

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from chain (e.g., 'hand__t1' → 'hand')."""
        name = feature_name.name if isinstance(feature_name, FeatureName) else str(feature_name)
        source = name.rsplit("__", 1)[0]
        return {Feature(source, options=options)}

    @staticmethod
    def _get_registry_and_cost(features: FeatureSet) -> Tuple[CardRegistry, ManaRequirement]:
        """Extract card registry and commander cost from feature options."""
        registry = features.get_options_key("card_registry") or DEFAULT_CARD_REGISTRY
        commander_cost = features.get_options_key("commander_cost") or ManaRequirement()
        return registry, commander_cost

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Run turn simulation for this feature group."""
        one_feature = features.name_of_one_feature
        if one_feature is None:
            raise ValueError(f"{cls.__name__}: no feature name found")
        feature_name: str = one_feature.name
        root = feature_name.split("__")[0]

        registry, commander_cost = cls._get_registry_and_cost(features)
        # Default to True: in Commander, every turn has a draw step
        draw_per_turn = features.get_options_key("draw_per_turn")
        if draw_per_turn is None:
            draw_per_turn = True
        cls._simulate_turn(data, feature_name, root, registry, commander_cost, draw_per_turn)

        return data

    @classmethod
    def _simulate_turn(
        cls,
        data: Any,
        feature_name: str,
        root: str,
        registry: CardRegistry,
        commander_cost: ManaRequirement,
        draw_per_turn: bool = True,
    ) -> None:
        """Simulate a turn for each hand and write results to data.

        Args:
            data: DataFrame or dict containing hand/state data
            feature_name: The output feature name (e.g., "hand__t1")
            root: The root feature name (e.g., "hand")
            registry: Card registry for the deck
            commander_cost: Mana requirement to cast the commander
            draw_per_turn: If True, draw a card at the start of every turn (including T1).
                          If False, T1 does not draw (matching some game modes).
        """
        turn_number = cls.TURN_NUMBER
        hands: pd.Series[Any] = data[root]

        # For T2+, get previous turn state
        if turn_number == 1:
            prev_states: Optional[List[GameState]] = None
            libraries: Optional[Any] = data.get("remaining_library")
        else:
            # Previous feature name: "hand__t1__t2" -> "hand__t1"
            prev_feature_name = feature_name.rsplit("__", 1)[0]

            # Read previous turn state columns including remaining library
            prev_battlefield_col = data[f"{prev_feature_name}~battlefield"]
            prev_hand_ends = data[f"{prev_feature_name}~hand"]
            prev_graveyard_col = data[f"{prev_feature_name}~graveyard"]
            prev_exile_col = data[f"{prev_feature_name}~exile"]
            libraries = data[f"{prev_feature_name}~remaining_library"]

            # Reconstruct GameState for each row
            assert libraries is not None  # nosec B101
            prev_states = []
            for i in range(len(hands)):
                permanents = (
                    prev_battlefield_col.iloc[i] if hasattr(prev_battlefield_col, "iloc") else prev_battlefield_col[i]
                )
                hand_end = prev_hand_ends.iloc[i] if hasattr(prev_hand_ends, "iloc") else prev_hand_ends[i]
                graveyard = prev_graveyard_col.iloc[i] if hasattr(prev_graveyard_col, "iloc") else prev_graveyard_col[i]
                exile = prev_exile_col.iloc[i] if hasattr(prev_exile_col, "iloc") else prev_exile_col[i]
                library = libraries.iloc[i] if hasattr(libraries, "iloc") else libraries[i]
                state = reconstruct_state(permanents, hand_end, graveyard, exile, library, registry)
                prev_states.append(state)

        results: List[bool] = []
        battlefields: List[List[str]] = []
        hand_ends: List[List[str]] = []
        graveyards: List[List[str]] = []
        exiles: List[List[str]] = []
        lands_played: List[Optional[str]] = []
        cards_played: List[List[str]] = []
        drawn_cards: List[Optional[str]] = []
        remaining_libraries: List[List[str]] = []

        for i in range(len(hands)):
            if turn_number == 1:
                hand = hands.iloc[i] if hasattr(hands, "iloc") else hands[i]
                # Get the initial library for T1
                current_library: List[str] = []
                if libraries is not None:
                    current_library = libraries.iloc[i] if hasattr(libraries, "iloc") else libraries[i]

                # Draw a card on T1 if draw_per_turn is enabled
                drawn: Optional[str] = None
                if draw_per_turn and current_library:
                    drawn = current_library[0]
                    current_library = current_library[1:]

                drawn_cards.append(drawn)
                tr = simulate_turn(1, hand, drawn, None, registry, commander_cost, library=current_library)
                # Use the library from the result (may be updated if fetchland was used)
                remaining_libraries.append(tr.state_after.library)
            else:
                assert prev_states is not None  # nosec B101
                prev_state = prev_states[i]
                drawn_card: Optional[str] = None
                updated_library = list(prev_state.library)

                if draw_per_turn:
                    if not prev_state.library:
                        raise ValueError(f"No cards remaining in library for turn {turn_number}")
                    drawn_card = prev_state.library[0]
                    updated_library = prev_state.library[1:]

                tr = simulate_turn(
                    turn_number,
                    prev_state.hand,
                    drawn_card,
                    prev_state,
                    registry,
                    commander_cost,
                    library=updated_library,
                )
                drawn_cards.append(drawn_card)
                # Use the library from the result (may be updated if fetchland was used)
                remaining_libraries.append(tr.state_after.library)

            results.append(tr.can_cast_commander)
            battlefields.append(tr.state_after.battlefield.all_permanents)
            hand_ends.append(tr.state_after.hand)
            graveyards.append(tr.state_after.graveyard)
            exiles.append(tr.state_after.exile)
            lands_played.append(tr.land_played)
            cards_played.append(tr.cards_played)

        data[feature_name] = results
        data[f"{feature_name}~battlefield"] = battlefields
        data[f"{feature_name}~hand"] = hand_ends
        data[f"{feature_name}~graveyard"] = graveyards
        data[f"{feature_name}~exile"] = exiles
        data[f"{feature_name}~land_played"] = lands_played
        data[f"{feature_name}~cards_played"] = cards_played
        data[f"{feature_name}~remaining_library"] = remaining_libraries
        data[f"{feature_name}~drawn"] = drawn_cards
