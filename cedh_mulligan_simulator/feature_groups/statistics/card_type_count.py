"""CardTypeCount feature group: counts cards of a given type in each hand."""

from typing import Any

import polars as pl

from mloda.provider import FeatureChainParser, FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options

from cedh_mulligan_simulator.card_registry import DEFAULT_CARD_REGISTRY, CardRegistry


class CardTypeCount(FeatureChainParserMixin, FeatureGroup):
    """Chained FG: counts cards of specified type in each hand.

    Usage:
        Feature("land__type_count")
        Feature("artifact__type_count")
        Feature("creature__type_count")
        Feature("ritual__type_count")
    """

    PREFIX_PATTERN = r"^(.+)__type_count$"
    MIN_IN_FEATURES = 0
    MAX_IN_FEATURES = 0

    # match_feature_group_criteria() inherited from FeatureChainParserMixin

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        """Always depends on 'hand' feature regardless of the card type."""
        return {Feature("hand", options=options)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pl.DataFrame = data
        card_registry = features.get_options_key("card_registry") or DEFAULT_CARD_REGISTRY

        new_cols = []
        for feature in features.features:
            name = feature.get_name()
            parsed_type, _ = FeatureChainParser.parse_feature_name(name, [cls.PREFIX_PATTERN])
            card_type = parsed_type or ""
            new_cols.append(
                df["hand"]
                .map_elements(
                    lambda hand, ct=card_type: _count_type(hand, card_registry, ct),  # type: ignore[misc]
                    return_dtype=pl.Int64,
                )
                .alias(name)
            )

        return df.with_columns(new_cols)


def _count_type(hand: list[str], registry: CardRegistry, card_type: str) -> int:
    """Count how many cards in hand match the given type."""
    count = 0
    for card_name in hand:
        card = registry.get(card_name)
        if card is not None and card.type == card_type:
            count += 1
    return count
