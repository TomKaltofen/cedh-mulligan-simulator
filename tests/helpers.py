"""Test helpers for hand-based scenario testing.

These allow writing focused simulation tests using real card display names::

    hand = build_hand(BRAIDS_REGISTRY, ["Dark Ritual", "Swamp", "Chrome Mox", ...])
    assert evaluate_t1(hand, BRAIDS_REGISTRY, BRAIDS_COST)
"""

from typing import Any, List

import polars as pl

from cedh_mulligan_simulator.card_registry import CardRegistry, ManaRequirement
from cedh_mulligan_simulator.mana import simulate_turn
from deck_importer.parser import to_snake_case


def concat_results(results: list[Any]) -> pl.DataFrame:
    """Horizontal concat of Polars DataFrames, deduplicating columns."""
    seen_cols: set[str] = set()
    parts: list[pl.DataFrame] = []
    for r in results:
        if isinstance(r, pl.DataFrame):
            new_cols = [c for c in r.columns if c not in seen_cols]
            if new_cols:
                parts.append(r.select(new_cols))
                seen_cols.update(new_cols)
    return pl.concat(parts, how="horizontal")


def build_hand(registry: CardRegistry, card_names: List[str]) -> List[str]:
    """Convert display names to snake_case keys and validate against registry.

    Args:
        registry: CardRegistry to validate against.
        card_names: Card display names, e.g. ``["Dark Ritual", "Swamp"]``.

    Returns:
        List of snake_case card name strings, padded with ``"filler"`` entries to
        reach a 7-card hand if fewer than 7 names are provided.

    Raises:
        KeyError: If any card name is not found in the registry.
    """
    snake_names: List[str] = []
    for display_name in card_names:
        snake = to_snake_case(display_name)
        if snake not in registry:
            raise KeyError(f"Card '{display_name}' ('{snake}') not found in registry")
        snake_names.append(snake)

    # Pad to 7 cards with filler
    while len(snake_names) < 7:
        snake_names.append("filler")

    return snake_names


def evaluate_t1(hand: List[str], registry: CardRegistry, cost: ManaRequirement) -> bool:
    """Evaluate whether a 7-card opening hand can cast the commander on Turn 1.

    Args:
        hand: Snake_case card names (7 cards, padded with filler if needed).
        registry: CardRegistry to use for simulation.
        cost: Commander mana cost target.

    Returns:
        ``True`` if the commander can be cast on Turn 1.
    """
    return simulate_turn(1, hand, None, None, registry, cost).can_cast_commander


def evaluate_t2(hand: List[str], registry: CardRegistry, cost: ManaRequirement, draw: str = "filler") -> bool:
    """Evaluate whether a 7-card opening hand can cast the commander by Turn 2.

    Args:
        hand: Snake_case card names (7 cards, padded with filler if needed).
        registry: CardRegistry to use for simulation.
        cost: Commander mana cost target.
        draw: Card drawn on Turn 2 (default: ``"filler"``).

    Returns:
        ``True`` if the commander can be cast on Turn 1 or Turn 2.
    """
    t1 = simulate_turn(1, hand, None, None, registry, cost)
    if t1.can_cast_commander:
        return True
    t2 = simulate_turn(2, t1.state_after.hand, draw, t1.state_after, registry, cost)
    return t2.can_cast_commander
