"""Deck registry builder.

Resolves each card in a deck list against the existing card_database first,
then falls back to Scryfall + the mapper for unknown cards.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import card_database.black as _black
import card_database.colorless as _colorless
import card_database.lands as _lands
from cedh_mulligan_simulator.card_registry import Card, ManaRequirement
from deck_importer.mapper import MappedCard, map_card, map_commander_cost
from deck_importer.parser import DeckList
from deck_importer.scryfall import fetch_card

# Ordered list of (module_path, module_object) pairs.
# Lands first so MDFC land faces take priority over creature faces.
_DB_MODULES: List[Tuple[str, Any]] = [
    ("card_database.lands", _lands),
    ("card_database.colorless", _colorless),
    ("card_database.black", _black),
]


@dataclass
class DeckRegistryEntry:
    """One card in the resolved deck registry."""

    snake_name: str
    card: Card
    # Source info for code generation
    source_module: Optional[str]  # e.g. "card_database.black"; None = auto-inferred
    var_name: Optional[str]  # Python variable name in source_module; None = auto-inferred
    needs_review: bool
    review_reason: str


def scan_card_database() -> Dict[str, Tuple[str, str]]:
    """Scan all card_database sub-modules and return a lookup dict.

    Returns:
        ``{card.name: (python_var_name, module_path)}``

    For cards that appear in multiple modules (e.g. MDFC land + creature faces)
    the entry from the first module in priority order is kept.
    """
    result: Dict[str, Tuple[str, str]] = {}
    for module_path, module_obj in _DB_MODULES:
        for var_name, obj in inspect.getmembers(module_obj):
            if not var_name.isupper():
                continue
            if not isinstance(obj, Card):
                continue
            if obj.name not in result:
                result[obj.name] = (var_name, module_path)
    return result


def _resolve_card(
    snake_name: str,
    db_lookup: Dict[str, Tuple[str, str]],
) -> DeckRegistryEntry:
    """Resolve a single card, returning a DeckRegistryEntry."""
    # 1. Check existing card_database
    if snake_name in db_lookup:
        var_name, module_path = db_lookup[snake_name]
        # Find the actual Card object
        for mod_path, mod_obj in _DB_MODULES:
            if mod_path == module_path:
                card = getattr(mod_obj, var_name, None)
                if isinstance(card, Card):
                    return DeckRegistryEntry(
                        snake_name=snake_name,
                        card=card,
                        source_module=module_path,
                        var_name=var_name,
                        needs_review=False,
                        review_reason="",
                    )

    # 2. Fall back to Scryfall
    scryfall_data = fetch_card(snake_name.replace("_", " "))
    if scryfall_data is None:
        # Card not found anywhere — create a minimal placeholder
        placeholder = Card(name=snake_name, type="unknown")
        return DeckRegistryEntry(
            snake_name=snake_name,
            card=placeholder,
            source_module=None,
            var_name=None,
            needs_review=True,
            review_reason="card not found in database or Scryfall — add manually",
        )

    mapped: MappedCard = map_card(scryfall_data, snake_name)
    return DeckRegistryEntry(
        snake_name=snake_name,
        card=mapped.card,
        source_module=None,
        var_name=None,
        needs_review=mapped.needs_review,
        review_reason=mapped.review_reason,
    )


def build_deck_registry(deck_list: DeckList) -> List[DeckRegistryEntry]:
    """Resolve all cards in a deck list to DeckRegistryEntry objects.

    Args:
        deck_list: Parsed deck list (mainboard only; sideboard cards are skipped).

    Returns:
        One DeckRegistryEntry per *unique* mainboard card name, in encounter order.
    """
    db_lookup = scan_card_database()

    seen: Dict[str, DeckRegistryEntry] = {}
    entries: List[DeckRegistryEntry] = []

    all_cards = list(deck_list.mainboard)
    # Exclude the commander itself from the 99
    if deck_list.commander and deck_list.commander in all_cards:
        all_cards.remove(deck_list.commander)

    for snake_name in all_cards:
        if snake_name in seen:
            continue
        entry = _resolve_card(snake_name, db_lookup)
        seen[snake_name] = entry
        entries.append(entry)

    return entries


def resolve_commander_cost(
    commander_name: str, db_lookup: Optional[Dict[str, Tuple[str, str]]] = None
) -> ManaRequirement:
    """Resolve a commander's mana cost to a ManaRequirement.

    Checks card_database first, then falls back to Scryfall.
    """
    if db_lookup is None:
        db_lookup = scan_card_database()

    # Check db
    if commander_name in db_lookup:
        var_name, module_path = db_lookup[commander_name]
        for mod_path, mod_obj in _DB_MODULES:
            if mod_path == module_path:
                card = getattr(mod_obj, var_name, None)
                if isinstance(card, Card) and card.cost is not None:
                    c = card.cost
                    return ManaRequirement(
                        total=c.total, white=c.white, blue=c.blue, black=c.black, red=c.red, green=c.green
                    )

    # Scryfall fallback
    scryfall_data = fetch_card(commander_name.replace("_", " "))
    if scryfall_data is not None:
        total, white, blue, black, red, green = map_commander_cost(scryfall_data)
        return ManaRequirement(total=total, white=white, blue=blue, black=black, red=red, green=green)

    # Unknown — return zero cost and let the user fix it
    return ManaRequirement()
