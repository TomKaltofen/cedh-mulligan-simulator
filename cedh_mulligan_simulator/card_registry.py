"""Card registry: generic card data system with a minimal default (wastes)."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

CardRegistry = Dict[str, "Card"]


@dataclass(frozen=True)
class Mana:
    """Mana amount across all five colors. Fields: total, white, blue, black, red, green."""

    total: int = 0
    white: int = 0
    blue: int = 0
    black: int = 0
    red: int = 0
    green: int = 0


@dataclass(frozen=True)
class ManaRequirement:
    """Mana cost target for any commander. All five colors plus total."""

    total: int = 0
    white: int = 0
    blue: int = 0
    black: int = 0
    red: int = 0
    green: int = 0


@dataclass(frozen=True)
class Card:
    """Immutable card data for mana-line evaluation."""

    name: str
    type: str
    cost: Optional[Mana] = None
    produces: Optional[Mana] = None
    t1_mana: Optional[Mana] = None
    t2_mana: Optional[Mana] = None
    sac_mana: Optional[Mana] = None
    tap_produces: Optional[Mana] = None
    evoke_cost: Optional[Mana] = None
    requires: Optional[str] = None
    cmc: Optional[int] = None
    free_condition: Optional[str] = None
    produces_base: Optional[Mana] = None
    produces_cmc: Optional[bool] = None
    destination: Optional[str] = None
    sacrifices_on_use: bool = False
    fetch_targets: Optional[Tuple[str, ...]] = None  # Card name prefixes to search for
    sac_search_turn: Optional[int] = None  # Turn when land can sacrifice to search for artifact
    sac_search_artifact_cmcs: Optional[Tuple[int, ...]] = None  # CMC values of artifacts to search for


def any_color(total: int) -> Mana:
    """Create a Mana that can satisfy any single color requirement up to *total*."""
    return Mana(total, white=total, blue=total, black=total, red=total, green=total)


def creature(name: str, **kwargs: Any) -> Card:
    return Card(name=name, type="creature", **kwargs)


def land(name: str, **kwargs: Any) -> Card:
    return Card(name=name, type="land", **kwargs)


def ritual(name: str, **kwargs: Any) -> Card:
    return Card(name=name, type="ritual", **kwargs)


def sacrifice_outlet(name: str, **kwargs: Any) -> Card:
    return Card(name=name, type="sacrifice_outlet", **kwargs)


def tutor(name: str, **kwargs: Any) -> Card:
    return Card(name=name, type="tutor", **kwargs)


# Cost-tier shorthands for common patterns
def creature_0(name: str, **kwargs: Any) -> Card:
    """Zero-cost creature (cmc=0)."""
    return Card(name=name, type="creature", cost=Mana(), cmc=0, **kwargs)


def artifact_creature_0(name: str, **kwargs: Any) -> Card:
    """Zero-cost artifact creature (cmc=0)."""
    return Card(name=name, type="artifact_creature", cost=Mana(), cmc=0, **kwargs)


def artifact_0(name: str, **kwargs: Any) -> Card:
    """Zero-cost artifact."""
    return Card(name=name, type="artifact", cost=Mana(), **kwargs)


def artifact_1(name: str, **kwargs: Any) -> Card:
    """1-cost artifact."""
    return Card(name=name, type="artifact", cost=Mana(1), **kwargs)


def equipment_1(name: str, **kwargs: Any) -> Card:
    """1-cost equipment."""
    return Card(name=name, type="equipment", cost=Mana(1), **kwargs)


def build_registry(*cards: Card) -> CardRegistry:
    """Build a CardRegistry from Card objects, keyed by card name."""
    return {card.name: card for card in cards}


WASTES = land("wastes", t1_mana=Mana(1), t2_mana=Mana(1))

DEFAULT_CARD_REGISTRY: CardRegistry = {"wastes": WASTES}
