"""Scryfall card data → simulation Card mapper.

Auto-infers Card fields from Scryfall's mana_cost, type_line, and oracle_text.
Cards whose rules cannot be fully modelled are flagged with needs_review=True.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cedh_mulligan_simulator.card_registry import Card, Mana, any_color


@dataclass
class MappedCard:
    """A Card produced by the mapper, optionally flagged for manual review."""

    card: Card
    needs_review: bool = False
    review_reason: str = ""


# ── Mana cost / symbol parsing ─────────────────────────────────────────────


def parse_mana_cost(mana_cost_str: str) -> Mana:
    """Parse a Scryfall mana cost string into a Mana object.

    Examples::

        "{B}"        -> Mana(1, black=1)
        "{1}{B}"     -> Mana(2, black=1)
        "{2}{B}{B}"  -> Mana(4, black=2)
        "{0}"        -> Mana()
        "{X}"        -> Mana()   # variable X treated as 0
        "{W/B}"      -> Mana(1, white=1)  # hybrid: primary colour counts
    """
    total = white = blue = black = red = green = 0

    for symbol in re.findall(r"\{([^}]+)\}", mana_cost_str):
        sym = symbol.upper()
        if sym == "W":
            total += 1
            white += 1
        elif sym == "U":
            total += 1
            blue += 1
        elif sym == "B":
            total += 1
            black += 1
        elif sym == "R":
            total += 1
            red += 1
        elif sym == "G":
            total += 1
            green += 1
        elif sym.isdigit():
            total += int(sym)
        elif sym in ("X", "C", "S"):
            pass  # variable / colorless / snow, ignore
        elif "/" in sym:
            # Hybrid ({W/B}, {2/B}) or phyrexian ({B/P}): count primary colour
            primary = sym.split("/")[0]
            if primary in ("W", "U", "B", "R", "G"):
                total += 1
                if primary == "W":
                    white += 1
                elif primary == "U":
                    blue += 1
                elif primary == "B":
                    black += 1
                elif primary == "R":
                    red += 1
                elif primary == "G":
                    green += 1
            elif primary.isdigit():
                total += 1  # {2/B} generic alternative, count as 1

    return Mana(total, white=white, blue=blue, black=black, red=red, green=green)


def parse_mana_symbols(text: str) -> Mana:
    """Parse mana symbols embedded in arbitrary oracle text (e.g. 'Add {B}{B}{B}.')."""
    return parse_mana_cost(text)


# ── Oracle text helpers ─────────────────────────────────────────────────────


def _enters_tapped(oracle_text: str) -> bool:
    lower = oracle_text.lower()
    return "enters the battlefield tapped" in lower or "enters tapped" in lower


def _parse_tap_add_mana(oracle_text: str) -> Optional[Mana]:
    """Extract mana from the first '{T}: Add ...' pattern in oracle text."""
    # Match "{T}: Add {symbols}." or "{T}: Add one mana of any color."
    tap_match = re.search(r"\{T\}: Add ([^.]+)\.", oracle_text)
    if not tap_match:
        return None
    mana_str = tap_match.group(1)
    if re.search(r"one mana of any color|mana of any type", mana_str, re.IGNORECASE):
        return any_color(1)
    # Parse embedded mana symbols
    parsed = parse_mana_symbols(mana_str)
    if parsed.total == 0 and mana_str.strip():
        # No recognised symbols. Conservative: return None to flag review
        return None
    return parsed


def _parse_ritual_produces(oracle_text: str) -> Optional[Mana]:
    """Extract mana from 'Add {X}' patterns (not preceded by {T}:)."""
    add_match = re.search(r"(?<!\{T\}: )Add (\{[^.]+\})", oracle_text)
    if not add_match:
        # Try "Add {symbols}." pattern anywhere
        add_match = re.search(r"\bAdd (\{[^.]+)\.", oracle_text)
    if not add_match:
        return None
    mana_str = add_match.group(1)
    parsed = parse_mana_symbols(mana_str)
    return parsed if parsed.total > 0 else None


def _detect_tutor_destination(oracle_text: str) -> Optional[str]:
    """Detect tutor destination from oracle text."""
    lower = oracle_text.lower()
    if re.search(r"put (it|that card) onto the battlefield", lower):
        return "battlefield"
    if re.search(r"put (it|that card) into your hand", lower):
        return "hand"
    if re.search(r"put (it|that card) into (your|its owner's) graveyard", lower):
        return "graveyard"
    if re.search(r"put (it|that card) on top", lower):
        return "top"
    if re.search(r"reveal it and put (it|that card) into your hand", lower):
        return "hand"
    return None


def _detect_requires(oracle_text: str) -> Optional[str]:
    """Detect common 'requires' keywords from oracle text."""
    lower = oracle_text.lower()
    if "exile a card from your hand" in lower:
        return "exile_from_hand"
    if "sacrifice a land" in lower:
        return "sacrifice_land"
    if "sacrifice a swamp" in lower:
        return "sacrifice_swamp"
    return None


# ── Type detection ──────────────────────────────────────────────────────────


def _detect_card_type(type_line: str, oracle_text: str, mana_cost_str: str) -> str:
    """Infer simulation card type from Scryfall type_line and oracle_text."""
    tl = type_line.lower()

    if "land" in tl:
        return "land"

    if "artifact" in tl and "creature" in tl:
        parsed_cost = parse_mana_cost(mana_cost_str)
        return "artifact_creature" if parsed_cost.total == 0 else "creature"

    if "creature" in tl:
        # Check for ritual-like mana production
        if re.search(r"add \{", oracle_text, re.IGNORECASE):
            return "creature"  # mana-dork, still a creature type
        return "creature"

    if "artifact" in tl:
        # Check if it produces mana
        if re.search(r"add \{", oracle_text, re.IGNORECASE):
            return "artifact"
        return "artifact"

    if "instant" in tl or "sorcery" in tl:
        # Check if it's a ritual (adds mana)
        if re.search(r"\badd \{[^}]+\}", oracle_text, re.IGNORECASE):
            return "ritual"
        # Check if it's a tutor (searches library)
        if re.search(r"search your library", oracle_text, re.IGNORECASE):
            return "tutor"
        if "instant" in tl:
            return "instant"
        return "sorcery"

    if "enchantment" in tl:
        return "enchantment"

    if "planeswalker" in tl:
        return "planeswalker"

    if "battle" in tl:
        return "battle"

    return "unknown"


# ── Main mapper ─────────────────────────────────────────────────────────────


def map_card(scryfall_data: Dict[str, Any], snake_name: str) -> MappedCard:
    """Map a Scryfall card object to a simulation Card.

    Args:
        scryfall_data: Raw Scryfall card object dict.
        snake_name: Pre-computed snake_case card name (used as ``Card.name``).

    Returns:
        MappedCard with auto-inferred fields and a needs_review flag if
        any simulation property could not be reliably determined.
    """
    mana_cost_str: str = scryfall_data.get("mana_cost", "") or ""
    cmc_raw = scryfall_data.get("cmc", 0)
    cmc: int = int(cmc_raw) if cmc_raw is not None else 0
    type_line: str = scryfall_data.get("type_line", "") or ""
    oracle_text: str = scryfall_data.get("oracle_text", "") or ""

    cost = parse_mana_cost(mana_cost_str) if mana_cost_str else None
    card_type = _detect_card_type(type_line, oracle_text, mana_cost_str)

    review_notes: List[str] = []

    if card_type == "land":
        return _map_land(snake_name, oracle_text, review_notes)

    if card_type == "ritual":
        return _map_ritual(snake_name, cost, oracle_text, review_notes)

    if card_type == "tutor":
        return _map_tutor(snake_name, cost, cmc, oracle_text, review_notes)

    if card_type in ("creature", "artifact_creature"):
        return _map_creature(snake_name, cost, cmc, card_type, oracle_text, review_notes)

    if card_type == "artifact":
        return _map_artifact(snake_name, cost, oracle_text, review_notes)

    # Catch-all: generic Card for instants, sorceries, enchantments, planeswalkers, etc.
    reason = f"type='{card_type}', verify simulation properties"
    return MappedCard(
        card=Card(name=snake_name, type=card_type, cost=cost, cmc=cmc),
        needs_review=True,
        review_reason=reason,
    )


def _map_land(snake_name: str, oracle_text: str, review_notes: List[str]) -> MappedCard:
    tapped = _enters_tapped(oracle_text)
    tap_mana = _parse_tap_add_mana(oracle_text)

    requires = _detect_requires(oracle_text)

    if tap_mana is None:
        review_notes.append("could not parse {T}: Add ability")
        # Default to a colourless tapped land rather than crashing.
        tap_mana = Mana(1)

    t1_mana = Mana() if tapped else tap_mana
    t2_mana = tap_mana

    needs_review = bool(review_notes)
    reason = "; ".join(review_notes) if review_notes else ""

    return MappedCard(
        card=Card(
            name=snake_name,
            type="land",
            t1_mana=t1_mana,
            t2_mana=t2_mana,
            requires=requires,
        ),
        needs_review=needs_review,
        review_reason=reason,
    )


def _map_ritual(
    snake_name: str,
    cost: Optional[Mana],
    oracle_text: str,
    review_notes: List[str],
) -> MappedCard:
    produces = _parse_ritual_produces(oracle_text)
    if produces is None:
        review_notes.append("could not parse mana production; set produces= manually")
        produces = Mana()

    needs_review = bool(review_notes)
    reason = "; ".join(review_notes) if review_notes else ""

    return MappedCard(
        card=Card(name=snake_name, type="ritual", cost=cost, produces=produces),
        needs_review=needs_review,
        review_reason=reason,
    )


def _map_tutor(
    snake_name: str,
    cost: Optional[Mana],
    cmc: int,
    oracle_text: str,
    review_notes: List[str],
) -> MappedCard:
    destination = _detect_tutor_destination(oracle_text)
    if destination is None:
        review_notes.append("could not detect tutor destination; set destination= manually")

    needs_review = bool(review_notes)
    reason = "; ".join(review_notes) if review_notes else ""

    return MappedCard(
        card=Card(name=snake_name, type="tutor", cost=cost, cmc=cmc, destination=destination),
        needs_review=needs_review,
        review_reason=reason,
    )


def _map_creature(
    snake_name: str,
    cost: Optional[Mana],
    cmc: int,
    card_type: str,
    oracle_text: str,
    review_notes: List[str],
) -> MappedCard:
    evoke_cost: Optional[Mana] = None
    free_condition: Optional[str] = None
    produces: Optional[Mana] = None

    # Evoke keyword
    evoke_match = re.search(r"Evoke\s+(\{[^}]+\}(?:\{[^}]+\})*)", oracle_text)
    if evoke_match:
        evoke_cost = parse_mana_cost(evoke_match.group(1))

    # Mana production (mana dork)
    if re.search(r"add \{", oracle_text, re.IGNORECASE):
        tap_mana = _parse_tap_add_mana(oracle_text)
        if tap_mana is not None:
            produces = tap_mana
        else:
            review_notes.append("creature may produce mana, verify produces=")

    review_notes.append("verify evoke, free conditions, and other special rules")
    needs_review = True
    reason = "; ".join(review_notes)

    return MappedCard(
        card=Card(
            name=snake_name,
            type=card_type,
            cost=cost,
            cmc=cmc,
            evoke_cost=evoke_cost,
            free_condition=free_condition,
            produces=produces,
        ),
        needs_review=needs_review,
        review_reason=reason,
    )


def _map_artifact(
    snake_name: str,
    cost: Optional[Mana],
    oracle_text: str,
    review_notes: List[str],
) -> MappedCard:
    produces: Optional[Mana] = None

    if re.search(r"add \{", oracle_text, re.IGNORECASE):
        produces = _parse_ritual_produces(oracle_text)
        if produces is None:
            review_notes.append("artifact may produce mana, verify produces=")

    requires = _detect_requires(oracle_text)

    # Flag artifacts with special rules
    if re.search(r"sacrifice|exile|threshold|metalcraft|affinity|convoke", oracle_text, re.IGNORECASE):
        review_notes.append("artifact has special activation condition, review requires=")

    needs_review = bool(review_notes)
    reason = "; ".join(review_notes) if review_notes else ""

    return MappedCard(
        card=Card(name=snake_name, type="artifact", cost=cost, produces=produces, requires=requires),
        needs_review=needs_review,
        review_reason=reason,
    )


def map_commander_cost(scryfall_data: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    """Return (total, white, blue, black, red, green) for a commander's mana cost."""
    mana_cost_str: str = scryfall_data.get("mana_cost", "") or ""
    m = parse_mana_cost(mana_cost_str)
    return m.total, m.white, m.blue, m.black, m.red, m.green
