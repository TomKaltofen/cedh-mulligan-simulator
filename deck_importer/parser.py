"""Deck list parser for Moxfield copy-as-text format."""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DeckList:
    """Parsed deck list with mainboard and sideboard card names (snake_case, duplicated for copies)."""

    mainboard: List[str] = field(default_factory=list)
    sideboard: List[str] = field(default_factory=list)
    commander: Optional[str] = None


def to_snake_case(name: str) -> str:
    """Convert a card display name to the snake_case key used in card_database.

    Examples:
        "Dark Ritual" -> "dark_ritual"
        "Urza's Saga" -> "urzas_saga"
        "Braids, Arisen Nightmare" -> "braids_arisen_nightmare"
        "Snow-Covered Swamp" -> "snow_covered_swamp"
    """
    result = name.lower()
    # Remove apostrophes (contractions stay as one word: urza's -> urzas)
    result = result.replace("'", "")
    # Replace any run of non-alphanumeric characters with a single underscore
    result = re.sub(r"[^a-z0-9]+", "_", result)
    # Strip leading/trailing underscores
    result = result.strip("_")
    return result


def parse_deck_list(text: str, commander: Optional[str] = None) -> DeckList:
    """Parse a Moxfield copy-as-text deck list.

    Format::

        1 Dark Ritual
        2 Swamp
        SIDEBOARD:
        1 Dauthi Voidwalker

    Args:
        text: Raw deck list text.
        commander: Optional commander display name; normalised to snake_case.

    Returns:
        DeckList with mainboard/sideboard card names (duplicate entries for multi-copy cards).
    """
    mainboard: List[str] = []
    sideboard: List[str] = []
    in_sideboard = False

    commander_snake = to_snake_case(commander) if commander else None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Section separator — Moxfield uses "SIDEBOARD:", "Sideboard:", etc.
        if re.match(r"^sideboard\s*:?$", line, re.IGNORECASE):
            in_sideboard = True
            continue

        # Parse "N Card Name" format
        match = re.match(r"^(\d+)\s+(.+)$", line)
        if not match:
            continue

        count = int(match.group(1))
        card_name = to_snake_case(match.group(2).strip())

        for _ in range(count):
            if in_sideboard:
                sideboard.append(card_name)
            else:
                mainboard.append(card_name)

    return DeckList(
        mainboard=mainboard,
        sideboard=sideboard,
        commander=commander_snake,
    )
