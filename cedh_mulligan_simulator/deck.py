"""Deck: shuffled singleton deck with draw, mulligan, and library access."""

import random
from typing import List, Optional

from cedh_mulligan_simulator.card_registry import CardRegistry


class Deck:
    """A shuffled singleton deck that tracks hand and library state."""

    def __init__(self, registry: CardRegistry, deck_size: int = 99) -> None:
        cards = list(registry.keys())
        if len(cards) > deck_size:
            # Randomly select deck_size cards from registry
            cards = random.sample(cards, deck_size)  # nosec B311
        else:
            cards = cards + ["filler"] * (deck_size - len(cards))
        random.shuffle(cards)  # nosec B311
        self._library: List[str] = cards
        self._hand: List[str] = []

    @property
    def hand(self) -> List[str]:
        """Return a copy of the current hand."""
        return list(self._hand)

    @property
    def library(self) -> List[str]:
        """Return a copy of the remaining library."""
        return list(self._library)

    @property
    def library_size(self) -> int:
        """Number of cards remaining in the library."""
        return len(self._library)

    def draw(self, n: int) -> List[str]:
        """Draw n cards from the top of the library into the hand."""
        drawn = self._library[:n]
        self._library = self._library[n:]
        self._hand.extend(drawn)
        return drawn

    def mulligan(self) -> None:
        """Shuffle hand back into library and reshuffle."""
        self._library.extend(self._hand)
        self._hand = []
        random.shuffle(self._library)  # nosec B311

    def search(self, card_name: str) -> Optional[str]:
        """Find and remove a named card from the library. Returns the card or None."""
        if card_name not in self._library:
            return None
        self._library.remove(card_name)
        return card_name

    def exile_hand(self) -> None:
        """Exile all cards from hand. Cards do not return to the library."""
        self._hand = []

    def put_on_top(self, card: str) -> None:
        """Place a card on top of the library."""
        self._library.insert(0, card)
