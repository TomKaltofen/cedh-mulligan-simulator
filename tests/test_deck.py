"""Tests for the Deck data structure."""

from card_registries.mono.black.braids import BRAIDS_REGISTRY
from cedh_mulligan_simulator.deck import Deck


def test_deck_size() -> None:
    """Deck should have exactly deck_size cards in the library."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    assert deck.library_size == 99


def test_deck_has_registry_cards() -> None:
    """All non-filler cards in library should come from the registry."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    lib = deck.library
    non_filler = [c for c in lib if c != "filler"]
    for card in non_filler:
        assert card in BRAIDS_REGISTRY


def test_draw_moves_cards_to_hand() -> None:
    """Drawing n cards should move them from library to hand."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    drawn = deck.draw(7)
    assert len(drawn) == 7
    assert len(deck.hand) == 7
    assert deck.library_size == 92


def test_draw_returns_top_cards() -> None:
    """draw() should return the top n cards of the library in order."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    top_7 = deck.library[:7]
    drawn = deck.draw(7)
    assert drawn == top_7


def test_singleton_no_duplicate_non_filler() -> None:
    """Library should contain no duplicate non-filler cards."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    lib = deck.library
    non_filler = [c for c in lib if c != "filler"]
    assert len(non_filler) == len(set(non_filler))


def test_mulligan_reshuffles() -> None:
    """After mulligan, hand should be empty and library restored to full size."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    deck.draw(7)
    assert deck.library_size == 92
    deck.mulligan()
    assert len(deck.hand) == 0
    assert deck.library_size == 99


def test_mulligan_cycle() -> None:
    """Multiple mulligan + draw cycles should preserve total card count."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    for _ in range(3):
        deck.draw(7)
        assert deck.library_size == 92
        deck.mulligan()
        assert deck.library_size == 99


def test_search_finds_card() -> None:
    """search() should find and remove a named card from the library."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    result = deck.search("swamp")
    assert result == "swamp"
    assert "swamp" not in deck.library


def test_search_missing_card_returns_none() -> None:
    """search() should return None for a card not in the library."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    result = deck.search("nonexistent_card")
    assert result is None


def test_search_reduces_library_size() -> None:
    """search() should reduce the library by 1 when it finds the card."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    original_size = deck.library_size
    deck.search("swamp")
    assert deck.library_size == original_size - 1


def test_put_on_top() -> None:
    """put_on_top() should place a card at index 0 of the library."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    deck.put_on_top("vampiric_tutor_target")
    assert deck.library[0] == "vampiric_tutor_target"
    assert deck.library_size == 100


def test_search_then_put_on_top() -> None:
    """Searching a card and putting it on top should make it the next draw."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    card = deck.search("dark_ritual")
    assert card is not None
    deck.put_on_top(card)
    assert deck.library[0] == "dark_ritual"
    assert deck.library_size == 99


def test_exile_hand_clears_hand() -> None:
    """exile_hand() should leave the hand empty."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    deck.draw(7)
    assert len(deck.hand) == 7
    deck.exile_hand()
    assert deck.hand == []


def test_exile_hand_does_not_restore_library() -> None:
    """exile_hand() should NOT return cards to the library — they are permanently removed."""
    deck = Deck(BRAIDS_REGISTRY, deck_size=99)
    deck.draw(7)
    assert deck.library_size == 92
    deck.exile_hand()
    assert deck.library_size == 92  # still 92, not 99


def test_exile_hand_vs_mulligan() -> None:
    """mulligan() restores library to full size; exile_hand() does not."""
    deck_exile = Deck(BRAIDS_REGISTRY, deck_size=99)
    deck_exile.draw(7)
    deck_exile.exile_hand()
    assert deck_exile.library_size == 92

    deck_mull = Deck(BRAIDS_REGISTRY, deck_size=99)
    deck_mull.draw(7)
    deck_mull.mulligan()
    assert deck_mull.library_size == 99
