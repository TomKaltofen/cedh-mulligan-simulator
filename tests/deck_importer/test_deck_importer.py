"""Tests for the deck_importer package."""

import json
import os
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from deck_importer.builder import build_deck_registry, scan_card_database
from deck_importer.mapper import map_card, parse_mana_cost
from deck_importer.parser import DeckList, parse_deck_list, to_snake_case
from deck_importer.scryfall import fetch_card
from tests.helpers import build_hand, evaluate_t1


# ── Parser tests ────────────────────────────────────────────────────────────


def test_to_snake_case_basic() -> None:
    assert to_snake_case("Dark Ritual") == "dark_ritual"


def test_to_snake_case_apostrophe() -> None:
    assert to_snake_case("Urza's Saga") == "urzas_saga"


def test_to_snake_case_comma_space() -> None:
    assert to_snake_case("Braids, Arisen Nightmare") == "braids_arisen_nightmare"


def test_to_snake_case_hyphen() -> None:
    assert to_snake_case("Snow-Covered Swamp") == "snow_covered_swamp"


def test_to_snake_case_all_caps() -> None:
    assert to_snake_case("SOL RING") == "sol_ring"


def test_parse_deck_list_simple() -> None:
    text = "1 Dark Ritual\n1 Swamp\n"
    dl = parse_deck_list(text)
    assert "dark_ritual" in dl.mainboard
    assert "swamp" in dl.mainboard
    assert len(dl.mainboard) == 2


def test_parse_deck_list_multi_count() -> None:
    text = "3 Swamp\n"
    dl = parse_deck_list(text)
    assert dl.mainboard.count("swamp") == 3


def test_parse_deck_list_sideboard() -> None:
    text = "1 Dark Ritual\nSIDEBOARD:\n1 Dauthi Voidwalker\n"
    dl = parse_deck_list(text)
    assert "dark_ritual" in dl.mainboard
    assert "dauthi_voidwalker" in dl.sideboard
    assert "dauthi_voidwalker" not in dl.mainboard


def test_parse_deck_list_commander_normalised() -> None:
    dl = parse_deck_list("1 Dark Ritual\n", commander="Braids, Arisen Nightmare")
    assert dl.commander == "braids_arisen_nightmare"


def test_parse_deck_list_ignores_blank_lines() -> None:
    text = "\n1 Dark Ritual\n\n1 Swamp\n\n"
    dl = parse_deck_list(text)
    assert len(dl.mainboard) == 2


def test_parse_deck_list_sideboard_lowercase_marker() -> None:
    text = "1 Dark Ritual\nsideboard:\n1 Dauthi Voidwalker\n"
    dl = parse_deck_list(text)
    assert "dark_ritual" in dl.mainboard
    assert "dauthi_voidwalker" in dl.sideboard


# ── Mana cost parser tests ──────────────────────────────────────────────────


def test_parse_mana_cost_single_black() -> None:
    m = parse_mana_cost("{B}")
    assert m.total == 1
    assert m.black == 1
    assert m.white == 0


def test_parse_mana_cost_generic_plus_black() -> None:
    m = parse_mana_cost("{1}{B}")
    assert m.total == 2
    assert m.black == 1


def test_parse_mana_cost_double_black() -> None:
    m = parse_mana_cost("{B}{B}")
    assert m.total == 2
    assert m.black == 2


def test_parse_mana_cost_generic_only() -> None:
    m = parse_mana_cost("{3}")
    assert m.total == 3
    assert m.black == 0


def test_parse_mana_cost_variable_x() -> None:
    m = parse_mana_cost("{X}")
    assert m.total == 0


def test_parse_mana_cost_zero() -> None:
    m = parse_mana_cost("{0}")
    assert m.total == 0


def test_parse_mana_cost_multicolour() -> None:
    m = parse_mana_cost("{2}{B}{B}")
    assert m.total == 4
    assert m.black == 2


# ── Mapper tests ────────────────────────────────────────────────────────────


def _make_scryfall_land(oracle: str) -> Dict[str, Any]:
    return {
        "name": "Test Land",
        "mana_cost": "",
        "cmc": 0,
        "type_line": "Land",
        "oracle_text": oracle,
    }


def _make_scryfall_creature(mana_cost: str, cmc: int, oracle: str = "") -> Dict[str, Any]:
    return {
        "name": "Test Creature",
        "mana_cost": mana_cost,
        "cmc": cmc,
        "type_line": "Creature — Human",
        "oracle_text": oracle,
    }


def _make_scryfall_sorcery(mana_cost: str, cmc: int, oracle: str) -> Dict[str, Any]:
    return {
        "name": "Test Ritual",
        "mana_cost": mana_cost,
        "cmc": cmc,
        "type_line": "Sorcery",
        "oracle_text": oracle,
    }


def test_map_card_land_untapped() -> None:
    data = _make_scryfall_land("{T}: Add {B}.")
    mc = map_card(data, "test_land")
    assert mc.card.type == "land"
    assert mc.card.t1_mana is not None and mc.card.t1_mana.black == 1
    assert mc.card.t2_mana is not None and mc.card.t2_mana.black == 1


def test_map_card_land_enters_tapped() -> None:
    data = _make_scryfall_land("Test Land enters the battlefield tapped.\n{T}: Add {B}.")
    mc = map_card(data, "test_tapland")
    assert mc.card.type == "land"
    assert mc.card.t1_mana is not None and mc.card.t1_mana.total == 0
    assert mc.card.t2_mana is not None and mc.card.t2_mana.black == 1


def test_map_card_ritual() -> None:
    data = _make_scryfall_sorcery("{B}", 1, "Add {B}{B}{B}.")
    mc = map_card(data, "dark_ritual")
    assert mc.card.type == "ritual"
    assert mc.card.produces is not None
    assert mc.card.produces.black == 3


def test_map_card_creature_flagged_for_review() -> None:
    data = _make_scryfall_creature("{3}{B}", 4, "Flying.")
    mc = map_card(data, "test_creature")
    assert mc.card.type == "creature"
    assert mc.needs_review is True


def test_map_card_tutor_hand_destination() -> None:
    data: Dict[str, Any] = {
        "name": "Demonic Tutor",
        "mana_cost": "{1}{B}",
        "cmc": 2,
        "type_line": "Sorcery",
        "oracle_text": "Search your library for a card, put it into your hand, then shuffle.",
    }
    mc = map_card(data, "demonic_tutor")
    assert mc.card.type == "tutor"
    assert mc.card.destination == "hand"


# ── Builder tests ───────────────────────────────────────────────────────────


def test_scan_card_database_finds_dark_ritual() -> None:
    lookup = scan_card_database()
    assert "dark_ritual" in lookup
    var_name, module_path = lookup["dark_ritual"]
    assert var_name == "DARK_RITUAL"
    assert "black" in module_path


def test_scan_card_database_finds_swamp() -> None:
    lookup = scan_card_database()
    assert "swamp" in lookup
    var_name, module_path = lookup["swamp"]
    assert var_name == "SWAMP"
    assert "lands" in module_path


def test_scan_card_database_finds_sol_ring() -> None:
    lookup = scan_card_database()
    assert "sol_ring" in lookup


def test_build_deck_registry_known_card() -> None:
    deck_list = DeckList(mainboard=["dark_ritual", "swamp"])
    entries = build_deck_registry(deck_list)
    by_name = {e.snake_name: e for e in entries}
    assert "dark_ritual" in by_name
    dr = by_name["dark_ritual"]
    assert dr.source_module is not None
    assert dr.var_name == "DARK_RITUAL"
    assert dr.needs_review is False


def test_build_deck_registry_unknown_card_calls_scryfall() -> None:
    """Unknown card triggers Scryfall fetch."""
    fake_scryfall: Dict[str, Any] = {
        "name": "Bogus Card",
        "mana_cost": "{2}{B}",
        "cmc": 3,
        "type_line": "Creature — Human",
        "oracle_text": "Flying.",
    }
    deck_list = DeckList(mainboard=["bogus_card_not_in_db"])

    with patch("deck_importer.builder.fetch_card") as mock_fetch:
        mock_fetch.return_value = fake_scryfall
        entries = build_deck_registry(deck_list)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.source_module is None
    assert entry.needs_review is True
    mock_fetch.assert_called_once()


def test_build_deck_registry_deduplicates() -> None:
    """Duplicate mainboard entries (multi-count) resolve to one entry."""
    deck_list = DeckList(mainboard=["swamp", "swamp", "dark_ritual"])
    entries = build_deck_registry(deck_list)
    names = [e.snake_name for e in entries]
    assert names.count("swamp") == 1


# ── Scryfall cache tests ────────────────────────────────────────────────────


def test_fetch_card_reads_cache() -> None:
    """fetch_card returns cached data without making an API call."""
    cached_data: Dict[str, Any] = {"name": "Dark Ritual", "mana_cost": "{B}", "cmc": 1}

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_key = "dark ritual"
        cache_path = os.path.join(tmpdir, f"{cache_key}.json")
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(cached_data, fh)

        # Patch the CACHE_DIR so our temp dir is used
        with patch("deck_importer.scryfall.CACHE_DIR", tmpdir):
            with patch("deck_importer.scryfall.urllib.request.urlopen") as mock_open:
                result = fetch_card("Dark Ritual")
                mock_open.assert_not_called()

    assert result is not None
    assert result["name"] == "Dark Ritual"


def test_fetch_card_returns_none_on_404() -> None:
    """fetch_card returns None when Scryfall returns 404 for both exact and fuzzy."""
    import urllib.error

    http_error = urllib.error.HTTPError(url="", code=404, msg="Not Found", hdrs=MagicMock(), fp=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("deck_importer.scryfall.CACHE_DIR", tmpdir):
            with patch("deck_importer.scryfall.urllib.request.urlopen", side_effect=http_error):
                result = fetch_card("Nonexistent Card XYZ")

    assert result is None


# ── Test helpers tests ──────────────────────────────────────────────────────


def test_build_hand_basic() -> None:
    hand = build_hand(BRAIDS_REGISTRY, ["Dark Ritual", "Swamp"])
    assert "dark_ritual" in hand
    assert "swamp" in hand
    assert len(hand) == 7  # padded to 7


def test_build_hand_invalid_card_raises() -> None:
    import pytest

    with pytest.raises(KeyError):
        build_hand(BRAIDS_REGISTRY, ["This Card Does Not Exist"])


def test_build_hand_exact_seven() -> None:
    names = ["Dark Ritual", "Swamp", "Chrome Mox", "Sol Ring", "Ancient Tomb", "Demonic Tutor", "Reanimate"]
    hand = build_hand(BRAIDS_REGISTRY, names)
    assert len(hand) == 7
    assert "filler" not in hand


def test_evaluate_t1_swamp_dark_ritual() -> None:
    """Swamp + Dark Ritual should cast Braids T1."""
    hand = build_hand(BRAIDS_REGISTRY, ["Swamp", "Dark Ritual"])
    assert evaluate_t1(hand, BRAIDS_REGISTRY, BRAIDS_COST) is True


def test_evaluate_t1_single_swamp_fails() -> None:
    """Single swamp cannot cast Braids T1 (1BB needs 3 mana including 2 black)."""
    hand = build_hand(BRAIDS_REGISTRY, ["Swamp"])
    assert evaluate_t1(hand, BRAIDS_REGISTRY, BRAIDS_COST) is False


def test_evaluate_t1_swamp_petal_chromemox() -> None:
    """Swamp + Lotus Petal + Chrome Mox (pitching filler) = 3B, enough for 1BB."""
    hand = build_hand(BRAIDS_REGISTRY, ["Swamp", "Lotus Petal", "Chrome Mox"])
    assert evaluate_t1(hand, BRAIDS_REGISTRY, BRAIDS_COST) is True
