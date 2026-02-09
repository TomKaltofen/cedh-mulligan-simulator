"""Tests for card_registry module."""

from card_database.lands import SWAMP
from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.card_registry import (
    DEFAULT_CARD_REGISTRY,
    WASTES,
    Card,
    Mana,
    ManaRequirement,
    any_color,
    build_registry,
)


# ── Generic tests ──


def test_mana_dataclass_defaults() -> None:
    m = Mana()
    assert m.total == 0
    assert m.white == 0
    assert m.blue == 0
    assert m.black == 0
    assert m.red == 0
    assert m.green == 0


def test_mana_dataclass_fields() -> None:
    m = Mana(5, white=1, blue=1, black=1, red=1, green=1)
    assert m.total == 5
    assert m.white == 1
    assert m.black == 1


def test_mana_requirement_defaults() -> None:
    req = ManaRequirement()
    assert req.total == 0
    assert req.black == 0


def test_mana_requirement_fields() -> None:
    req = ManaRequirement(total=3, black=2)
    assert req.total == 3
    assert req.black == 2


def test_wastes() -> None:
    assert WASTES.type == "land"
    assert WASTES.t1_mana == Mana(1)
    assert WASTES.t2_mana == Mana(1)


def test_default_registry_has_wastes() -> None:
    assert "wastes" in DEFAULT_CARD_REGISTRY
    assert len(DEFAULT_CARD_REGISTRY) == 1


def test_card_dataclass_frozen() -> None:
    card = Card(name="test", type="land")
    try:
        card.name = "other"  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_any_color() -> None:
    m = any_color(3)
    assert m.total == 3
    assert m.white == 3
    assert m.blue == 3
    assert m.black == 3
    assert m.red == 3
    assert m.green == 3


def test_any_color_one() -> None:
    m = any_color(1)
    assert m == Mana(1, white=1, blue=1, black=1, red=1, green=1)


def test_build_registry() -> None:
    c1 = Card(name="alpha", type="land")
    c2 = Card(name="beta", type="artifact")
    reg = build_registry(c1, c2)
    assert reg["alpha"] is c1
    assert reg["beta"] is c2
    assert len(reg) == 2


# ── Braids-specific tests ──


def test_braids_cost() -> None:
    assert BRAIDS_COST.total == 3
    assert BRAIDS_COST.black == 2


def test_braids_registry_has_all_card_types() -> None:
    types = {v.type for v in BRAIDS_REGISTRY.values()}
    assert "land" in types
    assert "ritual" in types
    assert "artifact" in types
    assert "creature" in types
    assert "sacrifice_outlet" in types
    assert "equipment" in types
    assert "tutor" in types


def test_braids_registry_card_count() -> None:
    assert len(BRAIDS_REGISTRY) >= 24


def test_braids_swamp() -> None:
    assert SWAMP.type == "land"
    assert SWAMP.t1_mana == Mana(1, black=1)


def test_braids_lands_have_mana() -> None:
    for name, card in BRAIDS_REGISTRY.items():
        if card.type == "land":
            assert card.t1_mana is not None, f"{name} missing t1_mana"
            assert card.t2_mana is not None, f"{name} missing t2_mana"


def test_braids_rituals_have_cost_and_produces() -> None:
    for name, card in BRAIDS_REGISTRY.items():
        if card.type == "ritual":
            assert card.cost is not None, f"{name} missing cost"
            assert card.produces is not None, f"{name} missing produces"
