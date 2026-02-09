"""Tests for mana line resolution engine."""

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.card_registry import ManaRequirement
from cedh_mulligan_simulator.mana import simulate_turn


def _can_cast_t1(hand: list[str]) -> bool:
    """Helper: check if hand can cast commander on T1."""
    return simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST).can_cast_commander


def _can_cast_t2(hand: list[str], draw: str = "filler") -> bool:
    """Helper: check if hand can cast commander on T2."""
    tr1 = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)
    if tr1.can_cast_commander:
        return True
    tr2 = simulate_turn(2, tr1.state_after.hand, draw, tr1.state_after, BRAIDS_REGISTRY, BRAIDS_COST)
    return tr2.can_cast_commander


# ── T1 Tests ──
def test_t1_swamp() -> None:
    """Single swamp = B, not enough for 1BB."""

    example_list = [
        "SNUFF_OUT",
        "TOXIC_DELUGE",
        "cut_down",
        "deadly_rollick",
        "dismember",
        "fatal_push",
    ]

    # 1 card
    assert _can_cast_t1(["swamp"] + example_list[1:]) is False

    # 2 cards
    assert _can_cast_t1(["swamp", "dark_ritual"] + example_list[2:])
    assert _can_cast_t1(["swamp", "lotus_petal"] + example_list[2:]) is False
    assert _can_cast_t1(["swamp", "sol_ring"] + example_list[2:]) is False

    # 3 cards
    assert _can_cast_t1(["swamp", "lotus_petal", "chrome_mox"] + example_list[3:]) is True
    # Lotus Petal now doesn't produce mana when played (stays on board for later sacrifice)
    # T1: swamp(1B) → mana_vault(spend 1B, get 3C) → play lotus_petal (0 mana) = 3C, need 2B
    assert _can_cast_t1(["swamp", "mana_vault", "lotus_petal"] + example_list[3:]) is False
    # T2: swamp(1B) + mana_vault(3C) + sac lotus_petal(1B) = 2B + 3C = enough for 1BB
    assert _can_cast_t2(["swamp", "mana_vault", "lotus_petal"] + example_list[3:]) is True
    assert _can_cast_t1(["swamp", "swamp", "chrome_mox"] + example_list[3:]) is False
    # Need to check chrome_mox logic


# ── T2 Tests ──


def test_t2_with_ritual_draw() -> None:
    """Drawing dark_ritual on T2 enables cast."""
    hand = ["swamp", "filler", "filler", "filler", "filler", "filler", "filler"]
    # T1: swamp gives 1B, not enough. T2: draw dark_ritual, cast it for BBB. Enough!
    assert _can_cast_t2(hand, draw="dark_ritual") is True


def test_t2_single_swamp_fails() -> None:
    """Single swamp T2 = (1,1), not enough."""
    hand = ["swamp", "filler", "filler", "filler", "filler", "filler", "filler"]
    assert _can_cast_t2(hand) is False


def test_t2_peat_bog_alone() -> None:
    """Peat Bog T2 = (2,2), not enough for (3,2)."""
    hand = ["peat_bog", "filler", "filler", "filler", "filler", "filler", "filler"]
    assert _can_cast_t2(hand) is False


def test_t2_swamp_dark_ritual() -> None:
    """T2: Swamp(1,1) + Dark Ritual costs (1,1) -> (0,0) + (3,3) -> (3,3). Enough!"""
    hand = ["swamp", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]
    assert _can_cast_t2(hand) is True


# ── Board State Tests ──


def test_non_castable_still_deploys() -> None:
    """If hand isn't T1 castable, still deploys best effort resources."""
    hand = ["swamp", "sol_ring", "filler", "filler", "filler", "filler", "filler"]
    tr = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)
    # Swamp + Sol Ring = 3 total but only 1 black. Not enough.
    assert tr.can_cast_commander is False
    # Best effort still plays the land and artifact
    assert "swamp" in tr.state_after.battlefield.lands
    assert "sol_ring" in tr.state_after.battlefield.artifacts


# ── Custom ManaRequirement Tests ──


def test_custom_cost() -> None:
    """Test with a cheaper commander cost."""
    cheap = ManaRequirement(total=1, black=1)
    hand = ["swamp", "filler", "filler", "filler", "filler", "filler", "filler"]
    tr = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, cheap)
    assert tr.can_cast_commander is True


def test_expensive_cost_fails() -> None:
    """Test with an expensive commander."""
    expensive = ManaRequirement(total=5, black=3)
    hand = ["swamp", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]
    # Swamp(1,1) + DR: (1,1)-(1,1)=>(0,0)+(3,3)=>(3,3). Total=3, need 5. Fails.
    tr = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, expensive)
    assert tr.can_cast_commander is False
