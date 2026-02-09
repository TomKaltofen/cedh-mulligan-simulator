"""Integration tests for specific hands using simulate_turn() directly."""

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.mana import simulate_turn


def test_swamp_dark_ritual_t1_castable() -> None:
    """Swamp + Dark Ritual should cast Braids T1."""
    hand = ["swamp", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]

    t1 = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)

    assert t1.can_cast_commander is True
    assert t1.land_played == "swamp"
    assert "dark_ritual" in t1.cards_played
    assert "swamp" in t1.state_after.battlefield.lands
    assert "dark_ritual" in t1.state_after.graveyard
    assert len(t1.state_after.hand) == 5  # 7 - swamp - dark_ritual = 5 fillers


def test_peat_bog_hand_not_t1_castable() -> None:
    """Peat Bog hand has no T1 mana (enters tapped), but the land is still played for T2 value."""
    hand = [
        "peat_bog",
        "imperial_seal",
        "rain_of_filth",
        "word_of_command",
        "reanimate",
        "corpse_dance",
        "deadly_rollick",
    ]

    t1 = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)

    assert t1.can_cast_commander is False
    # Peat Bog produces no T1 mana, but is still played so it can produce mana on T2
    assert t1.land_played == "peat_bog"
    assert "peat_bog" in t1.state_after.battlefield.lands


def test_peat_bog_draw_swamp_t2() -> None:
    """Peat Bog played T1 + draw Swamp on T2 gives 3B (exactly enough for Braids)."""
    hand = [
        "peat_bog",
        "imperial_seal",
        "rain_of_filth",
        "word_of_command",
        "reanimate",
        "corpse_dance",
        "deadly_rollick",
    ]

    t1 = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)
    t2 = simulate_turn(2, t1.state_after.hand, "swamp", t1.state_after, BRAIDS_REGISTRY, BRAIDS_COST)

    # T1: Peat Bog is played (enters tapped, 0 mana)
    # T2: Peat Bog produces 2B (t2_mana), Swamp is drawn and played for 1B = 3B total
    assert t2.can_cast_commander is True
    assert t2.land_played == "swamp"
    assert "swamp" in t2.state_after.battlefield.lands
    assert "peat_bog" in t2.state_after.battlefield.lands


def test_peat_bog_draw_dark_ritual_t2() -> None:
    """Peat Bog played T1 + draw Dark Ritual on T2 gives 4B (enough for Braids)."""
    hand = [
        "peat_bog",
        "imperial_seal",
        "rain_of_filth",
        "word_of_command",
        "reanimate",
        "corpse_dance",
        "deadly_rollick",
    ]

    t1 = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)
    t2 = simulate_turn(2, t1.state_after.hand, "dark_ritual", t1.state_after, BRAIDS_REGISTRY, BRAIDS_COST)

    # T1: Peat Bog is played (enters tapped, 0 mana)
    # T2: Peat Bog produces 2B, Dark Ritual is drawn and cast (1B -> 3B) = 4B total
    assert t2.can_cast_commander is True


def test_t2_chaining_preserves_zones() -> None:
    """T2 should preserve T1's graveyard and battlefield."""
    hand = ["swamp", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]

    t1 = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)

    # Verify T1 state
    assert "swamp" in t1.state_after.battlefield.lands
    assert "dark_ritual" in t1.state_after.graveyard

    # T2 with filler draw
    t2 = simulate_turn(2, t1.state_after.hand, "filler", t1.state_after, BRAIDS_REGISTRY, BRAIDS_COST)

    # T2 should preserve T1 zones
    assert "swamp" in t2.state_before.battlefield.lands
    assert "swamp" in t2.state_after.battlefield.lands
    assert "dark_ritual" in t2.state_before.graveyard
    assert "dark_ritual" in t2.state_after.graveyard


def test_chrome_mox_exile_tracking() -> None:
    """Chrome Mox should track exiled card."""
    hand = ["swamp", "chrome_mox", "dark_ritual", "filler", "filler", "filler", "filler"]

    t1 = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)

    # Should be castable with swamp + chrome_mox (pitch filler) + dark_ritual
    assert t1.can_cast_commander is True

    # Chrome Mox should have exiled something
    if "chrome_mox" in t1.state_after.battlefield.artifacts:
        assert len(t1.state_after.exile) >= 1
        # The exiled card count should decrease by 1 in hand
        for exiled in t1.state_after.exile:
            original_count = hand.count(exiled)
            final_count = t1.state_after.hand.count(exiled)
            assert final_count < original_count


def test_gemstone_caverns_exile_tracking() -> None:
    """Gemstone Caverns should track exiled card."""
    hand = ["gemstone_caverns", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]

    t1 = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)

    if t1.land_played == "gemstone_caverns":
        # Should be castable: gemstone_caverns (any color) + dark_ritual
        assert t1.can_cast_commander is True
        # Should have exiled a card
        assert len(t1.state_after.exile) >= 1
        # The exiled card count should decrease by 1 in hand
        for exiled in t1.state_after.exile:
            original_count = hand.count(exiled)
            final_count = t1.state_after.hand.count(exiled)
            assert final_count < original_count


def test_sol_ring_t2_accumulation() -> None:
    """T2 should use T1 board mana from Sol Ring."""
    hand = ["swamp", "sol_ring", "filler", "filler", "filler", "filler", "filler"]

    t1 = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)
    # T1: swamp (B) + sol_ring (2) = 3 mana, but only 1B (need 2B)
    assert t1.can_cast_commander is False
    assert "swamp" in t1.state_after.battlefield.lands
    assert "sol_ring" in t1.state_after.battlefield.artifacts

    # T2: Draw second swamp
    t2 = simulate_turn(2, t1.state_after.hand, "swamp_2", t1.state_after, BRAIDS_REGISTRY, BRAIDS_COST)
    # T2: swamp (B) + sol_ring (2) + swamp_2 (B) = 4 mana, 2B - enough!
    assert t2.can_cast_commander is True
    assert "swamp" in t2.state_after.battlefield.lands
    assert "swamp_2" in t2.state_after.battlefield.lands
    assert "sol_ring" in t2.state_after.battlefield.artifacts


def test_lotus_petal_to_graveyard() -> None:
    """Lotus Petal should go to graveyard after use."""
    hand = ["swamp", "lotus_petal", "chrome_mox", "filler", "filler", "filler", "filler"]

    t1 = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)

    # This should be castable: swamp (B) + lotus_petal (any) + chrome_mox (any) = 3 mana, 1B+
    assert t1.can_cast_commander is True

    # Lotus Petal should be in graveyard (sacrifices_on_use)
    if "lotus_petal" in t1.cards_played:
        assert "lotus_petal" in t1.state_after.graveyard
        assert "lotus_petal" not in t1.state_after.battlefield.artifacts


def test_state_before_matches_previous_state_after() -> None:
    """T2.state_before should match T1.state_after."""
    hand = ["swamp", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]

    t1 = simulate_turn(1, hand, None, None, BRAIDS_REGISTRY, BRAIDS_COST)
    t2 = simulate_turn(2, t1.state_after.hand, "filler", t1.state_after, BRAIDS_REGISTRY, BRAIDS_COST)

    # state_before should reflect T1's end state (plus the drawn card in hand)
    assert t2.state_before.battlefield.lands == t1.state_after.battlefield.lands
    assert t2.state_before.battlefield.artifacts == t1.state_after.battlefield.artifacts
    assert t2.state_before.graveyard == t1.state_after.graveyard
    assert t2.state_before.exile == t1.state_after.exile
    # Hand should be T1 hand + drawn card
    assert "filler" in t2.state_before.hand  # drawn card
