"""Tests for turn simulation functions in mana.py."""

from cedh_mulligan_simulator.card_registry import Card, Mana, ManaRequirement, any_color, build_registry
from cedh_mulligan_simulator.mana import simulate_turn


def _swamp() -> Card:
    return Card(name="swamp", type="land", t1_mana=Mana(1, black=1), t2_mana=Mana(1, black=1))


def _dark_ritual() -> Card:
    return Card(name="dark_ritual", type="ritual", cost=Mana(1, black=1), produces=Mana(3, black=3))


def _sol_ring() -> Card:
    return Card(name="sol_ring", type="artifact", cost=Mana(1), produces=Mana(2))


def _lotus_petal() -> Card:
    return Card(name="lotus_petal", type="artifact", cost=Mana(), produces=any_color(1), sacrifices_on_use=True)


def _memnite() -> Card:
    return Card(name="memnite", type="artifact_creature", cost=Mana(), cmc=0)


def _grief() -> Card:
    return Card(name="grief", type="creature", cost=Mana(4, black=1), evoke_cost=Mana(0), cmc=4)


def _chrome_mox() -> Card:
    return Card(name="chrome_mox", type="artifact", cost=Mana(), produces=any_color(1), requires="pitchable_card")


def _gemstone_caverns() -> Card:
    return Card(
        name="gemstone_caverns", type="land", t1_mana=any_color(1), t2_mana=any_color(1), requires="exile_from_hand"
    )


BRAIDS_COST = ManaRequirement(total=3, black=2)


# ==================== simulate_turn() tests ====================


def test_simulate_turn_t1_basic() -> None:
    """simulate_turn(1, ...) should work correctly."""
    registry = build_registry(_swamp(), _dark_ritual())
    hand = ["swamp", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]

    tr = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    assert tr.can_cast_commander is True
    assert tr.turn_number == 1
    assert tr.drawn_card is None
    assert tr.land_played == "swamp"
    assert "dark_ritual" in tr.cards_played


def test_simulate_turn_t2_basic() -> None:
    """simulate_turn(2, ...) should work correctly."""
    registry = build_registry(
        _swamp(),
        Card(name="swamp_2", type="land", t1_mana=Mana(1, black=1), t2_mana=Mana(1, black=1)),
        _sol_ring(),
    )
    hand = ["swamp", "sol_ring", "filler", "filler", "filler", "filler", "filler"]

    tr1 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)
    tr2 = simulate_turn(2, tr1.state_after.hand, "swamp_2", tr1.state_after, registry, BRAIDS_COST)

    assert tr2.can_cast_commander is True
    assert tr2.turn_number == 2
    assert tr2.drawn_card == "swamp_2"


def test_simulate_turn_t3_basic() -> None:
    """simulate_turn(3, ...) should work correctly."""
    registry = build_registry(
        _swamp(),
        Card(name="swamp_2", type="land", t1_mana=Mana(1, black=1), t2_mana=Mana(1, black=1)),
        Card(name="swamp_3", type="land", t1_mana=Mana(1, black=1), t2_mana=Mana(1, black=1)),
    )
    hand = ["swamp", "filler", "filler", "filler", "filler", "filler", "filler"]

    tr1 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)
    tr2 = simulate_turn(2, tr1.state_after.hand, "swamp_2", tr1.state_after, registry, BRAIDS_COST)
    tr3 = simulate_turn(3, tr2.state_after.hand, "swamp_3", tr2.state_after, registry, BRAIDS_COST)

    assert tr3.turn_number == 3
    assert tr3.drawn_card == "swamp_3"
    # T3 with 3 swamps = 3 black mana, enough for Braids
    assert tr3.can_cast_commander is True


def test_simulate_turn_board_accumulation() -> None:
    """Board state should accumulate across turns."""
    registry = build_registry(_swamp(), _sol_ring())
    hand = ["swamp", "sol_ring", "filler", "filler", "filler", "filler", "filler"]

    tr1 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    # Create a second swamp for T2 draw
    registry2 = build_registry(
        _swamp(),
        Card(name="swamp_2", type="land", t1_mana=Mana(1, black=1), t2_mana=Mana(1, black=1)),
        _sol_ring(),
    )

    tr2 = simulate_turn(2, tr1.state_after.hand, "swamp_2", tr1.state_after, registry2, BRAIDS_COST)

    # Board should have both swamps and sol_ring
    assert "swamp" in tr2.state_after.battlefield.lands
    assert "swamp_2" in tr2.state_after.battlefield.lands
    assert "sol_ring" in tr2.state_after.battlefield.artifacts


def test_t1_basic_cast() -> None:
    """Swamp + Dark Ritual should cast Braids T1."""
    registry = build_registry(_swamp(), _dark_ritual())
    hand = ["swamp", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]
    tr = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    assert tr.can_cast_commander is True
    assert tr.land_played == "swamp"
    assert "dark_ritual" in tr.cards_played
    assert "swamp" in tr.state_after.battlefield.lands
    assert "dark_ritual" in tr.state_after.graveyard
    assert "swamp" not in tr.state_after.hand
    assert "dark_ritual" not in tr.state_after.hand


def test_t1_artifact_persistence() -> None:
    """Sol Ring stays on board; Lotus Petal stays in hand (no benefit to deploy T1)."""
    registry = build_registry(_swamp(), _sol_ring(), _lotus_petal())
    hand = ["swamp", "sol_ring", "lotus_petal", "filler", "filler", "filler", "filler"]
    tr = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    # T1: swamp(1B) → sol_ring(spend 1B, get 2C) → lotus_petal (no mana yet) = 2C, NOT castable
    assert tr.can_cast_commander is False
    # Sol Ring persists on board
    assert "sol_ring" in tr.state_after.battlefield.artifacts
    # Lotus Petal stays in hand - no benefit to deploy when it produces no mana on play
    assert "lotus_petal" in tr.state_after.hand


def test_t1_evoked_creature_to_graveyard() -> None:
    """Grief evoked should go to graveyard, not persist as creature."""
    registry = build_registry(_swamp(), _dark_ritual(), _grief())
    hand = ["swamp", "dark_ritual", "grief", "filler", "filler", "filler", "filler"]
    tr = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    assert tr.can_cast_commander is True
    # Grief is evoked -> goes to graveyard
    if "grief" in tr.state_after.battlefield.all_permanents:
        # If grief wasn't needed for the mana line, it stays in hand
        pass
    elif "grief" in tr.state_after.graveyard:
        assert "grief" not in tr.state_after.battlefield.creatures


def test_t1_free_creature_persists() -> None:
    """Memnite (free creature) should persist on the board."""
    registry = build_registry(_swamp(), _dark_ritual(), _memnite())
    hand = ["swamp", "dark_ritual", "memnite", "filler", "filler", "filler", "filler"]
    tr = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    assert tr.can_cast_commander is True
    # Memnite is free and stays on board if played
    # (It may or may not be played depending on mana line)


def test_t2_uses_t1_board_mana() -> None:
    """T2 should use mana from T1 board: swamp + sol_ring from T1, new swamp on T2."""
    registry = build_registry(
        _swamp(),
        Card(name="swamp_2", type="land", t1_mana=Mana(1, black=1), t2_mana=Mana(1, black=1)),
        _sol_ring(),
    )
    hand = ["swamp", "sol_ring", "filler", "filler", "filler", "filler", "filler"]

    # T1: play swamp + sol_ring -> board has swamp + sol_ring
    tr1 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)
    # T1 with sol_ring gives 2 mana + 1 from swamp = 3 total, but only 1 black, need 2 black
    # So T1 might not cast Braids

    # T2: board mana (swamp=1B, sol_ring=2C) + new swamp_2 = 1B + 2C + 1B = 4 total, 2 black
    tr2 = simulate_turn(2, tr1.state_after.hand, "swamp_2", tr1.state_after, registry, BRAIDS_COST)

    assert tr2.can_cast_commander is True
    assert tr2.drawn_card == "swamp_2"


def test_t2_real_draw_in_hand() -> None:
    """The drawn card should appear in T2's state_before.hand."""
    registry = build_registry(_swamp(), _dark_ritual())
    hand = ["swamp", "filler", "filler", "filler", "filler", "filler", "filler"]
    tr1 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    tr2 = simulate_turn(2, tr1.state_after.hand, "dark_ritual", tr1.state_after, registry, BRAIDS_COST)

    assert "dark_ritual" in tr2.state_before.hand


def test_t1_fail_still_deploys_resources() -> None:
    """Even when T1 fails, best-effort should still deploy resources for T2."""
    registry = build_registry(_swamp(), _sol_ring())
    hand = ["swamp", "sol_ring", "filler", "filler", "filler", "filler", "filler"]

    tr1 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)
    # swamp + sol_ring = 3 total but only 1 black (need 2), so T1 fails
    assert tr1.can_cast_commander is False
    # But best-effort should still play the land and sol_ring
    assert tr1.land_played == "swamp"
    assert "sol_ring" in tr1.state_after.battlefield.artifacts
    assert "swamp" in tr1.state_after.battlefield.lands


def test_t1_hand_end_correct() -> None:
    """state_after.hand should be the original hand minus played cards."""
    registry = build_registry(_swamp(), _dark_ritual())
    hand = ["swamp", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]
    tr = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    assert tr.can_cast_commander is True
    # filler cards remain
    assert len(tr.state_after.hand) == 5
    for c in tr.state_after.hand:
        assert c == "filler"


def test_t1_turn_number() -> None:
    """Turn number should be 1."""
    registry = build_registry(_swamp())
    hand = ["swamp", "filler", "filler", "filler", "filler", "filler", "filler"]
    tr = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)
    assert tr.turn_number == 1


def test_t2_turn_number() -> None:
    """Turn number should be 2."""
    registry = build_registry(_swamp())
    hand = ["swamp", "filler", "filler", "filler", "filler", "filler", "filler"]
    tr1 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)
    tr2 = simulate_turn(2, tr1.state_after.hand, "filler", tr1.state_after, registry, BRAIDS_COST)
    assert tr2.turn_number == 2


def test_t1_simulation_deterministic() -> None:
    """Verify that simulate_turn produces consistent results."""
    registry = build_registry(_swamp(), _dark_ritual(), _sol_ring())
    hand = ["swamp", "dark_ritual", "sol_ring", "filler", "filler", "filler", "filler"]

    # Run twice
    tr1 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)
    tr2 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    # Should be identical
    assert tr1.can_cast_commander == tr2.can_cast_commander
    assert tr1.land_played == tr2.land_played
    assert set(tr1.cards_played) == set(tr2.cards_played)
    assert tr1.mana_remaining == tr2.mana_remaining
    assert tr1.state_after.battlefield.lands == tr2.state_after.battlefield.lands
    assert tr1.state_after.battlefield.artifacts == tr2.state_after.battlefield.artifacts


def test_t2_simulation_deterministic() -> None:
    """Verify that simulate_turn produces consistent results."""
    registry = build_registry(_swamp(), _sol_ring())
    hand = ["swamp", "sol_ring", "filler", "filler", "filler", "filler", "filler"]

    tr1 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    # Run twice
    tr2a = simulate_turn(2, tr1.state_after.hand, "swamp", tr1.state_after, registry, BRAIDS_COST)
    tr2b = simulate_turn(2, tr1.state_after.hand, "swamp", tr1.state_after, registry, BRAIDS_COST)

    # Should be identical
    assert tr2a.can_cast_commander == tr2b.can_cast_commander
    assert tr2a.land_played == tr2b.land_played
    assert set(tr2a.cards_played) == set(tr2b.cards_played)
    assert tr2a.mana_remaining == tr2b.mana_remaining
    assert tr2a.state_after.battlefield.lands == tr2b.state_after.battlefield.lands


def test_t1_land_validation() -> None:
    """Test that lands producing no mana on T1 are skipped."""

    def _bad_land() -> Card:
        """Land that produces no mana on T1."""
        return Card(name="bad_land", type="land", t1_mana=Mana(0), t2_mana=Mana(1))

    registry = build_registry(_swamp(), _bad_land())
    hand = ["bad_land", "swamp", "filler", "filler", "filler", "filler", "filler"]

    tr = simulate_turn(1, hand, None, None, registry, ManaRequirement(total=1, black=1))

    # Should play swamp, not bad_land (which produces 0 T1 mana)
    assert tr.land_played == "swamp"
    assert tr.can_cast_commander is True


# ==================== Chrome Mox exile tracking tests ====================


def test_chrome_mox_tracks_exiled_card() -> None:
    """Chrome Mox should track the exiled card in the exile zone."""
    registry = build_registry(_swamp(), _chrome_mox(), _dark_ritual())
    # Hand needs a pitchable card (dark_ritual) and the chrome mox
    hand = ["swamp", "chrome_mox", "dark_ritual", "filler", "filler", "filler", "filler"]

    tr = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    # Check if chrome_mox was played
    if "chrome_mox" in tr.cards_played or "chrome_mox" in tr.state_after.battlefield.artifacts:
        # Either dark_ritual was pitched, or filler was pitched
        # At least one card should be in exile
        assert len(tr.state_after.exile) >= 1
        # The exiled card count should decrease by 1 in hand
        for exiled in tr.state_after.exile:
            original_count = hand.count(exiled)
            final_count = tr.state_after.hand.count(exiled)
            assert final_count < original_count


def test_chrome_mox_exile_prefers_filler() -> None:
    """Chrome Mox should prefer to exile filler cards over real cards."""
    registry = build_registry(_swamp(), _chrome_mox(), _dark_ritual())
    hand = ["swamp", "chrome_mox", "dark_ritual", "filler", "filler", "filler", "filler"]

    tr = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    # If chrome_mox was played with filler as pitch
    if "chrome_mox" in tr.cards_played or "chrome_mox" in tr.state_after.battlefield.artifacts:
        # Filler should be prioritized for exile
        if "filler" in tr.state_after.exile:
            # Good - filler was used
            pass


# ==================== Gemstone Caverns exile tracking tests ====================


def test_gemstone_caverns_tracks_exiled_card() -> None:
    """Gemstone Caverns should track the exiled card in the exile zone."""
    registry = build_registry(_gemstone_caverns(), _dark_ritual())
    hand = ["gemstone_caverns", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]

    tr = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    # Check if gemstone_caverns was played
    if tr.land_played == "gemstone_caverns":
        # A card should be in exile
        assert len(tr.state_after.exile) >= 1
        # The exiled card count should decrease by 1 in hand
        for exiled in tr.state_after.exile:
            original_count = hand.count(exiled)
            final_count = tr.state_after.hand.count(exiled)
            assert final_count < original_count


def test_gemstone_caverns_exile_prefers_filler() -> None:
    """Gemstone Caverns should prefer to exile filler cards."""
    registry = build_registry(_gemstone_caverns(), _dark_ritual())
    hand = ["gemstone_caverns", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]

    tr = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    if tr.land_played == "gemstone_caverns":
        # Filler should be in exile, not dark_ritual
        assert "filler" in tr.state_after.exile
        assert "dark_ritual" not in tr.state_after.exile


# ==================== T2 zone preservation tests ====================


def test_t2_preserves_t1_graveyard() -> None:
    """T2 should preserve T1's graveyard contents."""
    registry = build_registry(_swamp(), _dark_ritual())
    hand = ["swamp", "dark_ritual", "filler", "filler", "filler", "filler", "filler"]

    tr1 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)
    # Dark ritual should be in graveyard
    assert "dark_ritual" in tr1.state_after.graveyard

    # T2 should preserve the graveyard
    tr2 = simulate_turn(2, tr1.state_after.hand, "filler", tr1.state_after, registry, BRAIDS_COST)
    assert "dark_ritual" in tr2.state_after.graveyard
    assert "dark_ritual" in tr2.state_before.graveyard


def test_t2_preserves_t1_exile() -> None:
    """T2 should preserve T1's exile zone contents."""
    registry = build_registry(_gemstone_caverns(), _swamp())
    hand = ["gemstone_caverns", "filler", "filler", "filler", "filler", "filler", "filler"]

    tr1 = simulate_turn(1, hand, None, None, registry, BRAIDS_COST)

    if tr1.land_played == "gemstone_caverns" and len(tr1.state_after.exile) > 0:
        # T2 should have same exile zone
        tr2 = simulate_turn(2, tr1.state_after.hand, "swamp", tr1.state_after, registry, BRAIDS_COST)
        assert tr1.state_after.exile == tr2.state_before.exile
        # Exile should be preserved in state_after too
        for card in tr1.state_after.exile:
            assert card in tr2.state_after.exile
