"""Mana line enumeration engine — commander-agnostic."""

from typing import Callable, List, Optional, Set, Tuple

from cedh_mulligan_simulator.card_registry import Card, CardRegistry, Mana, ManaRequirement
from cedh_mulligan_simulator.turn_result import Battlefield, GameState, TurnResult

_ZERO = Mana()
_UNPLAYABLE = Mana(99, 99, 99, 99, 99, 99)


def _add_mana(a: Mana, b: Mana) -> Mana:
    return Mana(
        a.total + b.total,
        a.white + b.white,
        a.blue + b.blue,
        a.black + b.black,
        a.red + b.red,
        a.green + b.green,
    )


def _sub_mana(pool: Mana, cost: Mana) -> Optional[Mana]:
    """Subtract cost from pool. Returns None if insufficient."""
    new_total = pool.total - cost.total
    new_white = pool.white - cost.white
    new_blue = pool.blue - cost.blue
    new_black = pool.black - cost.black
    new_red = pool.red - cost.red
    new_green = pool.green - cost.green
    if new_total < 0 or new_white < 0 or new_blue < 0 or new_black < 0 or new_red < 0 or new_green < 0:
        return None

    # Colored mana used to pay generic costs must be consumed.
    # Reduce colors in priority order (B > U > R > G > W) until sum <= total.
    colored_sum = new_white + new_blue + new_black + new_red + new_green
    excess = colored_sum - new_total
    if excess > 0:
        for _ in range(excess):
            if new_black > 0:
                new_black -= 1
            elif new_blue > 0:
                new_blue -= 1
            elif new_red > 0:
                new_red -= 1
            elif new_green > 0:
                new_green -= 1
            elif new_white > 0:
                new_white -= 1

    return Mana(new_total, new_white, new_blue, new_black, new_red, new_green)


def _meets_requirement(pool: Mana, target: ManaRequirement) -> bool:
    return (
        pool.total >= target.total
        and pool.white >= target.white
        and pool.blue >= target.blue
        and pool.black >= target.black
        and pool.red >= target.red
        and pool.green >= target.green
    )


def _is_better_pool(new_pool: Mana, current_pool: Mana) -> bool:
    """Compare two mana pools. Returns True if new_pool is strictly better."""
    new_val = (
        new_pool.total,
        new_pool.white + new_pool.blue + new_pool.black + new_pool.red + new_pool.green,
    )
    curr_val = (
        current_pool.total,
        current_pool.white + current_pool.blue + current_pool.black + current_pool.red + current_pool.green,
    )
    return new_val[0] > curr_val[0] or (new_val[0] == curr_val[0] and new_val[1] > curr_val[1])


def _hand_has_swamp(hand: List[str], registry: CardRegistry) -> bool:
    """Check if hand contains a swamp (for Deepwood Legate free condition)."""
    for card_name in hand:
        card_obj = registry.get(card_name)
        if card_obj is None:
            continue
        if card_obj.type == "land":
            t1 = card_obj.t1_mana or _ZERO
            if t1.black > 0:
                return True
    return False


def _count_lands(hand: List[str], registry: CardRegistry) -> int:
    return sum(1 for c in hand if (obj := registry.get(c)) is not None and obj.type == "land")


def _count_artifacts_on_board(played: Set[str], registry: CardRegistry, exclude: Optional[str] = None) -> int:
    return sum(1 for c in played if c != exclude and (obj := registry.get(c)) is not None and "artifact" in obj.type)


def _has_creature_on_board(played: Set[str], registry: CardRegistry) -> bool:
    for c in played:
        card_obj = registry.get(c)
        if card_obj is not None and "creature" in card_obj.type:
            return True
    return False


def _has_pitchable_card(hand: List[str], played: Set[str], registry: CardRegistry) -> bool:
    """Check if hand has a nonland, nonartifact card to exile for Chrome Mox."""
    for card_name in hand:
        if card_name in played:
            continue
        card_obj = registry.get(card_name)
        if (
            card_obj is not None
            and "artifact" not in card_obj.type
            and "land" not in card_obj.type
            and card_obj.type != ""
        ):
            return True
    # Filler cards count as pitchable
    for card_name in hand:
        if card_name not in played and card_name == "filler":
            return True
    return False


def _find_pitchable_card(hand: List[str], played: Set[str], registry: CardRegistry) -> Optional[str]:
    """Find a card to exile for Chrome Mox. Returns the card name or None."""
    # Prefer filler cards first
    for card_name in hand:
        if card_name not in played and card_name == "filler":
            return card_name
    # Then find any nonland, nonartifact card
    for card_name in hand:
        if card_name in played:
            continue
        card_obj = registry.get(card_name)
        if (
            card_obj is not None
            and "artifact" not in card_obj.type
            and "land" not in card_obj.type
            and card_obj.type != ""
        ):
            return card_name
    return None


def _find_exile_card(hand: List[str], played: Set[str], land_played: Optional[str]) -> Optional[str]:
    """Find a card to exile for Gemstone Caverns. Returns the card name or None."""
    # Prefer filler cards first
    for card_name in hand:
        if card_name not in played and card_name != land_played and card_name == "filler":
            return card_name
    # Then find any card that isn't the land we're playing
    for card_name in hand:
        if card_name not in played and card_name != land_played:
            return card_name
    return None


def _find_fetch_target(library: List[str], fetch_targets: Tuple[str, ...], registry: CardRegistry) -> Optional[str]:
    """Find first card in library matching any fetch target prefix."""
    for card_name in library:
        card_obj = registry.get(card_name)
        if card_obj is None or card_obj.type != "land":
            continue
        for prefix in fetch_targets:
            if card_name.startswith(prefix):
                return card_name
    return None


def _find_saga_artifact(library: List[str], allowed_cmcs: Tuple[int, ...], registry: CardRegistry) -> Optional[str]:
    """Find first artifact in library with CMC in allowed_cmcs."""
    for card_name in library:
        card_obj = registry.get(card_name)
        if card_obj is None:
            continue
        if "artifact" not in card_obj.type:
            continue
        card_cmc = card_obj.cost.total if card_obj.cost else 0
        if card_cmc in allowed_cmcs:
            return card_name
    return None


def _can_play_card(
    card_name: str,
    card_obj: Card,
    pool: Mana,
    hand: List[str],
    played: Set[str],
    registry: CardRegistry,
    _land_played: bool,
    exiled: Set[str],
) -> Optional[Tuple[Mana, Optional[str]]]:
    """Try to play a card. Returns (new_pool, exiled_card) if successful, None otherwise."""
    ctype = card_obj.type
    cost = card_obj.cost or _ZERO

    if ctype == "land":
        return None  # Lands are played separately

    if "artifact" in ctype:
        requires = card_obj.requires
        if requires == "spare_land":
            if _count_lands(hand, registry) < 2:
                return None
        elif requires == "pitchable_card":
            if not _has_pitchable_card(hand, played, registry):
                return None
            # Find and track the exiled card
            pitch_card = _find_pitchable_card(hand, played | exiled, registry)
            if pitch_card is None:
                return None
            after_cost = _sub_mana(pool, cost)
            if after_cost is None:
                return None
            produces = card_obj.produces or _ZERO
            return _add_mana(after_cost, produces), pitch_card
        elif requires == "metalcraft":
            if _count_artifacts_on_board(played, registry, exclude=card_name) < 2:
                return None
        elif requires == "led_condition":
            return None  # LED is too conditional for T1, handled separately

        after_cost = _sub_mana(pool, cost)
        if after_cost is None:
            return None

        # Sacrifice artifacts don't produce mana when played - they stay on board
        if card_obj.sacrifices_on_use:
            return after_cost, None  # No mana gain, just play the artifact

        produces = card_obj.produces or _ZERO
        return _add_mana(after_cost, produces), None

    if ctype == "ritual":
        after_cost = _sub_mana(pool, cost)
        if after_cost is None:
            return None
        produces = card_obj.produces or _ZERO
        return _add_mana(after_cost, produces), None

    if ctype == "creature":
        creature_cost = card_obj.cost or _UNPLAYABLE
        if card_obj.evoke_cost is not None:
            creature_cost = card_obj.evoke_cost
        free_cond = card_obj.free_condition
        if free_cond == "swamp" and _hand_has_swamp(hand, registry):
            creature_cost = _ZERO
        if creature_cost == _ZERO:
            return pool, None  # Free creature, no mana change
        after_cost = _sub_mana(pool, creature_cost)
        if after_cost is None:
            return None
        return after_cost, None  # Creatures don't produce mana directly

    if ctype == "sacrifice_outlet":
        sac_cost = card_obj.cost or Mana(1, black=1)
        after_cost = _sub_mana(pool, sac_cost)
        if after_cost is None:
            return None
        if not _has_creature_on_board(played, registry):
            return None
        # Find best creature to sacrifice
        if card_obj.produces_base is not None:
            return _add_mana(after_cost, card_obj.produces_base), None
        if card_obj.produces_cmc:
            best_cmc = 0
            for p in played:
                p_obj = registry.get(p)
                if p_obj is not None and p_obj.type == "creature":
                    best_cmc = max(best_cmc, p_obj.cmc or 0)
            return _add_mana(after_cost, Mana(best_cmc, black=best_cmc)), None
        return after_cost, None

    if ctype == "equipment":
        equip_cost = card_obj.cost or Mana(1)
        after_cost = _sub_mana(pool, equip_cost)
        if after_cost is None:
            return None
        if not _has_creature_on_board(played, registry):
            return None
        tap_produces = card_obj.tap_produces or Mana(1)
        return _add_mana(after_cost, tap_produces), None

    if ctype == "tutor":
        return None  # Tutors don't produce mana on T1

    return None


def _enumerate_mana_lines(
    hand: List[str],
    registry: CardRegistry,
    pool: Mana,
    played: Set[str],
    remaining: List[str],
    target: ManaRequirement,
    land_played: bool,
    exiled: Set[str],
    sacrificed: Optional[Set[str]] = None,
) -> Tuple[bool, Mana, Set[str], Set[str], Set[str]]:
    """Recursive mana line enumeration. Returns (success, best_pool, best_played, exiled, sacrificed)."""
    if sacrificed is None:
        sacrificed = set()

    if _meets_requirement(pool, target):
        return True, pool, played.copy(), exiled.copy(), sacrificed.copy()

    best_pool = pool
    best_played = played.copy()
    best_exiled = exiled.copy()
    best_sacrificed = sacrificed.copy()
    found = False

    # Try playing cards from remaining
    for i, card_name in enumerate(remaining):
        card_obj = registry.get(card_name)
        if card_obj is None:
            continue

        new_remaining = remaining[:i] + remaining[i + 1 :]
        new_played = played | {card_name}

        result = _can_play_card(card_name, card_obj, pool, hand, new_played, registry, land_played, exiled)
        if result is None:
            continue

        new_pool, exiled_card = result
        new_exiled = exiled.copy()
        if exiled_card is not None:
            new_exiled.add(exiled_card)

        success, result_pool, result_played, result_exiled, result_sacrificed = _enumerate_mana_lines(
            hand, registry, new_pool, new_played, new_remaining, target, land_played, new_exiled, sacrificed
        )
        if success:
            return True, result_pool, result_played, result_exiled, result_sacrificed
        if _is_better_pool(result_pool, best_pool):
            best_pool = result_pool
            best_played = result_played
            best_exiled = result_exiled
            best_sacrificed = result_sacrificed
            found = True

    # Try sacrificing artifacts already on board for mana
    # Only sacrifice if it leads to a successful cast (don't sacrifice for best-effort)
    for card_name in played:
        if card_name in sacrificed:
            continue
        card_obj = registry.get(card_name)
        if card_obj is None or not card_obj.sacrifices_on_use:
            continue

        # Sacrifice the artifact for its mana
        produces = card_obj.produces or _ZERO
        new_pool = _add_mana(pool, produces)
        new_sacrificed = sacrificed | {card_name}

        success, result_pool, result_played, result_exiled, result_sacrificed = _enumerate_mana_lines(
            hand, registry, new_pool, played, remaining, target, land_played, exiled, new_sacrificed
        )
        if success:
            return True, result_pool, result_played, result_exiled, result_sacrificed
        # Don't update best-effort when sacrificing fails - keep the artifact for next turn

    # Try using land sac_mana abilities (e.g., Phyrexian Tower sacrificing a creature)
    # Only use if it leads to a successful cast
    for land_name in played:
        if land_name in sacrificed:
            continue
        land_obj = registry.get(land_name)
        if land_obj is None or land_obj.type != "land" or land_obj.sac_mana is None:
            continue

        # Find a creature to sacrifice
        for creature_name in played:
            if creature_name in sacrificed:
                continue
            creature_obj = registry.get(creature_name)
            if creature_obj is None or "creature" not in creature_obj.type:
                continue

            # Sacrifice the creature using the land's sac_mana ability
            # Replace base mana with sac_mana (mutually exclusive tap abilities)
            base_mana = land_obj.t1_mana or _ZERO
            pool_without_base = _sub_mana(pool, base_mana)
            if pool_without_base is None:
                # Base mana already spent, can't use sac_mana instead
                continue
            new_pool = _add_mana(pool_without_base, land_obj.sac_mana)
            new_sacrificed = sacrificed | {creature_name}

            success, result_pool, result_played, result_exiled, result_sacrificed = _enumerate_mana_lines(
                hand, registry, new_pool, played, remaining, target, land_played, exiled, new_sacrificed
            )
            if success:
                return True, result_pool, result_played, result_exiled, result_sacrificed
            # Don't update best-effort - keep the creature for next turn

    if not found:
        return False, pool, played.copy(), exiled.copy(), sacrificed.copy()
    return False, best_pool, best_played, best_exiled, best_sacrificed


def _compute_zones_after_play(
    land_played: Optional[str],
    cards_played: Set[str],
    exiled: Set[str],
    registry: CardRegistry,
    sacrificed: Optional[Set[str]] = None,
    fetched_land: Optional[str] = None,
) -> Tuple[Battlefield, List[str], List[str]]:
    """Categorize played cards into battlefield, graveyard, and exile zones."""
    if sacrificed is None:
        sacrificed = set()

    battlefield = Battlefield()
    graveyard: List[str] = []

    if land_played is not None:
        land_obj = registry.get(land_played)
        # Check if this is a fetchland
        if land_obj is not None and land_obj.fetch_targets is not None:
            # Fetchland goes to graveyard after being cracked
            graveyard.append(land_played)
            # Fetched land goes to battlefield
            if fetched_land is not None:
                battlefield.lands.append(fetched_land)
        else:
            battlefield.lands.append(land_played)

    for card_name in cards_played:
        if card_name == land_played:
            continue
        card_obj = registry.get(card_name)
        if card_obj is None:
            continue
        ctype = card_obj.type

        if ctype == "artifact":
            # Sacrifice artifacts go to graveyard only if actually sacrificed
            if card_name in sacrificed:
                graveyard.append(card_name)
            else:
                battlefield.artifacts.append(card_name)
        elif ctype == "creature":
            if card_obj.evoke_cost is not None:
                graveyard.append(card_name)
            else:
                battlefield.creatures.append(card_name)
        elif ctype == "equipment":
            if _has_creature_on_board(cards_played, registry):
                battlefield.equipment_attached.append(card_name)
            else:
                battlefield.artifacts.append(card_name)
        elif ctype in ("ritual", "sacrifice_outlet"):
            graveyard.append(card_name)
        else:
            graveyard.append(card_name)

    return battlefield, graveyard, list(exiled)


def _compute_battlefield_mana(battlefield: Battlefield, registry: CardRegistry) -> Mana:
    """Compute mana available from persistent battlefield permanents on T2+."""
    pool = _ZERO

    for land_name in battlefield.lands:
        card_obj = registry.get(land_name)
        if card_obj is None:
            continue
        t2_mana = card_obj.t2_mana or _ZERO
        pool = _add_mana(pool, t2_mana)

    for art_name in battlefield.artifacts:
        card_obj = registry.get(art_name)
        if card_obj is None:
            continue
        # Sacrifice artifacts don't produce mana passively - they need to be sacrificed
        if card_obj.sacrifices_on_use:
            continue
        produces = card_obj.produces or _ZERO
        pool = _add_mana(pool, produces)

    for equip_name in battlefield.equipment_attached:
        card_obj = registry.get(equip_name)
        if card_obj is None:
            continue
        tap_produces = card_obj.tap_produces or _ZERO
        pool = _add_mana(pool, tap_produces)

    return pool


def reconstruct_state(
    permanents: List[str],
    hand: List[str],
    graveyard: List[str],
    exile: List[str],
    library: List[str],
    registry: CardRegistry,
) -> GameState:
    """Reconstruct a GameState from zone contents."""
    battlefield = Battlefield()
    for card_name in permanents:
        card_obj = registry.get(card_name)
        if card_obj is None:
            continue
        ctype = card_obj.type
        if ctype == "land":
            battlefield.lands.append(card_name)
        elif ctype == "artifact":
            battlefield.artifacts.append(card_name)
        elif ctype == "creature":
            battlefield.creatures.append(card_name)
        elif ctype == "equipment":
            battlefield.equipment_attached.append(card_name)
    return GameState(
        hand=list(hand),
        library=list(library),
        battlefield=battlefield,
        graveyard=list(graveyard),
        exile=list(exile),
    )


class LandSelectionResult:
    """Result of land selection including fetchland mechanics."""

    __slots__ = (
        "success",
        "land_played",
        "best_pool",
        "best_played",
        "exiled",
        "sacrificed",
        "gemstone_exile",
        "fetched_land",
        "updated_library",
    )

    def __init__(
        self,
        success: bool,
        land_played: Optional[str],
        best_pool: Mana,
        best_played: Set[str],
        exiled: Set[str],
        sacrificed: Set[str],
        gemstone_exile: Optional[str],
        fetched_land: Optional[str] = None,
        updated_library: Optional[List[str]] = None,
    ) -> None:
        self.success = success
        self.land_played = land_played
        self.best_pool = best_pool
        self.best_played = best_played
        self.exiled = exiled
        self.sacrificed = sacrificed
        self.gemstone_exile = gemstone_exile
        self.fetched_land = fetched_land
        self.updated_library = updated_library if updated_library is not None else []


def _run_land_selection(
    hand: List[str],
    lands_in_hand: List[str],
    non_lands: List[str],
    initial_pool: Mana,
    existing_board: Set[str],
    registry: CardRegistry,
    target: ManaRequirement,
    compute_land_pool: Callable[[str, Mana], Optional[Mana]],
    land_was_played: bool,
    library: Optional[List[str]] = None,
) -> LandSelectionResult:
    """
    Template function for land selection and mana line enumeration.

    Returns LandSelectionResult with all zone tracking info.
    """
    if library is None:
        library = []

    best_overall_pool = initial_pool
    best_overall_played: Set[str] = set()
    best_overall_land: Optional[str] = None
    best_overall_exiled: Set[str] = set()
    best_overall_sacrificed: Set[str] = set()
    best_gemstone_exile: Optional[str] = None
    best_fetched_land: Optional[str] = None
    best_updated_library: List[str] = list(library)
    found_any = False

    # Try each land in hand as the land drop
    # Sort by t2_mana descending so "enters tapped" lands are played first
    # (they need a turn to untap, so playing them early maximizes future mana)
    def land_t2_mana(land_name: str) -> int:
        card = registry.get(land_name)
        return card.t2_mana.total if card and card.t2_mana else 0

    sorted_lands = sorted(lands_in_hand, key=land_t2_mana, reverse=True)
    for land in sorted_lands:
        card_obj = registry[land]

        # Check land-specific requirements
        gemstone_exile: Optional[str] = None
        if card_obj.requires == "exile_from_hand":
            if len(hand) < 2:
                continue
            # Find a card to exile for Gemstone Caverns
            gemstone_exile = _find_exile_card(hand, set(), land)
            if gemstone_exile is None:
                continue

        # Handle fetchlands
        fetched_land: Optional[str] = None
        current_library = list(library)
        if card_obj.fetch_targets is not None:
            # Search library for a valid target
            fetched_land = _find_fetch_target(current_library, card_obj.fetch_targets, registry)
            if fetched_land is None:
                # No valid target in library - skip this fetchland
                continue
            # Remove fetched land from library
            current_library.remove(fetched_land)
            # Use the fetched land's mana, not the fetchland's
            fetched_card = registry[fetched_land]
            land_pool = fetched_card.t1_mana or _ZERO
        else:
            # Compute the mana pool with this land
            maybe_pool = compute_land_pool(land, initial_pool)
            if maybe_pool is None:
                continue
            land_pool = maybe_pool

        played: Set[str] = existing_board | {land}
        # For fetchlands, also add the fetched land to played set
        if fetched_land is not None:
            played = played | {fetched_land}
        initial_exiled: Set[str] = {gemstone_exile} if gemstone_exile else set()
        success, result_pool, result_played, result_exiled, result_sacrificed = _enumerate_mana_lines(
            hand, registry, land_pool, played, non_lands, target, land_was_played, initial_exiled
        )
        if success:
            return LandSelectionResult(
                success=True,
                land_played=land,
                best_pool=result_pool,
                best_played=result_played,
                exiled=result_exiled,
                sacrificed=result_sacrificed,
                gemstone_exile=gemstone_exile,
                fetched_land=fetched_land,
                updated_library=current_library,
            )

        # Track best effort - prefer the first land tried (sorted by t2_mana descending)
        # Only update if pool is strictly better OR if we haven't tracked any land yet
        # Once a high-t2_mana land is tracked, don't replace it with a lower-t2_mana land
        # just because it has better T1 pool (we want the best T2 outcome for best-effort)
        current_t2 = land_t2_mana(land)
        best_t2 = land_t2_mana(best_overall_land) if best_overall_land else -1
        should_update = best_overall_land is None or (
            _is_better_pool(result_pool, best_overall_pool) and current_t2 >= best_t2
        )
        if should_update:
            best_overall_pool = result_pool
            best_overall_played = result_played
            best_overall_land = land
            best_overall_exiled = result_exiled
            best_overall_sacrificed = result_sacrificed
            best_gemstone_exile = gemstone_exile
            best_fetched_land = fetched_land
            best_updated_library = current_library
            found_any = True

    # Try without playing a land
    played_no_land: Set[str] = set(existing_board)
    success, result_pool, result_played, result_exiled, result_sacrificed = _enumerate_mana_lines(
        hand, registry, initial_pool, played_no_land, non_lands, target, False, set()
    )
    if success:
        return LandSelectionResult(
            success=True,
            land_played=None,
            best_pool=result_pool,
            best_played=result_played,
            exiled=result_exiled,
            sacrificed=result_sacrificed,
            gemstone_exile=None,
            fetched_land=None,
            updated_library=list(library),
        )

    if _is_better_pool(result_pool, best_overall_pool):
        best_overall_pool = result_pool
        best_overall_played = result_played
        best_overall_land = None
        best_overall_exiled = result_exiled
        best_overall_sacrificed = result_sacrificed
        best_gemstone_exile = None
        best_fetched_land = None
        best_updated_library = list(library)
        found_any = True

    if found_any:
        return LandSelectionResult(
            success=False,
            land_played=best_overall_land,
            best_pool=best_overall_pool,
            best_played=best_overall_played,
            exiled=best_overall_exiled,
            sacrificed=best_overall_sacrificed,
            gemstone_exile=best_gemstone_exile,
            fetched_land=best_fetched_land,
            updated_library=best_updated_library,
        )
    return LandSelectionResult(
        success=False,
        land_played=None,
        best_pool=initial_pool,
        best_played=set(),
        exiled=set(),
        sacrificed=set(),
        gemstone_exile=None,
        fetched_land=None,
        updated_library=list(library),
    )


def _compute_t1_land_pool(land: str, _initial_pool: Mana, registry: CardRegistry) -> Mana:
    """Compute mana pool for a fresh land on T1.

    Even if the land produces 0 mana on T1 (e.g., enters tapped), we still
    return _ZERO so it gets played - it may produce mana on T2.
    """
    card_obj = registry[land]
    t1_mana = card_obj.t1_mana or _ZERO
    return t1_mana


def _compute_t2_land_pool(land: str, initial_pool: Mana, registry: CardRegistry) -> Mana:
    """Compute mana pool for a new land on T2 (board mana + land's t1_mana)."""
    card_obj = registry[land]
    t1_mana = card_obj.t1_mana or _ZERO  # New land taps for t1_mana on its first turn

    return _add_mana(initial_pool, t1_mana)


def simulate_turn(
    turn_number: int,
    hand: List[str],
    drawn_card: Optional[str],
    prev_state: Optional[GameState],
    registry: CardRegistry,
    target: ManaRequirement,
    library: Optional[List[str]] = None,
) -> TurnResult:
    """Simulate a single turn with full zone tracking.

    Args:
        turn_number: Which turn to simulate (1, 2, 3, ...).
        hand: Cards in hand at start of turn (for T1, this is the opening hand;
              for T2+, this is prev state_after.hand — drawn_card will be appended).
        drawn_card: Card drawn this turn (None for T1, required for T2+).
        prev_state: Game state from previous turn (None for T1).
        registry: Card registry for lookups.
        target: Mana requirement to cast commander.
        library: Current library (for T1, optional for fetchlands; for T2+ taken from prev_state).

    Returns:
        TurnResult with before/after game states, mana pool, and cast success.
    """
    if turn_number == 1:
        # T1: no draw, empty board
        full_hand = list(hand)
        initial_pool = _ZERO
        board_card_names: Set[str] = set()
        prev_graveyard: List[str] = []
        prev_exile: List[str] = []
        prev_library: List[str] = list(library) if library is not None else []
        prev_battlefield = Battlefield()

        def compute_land_pool_t1(land: str, initial_pool: Mana) -> Optional[Mana]:
            return _compute_t1_land_pool(land, initial_pool, registry)

        compute_land_pool: Callable[[str, Mana], Optional[Mana]] = compute_land_pool_t1
    else:
        # T2+: optionally draw a card, use previous board mana
        if prev_state is None:
            raise ValueError(f"Turn {turn_number} requires prev_state")

        full_hand = list(hand) + ([drawn_card] if drawn_card is not None else [])
        prev_battlefield = prev_state.battlefield
        prev_graveyard = list(prev_state.graveyard)
        prev_exile = list(prev_state.exile)
        prev_library = list(prev_state.library)

        # Handle saga lands that sacrifice on this turn to search for artifacts
        # The saga can be tapped for its mana BEFORE sacrificing, so we include its mana
        saga_sacrificed_lands: List[str] = []
        saga_fetched_artifacts: List[str] = []
        saga_tap_mana = _ZERO
        for land_name in prev_battlefield.lands:
            land_obj = registry.get(land_name)
            if land_obj is None:
                continue
            if land_obj.sac_search_turn == turn_number and land_obj.sac_search_artifact_cmcs is not None:
                # This saga sacrifices on this turn
                artifact = _find_saga_artifact(prev_library, land_obj.sac_search_artifact_cmcs, registry)
                if artifact is not None:
                    saga_sacrificed_lands.append(land_name)
                    saga_fetched_artifacts.append(artifact)
                    prev_library = [c for c in prev_library if c != artifact]
                    # Saga can be tapped for its t2_mana before sacrificing
                    if land_obj.t2_mana is not None:
                        saga_tap_mana = _add_mana(saga_tap_mana, land_obj.t2_mana)

        # Update battlefield: remove sacrificed sagas
        # Fetched artifacts go to hand (as free plays) so they can be played during enumeration
        # This allows artifacts with requirements (like Chrome Mox) to check their conditions
        if saga_sacrificed_lands:
            prev_battlefield = Battlefield(
                lands=[land for land in prev_battlefield.lands if land not in saga_sacrificed_lands],
                artifacts=prev_battlefield.artifacts,
                creatures=prev_battlefield.creatures,
                equipment_attached=prev_battlefield.equipment_attached,
            )
            prev_graveyard = prev_graveyard + saga_sacrificed_lands
            # Add fetched artifacts to hand so they can be "played" during mana enumeration
            full_hand = full_hand + saga_fetched_artifacts

        # Compute mana from remaining permanents plus saga tap mana
        initial_pool = _add_mana(_compute_battlefield_mana(prev_battlefield, registry), saga_tap_mana)
        board_card_names = set(prev_battlefield.all_permanents)

        def compute_land_pool_t2plus(land: str, initial_pool: Mana) -> Mana:
            return _compute_t2_land_pool(land, initial_pool, registry)

        compute_land_pool = compute_land_pool_t2plus

    lands_in_hand = [c for c in full_hand if (obj := registry.get(c)) is not None and obj.type == "land"]
    non_lands = [c for c in full_hand if (obj := registry.get(c)) is None or obj.type != "land" and c != "filler"]

    result = _run_land_selection(
        full_hand,
        lands_in_hand,
        non_lands,
        initial_pool,
        board_card_names,
        registry,
        target,
        compute_land_pool,
        True,
        library=prev_library,
    )

    # Extract values from result
    success = result.success
    land_played = result.land_played
    mana_pool = result.best_pool
    played_set = result.best_played
    exiled_set = result.exiled
    sacrificed_set = result.sacrificed
    gemstone_exile = result.gemstone_exile
    fetched_land = result.fetched_land
    updated_library = result.updated_library

    # Separate newly-played cards from previous board cards
    new_played = played_set - board_card_names
    # Don't include land_played or fetched_land in cards_played list (they're handled separately)
    cards_played_list = [c for c in new_played if c != land_played and c != fetched_land]

    # Build zones from new plays
    new_battlefield, new_graveyard, new_exile = _compute_zones_after_play(
        land_played, new_played, exiled_set, registry, sacrificed_set, fetched_land
    )

    # Add gemstone caverns exile if applicable
    if gemstone_exile is not None and gemstone_exile not in new_exile:
        new_exile.append(gemstone_exile)

    # Merge with previous state if applicable
    # Remove sacrificed artifacts from previous battlefield
    prev_artifacts_remaining = [a for a in prev_battlefield.artifacts if a not in sacrificed_set]
    # Add sacrificed artifacts (from previous board) to graveyard
    sacrificed_from_prev = [a for a in prev_battlefield.artifacts if a in sacrificed_set]

    merged_battlefield = Battlefield(
        lands=prev_battlefield.lands + new_battlefield.lands,
        artifacts=prev_artifacts_remaining + new_battlefield.artifacts,
        creatures=prev_battlefield.creatures + new_battlefield.creatures,
        equipment_attached=prev_battlefield.equipment_attached + new_battlefield.equipment_attached,
    )
    merged_graveyard = prev_graveyard + new_graveyard + sacrificed_from_prev
    merged_exile = prev_exile + new_exile

    # Hand after playing cards (remove played and exiled cards, respecting duplicates)
    hand_end = list(full_hand)
    for card in new_played:
        if card in hand_end:
            hand_end.remove(card)
    for card in exiled_set:
        if card in hand_end:
            hand_end.remove(card)
    if gemstone_exile is not None and gemstone_exile in hand_end:
        hand_end.remove(gemstone_exile)

    # Build state_before
    state_before = GameState(
        hand=full_hand,
        library=prev_library,
        battlefield=prev_battlefield,
        graveyard=prev_graveyard,
        exile=prev_exile,
    )

    # Build state_after - use updated library if fetchland was used
    state_after = GameState(
        hand=hand_end,
        library=updated_library,
        battlefield=merged_battlefield,
        graveyard=merged_graveyard,
        exile=merged_exile,
    )

    return TurnResult(
        turn_number=turn_number,
        state_before=state_before,
        state_after=state_after,
        drawn_card=drawn_card,
        land_played=land_played,
        cards_played=cards_played_list,
        mana_remaining=mana_pool,
        can_cast_commander=success,
    )
