"""Integration tests for specific hands using mloda.run_all()."""

from typing import Any, List, Optional

import polars as pl

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from card_database.colorless import CHROME_MOX, JEWELED_LOTUS, LOTUS_PETAL
from card_database.lands import (
    ANCIENT_TOMB,
    BLOODSTAINED_MIRE,
    BOGGART_TRAWLER,
    BOSEIJU_WHO_SHELTERS_ALL,
    CABAL_PIT,
    CITY_OF_TRAITORS,
    EUMIDIAN_HATCHERY,
    FELL_THE_PROFANE,
    PEAT_BOG,
    PHYREXIAN_TOWER,
    SWAMP,
    TALON_GATES_OF_MADARA,
    URBORG_TOMB_OF_YAWGMOTH,
    URZAS_CAVE,
    URZAS_SAGA,
)
from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.card_registry import CardRegistry, ManaRequirement, build_registry
from cedh_mulligan_simulator.feature_groups.mulligan import HandGenerator, MulliganResult
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider, SpecificHandProvider


def _concat_results(results: list[Any]) -> pl.DataFrame:
    """Horizontal concat of Polars DataFrames, deduplicating columns."""
    seen_cols: set[str] = set()
    parts: list[pl.DataFrame] = []
    for r in results:
        if isinstance(r, pl.DataFrame):
            new_cols = [c for c in r.columns if c not in seen_cols]
            if new_cols:
                parts.append(r.select(new_cols))
                seen_cols.update(new_cols)
    return pl.concat(parts, how="horizontal")


def run_hand_simulation(
    hand: List[str],
    remaining_library: Optional[List[str]] = None,
    registry: Optional[CardRegistry] = None,
    commander_cost: Optional[ManaRequirement] = None,
) -> pl.DataFrame:
    """Run mloda simulation for a specific hand.

    Args:
        hand: List of 7 card names
        remaining_library: Cards remaining in library (defaults to all registry cards not in hand)
        registry: Card registry to use (defaults to BRAIDS_REGISTRY)
        commander_cost: Commander cost to use (defaults to BRAIDS_COST)

    Returns:
        DataFrame with all turn simulation columns
    """
    if registry is None:
        registry = BRAIDS_REGISTRY
    if commander_cost is None:
        commander_cost = BRAIDS_COST

    # Compute remaining library if not specified
    if remaining_library is None:
        all_cards = list(registry.keys())
        hand_set = set(hand)
        remaining_library = [c for c in all_cards if c not in hand_set]

    # Set up test data
    SpecificHandProvider._test_data = pl.DataFrame(
        {
            "hand": [hand],
            "simulation_id": [0],
            "mulligan_count": [0],
            "scenario_id": ["test"],
            "remaining_library": [remaining_library],
        }
    )

    opts = Options(
        group={"scenario_id": "test"},
        context={"card_registry": registry, "commander_cost": commander_cost, "draw_per_turn": False},
    )

    # Run with SpecificHandProvider, disable conflicting providers
    results = mlodaAPI.run_all(
        features=[
            Feature("hand", options=opts),
            Feature("hand__t1", options=opts),
            Feature("hand__t1__t2", options=opts),
            Feature("hand__t1__t2__t3", options=opts),
        ],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups(
            {
                HandGenerator,
                MulliganResult,
                HandMulliganTestDataProvider,
            }
        ),
    )

    return _concat_results(results)


base_hand = [
    "bitter_triumph",
    "cut_down",
    "deadly_rollick",
    "inquisition_of_kozilek",
    "dismember",
    "saw_in_half",
    "shallow_grave",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBraidsSpecificHands:
    def test_swamp_dark_ritual_t1_castable(self) -> None:
        """Swamp + Dark Ritual should cast Braids T1."""
        hand = base_hand[:5] + ["swamp", "dark_ritual"]  # Add swamp and ritual to base hand
        # Define a specific library so we know what gets drawn
        remaining_library = ["swamp", "swamp"]  # Simple library for test
        df = run_hand_simulation(hand, remaining_library)
        df_dict = df.to_dict(as_series=False)

        # Check result
        assert df_dict["hand__t1"] == [True]
        assert df_dict["hand__t1__t2"] == [False]

        # Check drawn cards columns exist and are None (no draw)
        assert df_dict["hand__t1~drawn"] == df_dict["hand__t1__t2~drawn"] == [None]

        # Check hand
        assert df_dict["hand"] == [hand]
        # hand__t1__t2~hand hand__t1~hand
        assert df_dict["hand__t1~hand"][0] == df_dict["hand__t1__t2~hand"][0] == base_hand[:5]

        assert "swamp" in df_dict["hand__t1~battlefield"][0] and "swamp" in df_dict["hand__t1__t2~battlefield"][0]
        assert (
            "dark_ritual" in df_dict["hand__t1~graveyard"][0] and "dark_ritual" in df_dict["hand__t1__t2~graveyard"][0]
        )
        assert (
            df_dict["hand__t1~remaining_library"][0]
            == ["swamp", "swamp"]
            == df_dict["hand__t1__t2~remaining_library"][0]
        )
        assert df_dict["hand__t1~land_played"][0] == "swamp"
        assert df_dict["hand__t1__t2~land_played"][0] is None
        assert df_dict["hand__t1~cards_played"][0] == ["dark_ritual"]
        assert df_dict["hand__t1__t2~cards_played"][0] == []
        assert df_dict["hand__t1__t2~exile"][0] == df_dict["hand__t1~exile"][0] == []

    def test_swamp_not_castable(self) -> None:
        """Swamp alone should not cast Braids T1."""
        hand = base_hand[:4] + ["swamp", "swamp", "swamp"]  # Add 3 swamps to base hand
        df = run_hand_simulation(hand)
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [False]
        assert df_dict["hand__t1__t2"] == [False]
        assert df_dict["hand__t1__t2__t3"] == [True]

        df = run_hand_simulation(base_hand[:4] + ["swamp", "talon_gates_of_madara", "emergence_zone"])
        df_dict = df.to_dict(as_series=False)
        assert df_dict["hand__t1__t2__t3"] == [False]

    def test_swamp__chrome_mox_t1_castable(self) -> None:
        """Swamp + Lotus Petal + Chrome Mox should cast Braids T1."""
        df = run_hand_simulation(base_hand[:4] + ["swamp", "lotus_petal", "chrome_mox"])
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [True]
        assert df_dict["hand__t1__t2"] == [False]

        df = run_hand_simulation(
            ["ornithopter", "ornithopter", "ornithopter", "ornithopter"] + ["swamp", "lotus_petal", "chrome_mox"]
        )
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [False]
        assert df_dict["hand__t1__t2"] == [False]

    def test_swamp__mox_opal(self) -> None:
        """Swamp + Lotus Petal + Mox Opal should NOT cast Braids T1 (needs metalcraft)."""
        df = run_hand_simulation(base_hand[:4] + ["swamp", "lotus_petal", "mox_opal"])
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [False]
        assert df_dict["hand__t1__t2"] == [False]

        df = run_hand_simulation(base_hand[:3] + ["ornithopter", "swamp", "lotus_petal", "mox_opal"])
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [True]
        assert df_dict["hand__t1__t2"] == [False]

    def test_swamp__sol_ring(self) -> None:
        """Swamp + Lotus Petal + Sol Ring should cast Braids T1."""

        for i in ["sol_ring", "mana_vault"]:
            df = run_hand_simulation(base_hand[:4] + ["swamp", "lotus_petal", i])
            df_dict = df.to_dict(as_series=False)

            assert df_dict["hand__t1"] == [False]
            assert df_dict["hand__t1__t2"] == [True]
            assert df_dict["hand__t1__t2__t3"] == [False]

    def test_swamp__mox_amber(self) -> None:
        df = run_hand_simulation(base_hand[:4] + ["swamp", "lotus_petal", "mox_amber"])
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [False]
        assert df_dict["hand__t1__t2"] == [False]
        assert df_dict["hand__t1__t2__t3"] == [False]

    def test_swamp__mox_diamond(self) -> None:
        df = run_hand_simulation(base_hand[:4] + ["swamp", "lotus_petal", "mox_diamond"])
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [False]
        assert df_dict["hand__t1__t2"] == [False]

        df = run_hand_simulation(base_hand[:3] + ["swamp", "swamp", "lotus_petal", "mox_diamond"])
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [True]
        assert df_dict["hand__t1__t2"] == [True]

    def test_swamp__jlo(self) -> None:
        # Create registry with Jeweled Lotus added
        jlo_registry = build_registry(*BRAIDS_REGISTRY.values(), JEWELED_LOTUS)

        df = run_hand_simulation(base_hand[:6] + ["jeweled_lotus"], registry=jlo_registry)
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [True]
        assert df_dict["hand__t1__t2"] == [False]

    def test_land_check(self) -> None:
        # Create registry with Jeweled Lotus added
        lands = [
            "swamp",
            "swamp_2",
            "snow_covered_swamp",
            "snow_covered_swamp_2",
            "marsh_flats",
            "polluted_delta",
            "prismatic_vista",
            "verdant_catacombs",
            "bloodstained_mire",
            "multiversal_passage",
            URBORG_TOMB_OF_YAWGMOTH.name,
            BOGGART_TRAWLER.name,
            FELL_THE_PROFANE.name,
            EUMIDIAN_HATCHERY.name,
            CABAL_PIT.name,
        ]
        for i in lands:
            df = run_hand_simulation(base_hand[:6] + [i] + ["lotus_petal", "chrome_mox"])
            df_dict = df.to_dict(as_series=False)

            assert df_dict["hand__t1"] == [True], f"{i} should allow T1 cast"
            assert df_dict["hand__t1__t2"] == [False]

            df = run_hand_simulation(base_hand[:6] + [i] + ["lotus_petal", TALON_GATES_OF_MADARA.name])
            df_dict = df.to_dict(as_series=False)

            assert df_dict["hand__t1"] == [False], f"{i} should not allow T1 cast without mana sources"
            assert df_dict["hand__t1__t2"] == [True], f"{i} should allow T2 cast"

        # Create registry with Jeweled Lotus added
        lands = [
            TALON_GATES_OF_MADARA.name,
            ANCIENT_TOMB.name,
            CITY_OF_TRAITORS.name,
            PHYREXIAN_TOWER.name,
            TALON_GATES_OF_MADARA.name,
            URZAS_SAGA.name,
        ]
        for i in lands:
            df = run_hand_simulation(base_hand[:6] + [i] + ["lotus_petal", "swamp"])
            df_dict = df.to_dict(as_series=False)

            assert df_dict["hand__t1"] == [False], f"{i} should not allow T1 cast without mana sources"
            assert df_dict["hand__t1__t2"] == [True], f"{i} should allow T2 cast"

    def test_land_check_2(self) -> None:
        lands = [ANCIENT_TOMB.name, CITY_OF_TRAITORS.name]
        for i in lands:
            df = run_hand_simulation(base_hand[:6] + [i] + ["lotus_petal", "chrome_mox"])
            df_dict = df.to_dict(as_series=False)
            assert df_dict["hand__t1"] == [True], f"{i} should allow T1 cast without mana sources"
            assert df_dict["hand__t1__t2"] == [False], f"{i} should not allow T2 cast"

        df = run_hand_simulation(base_hand[:4] + [PHYREXIAN_TOWER.name, LOTUS_PETAL.name, "ornithopter"])
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [True]
        assert df_dict["hand__t1__t2"] == [False]

    def test_land_check_3(self) -> None:
        df = run_hand_simulation(base_hand[:4] + [BOSEIJU_WHO_SHELTERS_ALL.name, LOTUS_PETAL.name, CHROME_MOX.name])
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [False]
        assert df_dict["hand__t1__t2"] == [True]

        df = run_hand_simulation(base_hand[:4] + [LOTUS_PETAL.name, PEAT_BOG.name])
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [False]
        assert df_dict["hand__t1__t2"] == [True]

        df = run_hand_simulation(base_hand[:4] + [PEAT_BOG.name, SWAMP.name])
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [False]
        assert df_dict["hand__t1__t2"] == [True]

    def test_fetchland_add_swamp_or_snowcovered_to_board_if_still_in_deck_and_fetchland_into_grave(self) -> None:
        for fetchland in [
            "marsh_flats",
            "polluted_delta",
            "prismatic_vista",
            "verdant_catacombs",
            BLOODSTAINED_MIRE.name,
        ]:
            df = run_hand_simulation(base_hand[:4] + [fetchland, "lotus_petal", "chrome_mox"])
            df_dict = df.to_dict(as_series=False)

            assert df_dict["hand__t1"] == [True], f"Should cast Braids T1 with {fetchland}"
            assert df_dict["hand__t1__t2"] == [False]

            # Fetchland should be in graveyard after being cracked
            graveyard = df_dict["hand__t1~graveyard"][0]
            assert fetchland in graveyard, f"Fetchland {fetchland} should be in graveyard, got: {graveyard}"
            assert fetchland not in df_dict["hand__t1~battlefield"][0], (
                f"Fetchland {fetchland} should NOT be on battlefield"
            )

            # A swamp or snow-covered swamp should be on the battlefield
            battlefield = df_dict["hand__t1~battlefield"][0]
            has_swamp = any(land.startswith("swamp") or land.startswith("snow_covered_swamp") for land in battlefield)
            assert has_swamp, f"Swamp or snow-covered swamp should be on battlefield, got: {battlefield}"

            # Verify the full board state: fetched land + chrome_mox
            assert "chrome_mox" in battlefield, f"chrome_mox should be on battlefield, got: {battlefield}"

            # lotus_petal is sacrificed for mana and goes to graveyard
            assert "lotus_petal" in graveyard, f"lotus_petal should be in graveyard (sacrificed), got: {graveyard}"

    def test_land_check_4(self) -> None:
        # Need to use a registry with URZAS_CAVE since it's not in BRAIDS_REGISTRY
        urzas_cave_registry = build_registry(*BRAIDS_REGISTRY.values(), URZAS_CAVE)
        df = run_hand_simulation(base_hand[:4] + [URZAS_CAVE.name, LOTUS_PETAL.name], registry=urzas_cave_registry)
        df_dict = df.to_dict(as_series=False)

        assert df_dict["hand__t1"] == [False]
        assert df_dict["hand__t1__t2"] == [False]
        assert df_dict["hand__t1__t2__t3"] == [True]

    def test_land_gemstone(self) -> None:
        """Gemstone Caverns enables T1 casting via exile; Lake of the Dead sacs a swamp for 4B on T3."""
        from card_database.lands import GEMSTONE_CAVERNS, LAKE_OF_THE_DEAD

        # Gemstone Caverns: exile a filler card from hand to tap for any-color mana on T1.
        # With Gemstone (1any) + Chrome Mox (1any, pitch base_hand card) + Lotus Petal (1any) = 3 any
        # → satisfies 1BB (total=3, black=2) since any_color satisfies black requirements.
        df = run_hand_simulation(
            base_hand[:4] + [GEMSTONE_CAVERNS.name, "lotus_petal", "chrome_mox"],
        )
        df_dict = df.to_dict(as_series=False)
        assert df_dict["hand__t1"] == [True], "Gemstone Caverns should enable T1 cast via exile"
        assert df_dict["hand__t1__t2"] == [False]

        # Gemstone Caverns without enough mana sources: hand size 1 can't exile (needs ≥ 2 cards).
        # With only Gemstone + 1 other card, exile reduces hand to 0 usable cards.
        df = run_hand_simulation(
            base_hand[:4] + [GEMSTONE_CAVERNS.name, "lotus_petal", "chrome_mox"],
            remaining_library=["swamp"],
        )
        df_dict = df.to_dict(as_series=False)
        # Still T1 castable (7-card hand has enough cards to exile)
        assert df_dict["hand__t1"] == [True]

        # Lake of the Dead: enters tapped (0 T1 mana), T2 land drop produces 0 (t1_mana=Mana()),
        # but on T3 it can sacrifice a swamp already on battlefield for 4B.
        # Hand: Swamp + Lake of the Dead + 5 fillers.
        # T1: play Swamp (1B) — not enough for 1BB → T1 False
        # T2: play Lake as land drop (t1_mana=0), pool = 1B from Swamp on BF → T2 False
        # T3: BF has Swamp + Lake; Lake sacs Swamp → 4B ≥ 1BB → T3 True
        lake_hand = base_hand[:5] + [LAKE_OF_THE_DEAD.name, "swamp"]
        df = run_hand_simulation(lake_hand, remaining_library=["filler", "filler", "filler"])
        df_dict = df.to_dict(as_series=False)
        assert df_dict["hand__t1"] == [False], "Lake of the Dead should not allow T1 cast"
        assert df_dict["hand__t1__t2"] == [False], "Lake of the Dead should not allow T2 cast (only 1B on BF)"
        assert df_dict["hand__t1__t2__t3"] == [True], "Lake + Swamp should allow T3 cast (4B)"

        # Lake alone (no swamp): T3 still False since lake produces 0 without swamp sacrifice.
        # Use a different land (Boseiju — produces only colorless, not black) so Lake can't sacrifice.
        from card_database.lands import BOSEIJU_WHO_SHELTERS_ALL

        lake_no_swamp = base_hand[:5] + [LAKE_OF_THE_DEAD.name, BOSEIJU_WHO_SHELTERS_ALL.name]
        df = run_hand_simulation(lake_no_swamp, remaining_library=["filler", "filler", "filler"])
        df_dict = df.to_dict(as_series=False)
        assert df_dict["hand__t1__t2__t3"] == [False], "Lake without swamp should not allow T3 cast"

    def test_draw_per_turn_true(self) -> None:
        """draw_per_turn=True: drawing a key card on T2 should open new mana lines."""
        from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
        from mloda.user import Feature, Options, mlodaAPI
        from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
        from cedh_mulligan_simulator.feature_groups.mulligan import HandGenerator, MulliganResult
        from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider

        # Hand with only a Swamp (cannot cast T1 without drawing mana).
        # T1 draws library[0] = filler (tracking only, not added to T1 hand evaluation).
        # T2 draws library[1] = dark_ritual → Swamp BF (1B) + DR (1B→3B) = 1BB → castable.
        hand = base_hand[:6] + ["swamp"]
        remaining_library = ["filler", "dark_ritual", "filler"]

        SpecificHandProvider._test_data = pl.DataFrame(
            {
                "hand": [hand],
                "simulation_id": [0],
                "mulligan_count": [0],
                "scenario_id": ["test"],
                "remaining_library": [remaining_library],
            }
        )

        opts_draw = Options(
            group={"scenario_id": "test"},
            context={"card_registry": BRAIDS_REGISTRY, "commander_cost": BRAIDS_COST, "draw_per_turn": True},
        )

        results_draw = mlodaAPI.run_all(
            features=[
                Feature("hand__t1", options=opts_draw),
                Feature("hand__t1__t2", options=opts_draw),
            ],
            compute_frameworks={PolarsDataFrame},
            plugin_collector=PluginCollector.disabled_feature_groups(
                {HandGenerator, MulliganResult, HandMulliganTestDataProvider}
            ),
        )
        df_draw = _concat_results(results_draw)
        # T1: Swamp alone → not castable
        assert not bool(df_draw["hand__t1"][0])
        # T2 with draw: draws Dark Ritual → Swamp (1B) + Dark Ritual (1B→3B) = 3B → castable
        assert bool(df_draw["hand__t1__t2"][0]), "Should cast T2 after drawing Dark Ritual"
