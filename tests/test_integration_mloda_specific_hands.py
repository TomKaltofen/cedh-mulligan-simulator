"""Integration tests for specific hands using mloda.run_all()."""

from typing import Any, List, Optional

import pandas as pd

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.provider import DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from card_database.colorless import CHROME_MOX, JEWELED_LOTUS, LOTUS_PETAL, MEMNITE
from card_database.lands import (
    AGADEEMS_AWAKENING,
    ANCIENT_TOMB,
    BLOODSTAINED_MIRE,
    BOGGART_TRAWLER,
    BOSEIJU_WHO_SHELTERS_ALL,
    CABAL_PIT,
    CITY_OF_TRAITORS,
    EMERGENCE_ZONE,
    EUMIDIAN_HATCHERY,
    FELL_THE_PROFANE,
    PEAT_BOG,
    PHYREXIAN_TOWER,
    SWAMP,
    TAKENUMA_ABANDONED_MIRE,
    TALON_GATES_OF_MADARA,
    URBORG_TOMB_OF_YAWGMOTH,
    URZAS_CAVE,
    URZAS_SAGA,
    VAULT_OF_WHISPERS,
)
from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.card_registry import CardRegistry, ManaRequirement, build_registry
from cedh_mulligan_simulator.feature_groups.mulligan import HandGenerator, MulliganResult
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider


pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)


class SpecificHandProvider(FeatureGroup):
    """Provides specific hand data for integration testing.

    Set _test_data before calling mlodaAPI.run_all().
    Required columns: hand, simulation_id, mulligan_count, scenario_id, remaining_library
    """

    _test_data: Optional[pd.DataFrame] = None

    @classmethod
    def input_data(cls) -> Optional[Any]:
        return DataCreator({"hand", "simulation_id", "mulligan_count", "scenario_id", "remaining_library"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return cls._test_data


def run_hand_simulation(
    hand: List[str],
    remaining_library: Optional[List[str]] = None,
    registry: Optional[CardRegistry] = None,
    commander_cost: Optional[ManaRequirement] = None,
) -> pd.DataFrame:
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
    SpecificHandProvider._test_data = pd.DataFrame(
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
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups(
            {
                HandGenerator,
                MulliganResult,
                HandMulliganTestDataProvider,
            }
        ),
    )

    return pd.concat([r for r in results if isinstance(r, pd.DataFrame)], axis=1)


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
        df_dict = df.to_dict()

        # Check result
        assert df_dict["hand__t1"] == {0: True}
        assert df_dict["hand__t1__t2"] == {0: False}

        # Check drawn cards columns exist and are None (no draw)
        assert df_dict["hand__t1~drawn"] == df_dict["hand__t1__t2~drawn"] == {0: None}

        # Check hand
        assert df_dict["hand"] == {0: hand}
        # hand__t1__t2~hand hand__t1~hand
        assert df_dict["hand__t1~hand"][0] == df_dict["hand__t1__t2~hand"][0] == base_hand[:5]

        assert "swamp" in df["hand__t1~battlefield"].iloc[0] and "swamp" in df["hand__t1__t2~battlefield"].iloc[0]
        assert (
            "dark_ritual" in df["hand__t1~graveyard"].iloc[0] and "dark_ritual" in df["hand__t1__t2~graveyard"].iloc[0]
        )
        assert (
            df["hand__t1~remaining_library"].iloc[0]
            == ["swamp", "swamp"]
            == df["hand__t1__t2~remaining_library"].iloc[0]
            == ["swamp", "swamp"]
        )
        assert df["hand__t1~land_played"].iloc[0] == "swamp"
        assert df["hand__t1__t2~land_played"].iloc[0] is None
        assert df["hand__t1~cards_played"].iloc[0] == ["dark_ritual"]
        assert df["hand__t1__t2~cards_played"].iloc[0] == []
        assert df["hand__t1__t2~exile"].iloc[0] == df["hand__t1~exile"].iloc[0] == []

    def test_swamp_not_castable(self) -> None:
        """Swamp alone should not cast Braids T1."""
        hand = base_hand[:4] + ["swamp", "swamp", "swamp"]  # Add 3 swamps to base hand
        df = run_hand_simulation(hand)
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: False}
        assert df_dict["hand__t1__t2"] == {0: False}
        assert df_dict["hand__t1__t2__t3"] == {0: True}

        df = run_hand_simulation(base_hand[:4] + ["swamp", "talon_gates_of_madara", "emergence_zone"])
        df_dict = df.to_dict()
        assert df_dict["hand__t1__t2__t3"] == {0: False}

    def test_swamp__chrome_mox_t1_castable(self) -> None:
        """Swamp + Lotus Petal + Chrome Mox should cast Braids T1."""
        df = run_hand_simulation(base_hand[:4] + ["swamp", "lotus_petal", "chrome_mox"])
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: True}
        assert df_dict["hand__t1__t2"] == {0: False}

        df = run_hand_simulation(["memnite", "memnite", "memnite", "memnite"] + ["swamp", "lotus_petal", "chrome_mox"])
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: False}
        assert df_dict["hand__t1__t2"] == {0: False}

    def test_swamp__mox_opal(self) -> None:
        """Swamp + Lotus Petal + Mox Opal should NOT cast Braids T1 (needs metalcraft)."""
        df = run_hand_simulation(base_hand[:4] + ["swamp", "lotus_petal", "mox_opal"])
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: False}
        assert df_dict["hand__t1__t2"] == {0: False}

        df = run_hand_simulation(base_hand[:3] + ["memnite", "swamp", "lotus_petal", "mox_opal"])
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: True}
        assert df_dict["hand__t1__t2"] == {0: False}

    def test_swamp__sol_ring(self) -> None:
        """Swamp + Lotus Petal + Sol Ring should cast Braids T1."""

        for i in ["sol_ring", "mana_vault"]:
            df = run_hand_simulation(base_hand[:4] + ["swamp", "lotus_petal", i])
            df_dict = df.to_dict()

            assert df_dict["hand__t1"] == {0: False}
            assert df_dict["hand__t1__t2"] == {0: True}
            assert df_dict["hand__t1__t2__t3"] == {0: False}

    def test_swamp__mox_amber(self) -> None:
        df = run_hand_simulation(base_hand[:4] + ["swamp", "lotus_petal", "mox_amber"])
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: False}
        assert df_dict["hand__t1__t2"] == {0: False}
        assert df_dict["hand__t1__t2__t3"] == {0: False}

    def test_swamp__mox_diamond(self) -> None:
        df = run_hand_simulation(base_hand[:4] + ["swamp", "lotus_petal", "mox_diamond"])
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: False}
        assert df_dict["hand__t1__t2"] == {0: False}

        df = run_hand_simulation(base_hand[:3] + ["swamp", "swamp", "lotus_petal", "mox_diamond"])
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: True}
        assert df_dict["hand__t1__t2"] == {0: True}

    def test_swamp__jlo(self) -> None:
        # Create registry with Jeweled Lotus added
        jlo_registry = build_registry(*BRAIDS_REGISTRY.values(), JEWELED_LOTUS)

        df = run_hand_simulation(base_hand[:6] + ["jeweled_lotus"], registry=jlo_registry)
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: True}
        assert df_dict["hand__t1__t2"] == {0: False}

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
            AGADEEMS_AWAKENING.name,
            BOGGART_TRAWLER.name,
            FELL_THE_PROFANE.name,
            EUMIDIAN_HATCHERY.name,
            CABAL_PIT.name,
            VAULT_OF_WHISPERS.name,
            TAKENUMA_ABANDONED_MIRE.name,
        ]
        for i in lands:
            df = run_hand_simulation(base_hand[:6] + [i] + ["lotus_petal", "chrome_mox"])
            df_dict = df.to_dict()

            assert df_dict["hand__t1"] == {0: True}, f"{i} should allow T1 cast"
            assert df_dict["hand__t1__t2"] == {0: False}

            df = run_hand_simulation(base_hand[:6] + [i] + ["lotus_petal", EMERGENCE_ZONE.name])
            df_dict = df.to_dict()

            assert df_dict["hand__t1"] == {0: False}, f"{i} should not allow T1 cast without mana sources"
            assert df_dict["hand__t1__t2"] == {0: True}, f"{i} should allow T2 cast"

        # Create registry with Jeweled Lotus added
        lands = [
            EMERGENCE_ZONE.name,
            ANCIENT_TOMB.name,
            CITY_OF_TRAITORS.name,
            PHYREXIAN_TOWER.name,
            TALON_GATES_OF_MADARA.name,
            URZAS_SAGA.name,
        ]
        for i in lands:
            df = run_hand_simulation(base_hand[:6] + [i] + ["lotus_petal", "swamp"])
            df_dict = df.to_dict()

            assert df_dict["hand__t1"] == {0: False}, f"{i} should not allow T1 cast without mana sources"
            assert df_dict["hand__t1__t2"] == {0: True}, f"{i} should allow T2 cast"

    def test_land_check_2(self) -> None:
        lands = [ANCIENT_TOMB.name, CITY_OF_TRAITORS.name]
        for i in lands:
            df = run_hand_simulation(base_hand[:6] + [i] + ["lotus_petal", "chrome_mox"])
            df_dict = df.to_dict()
            assert df_dict["hand__t1"] == {0: True}, f"{i} should allow T1 cast without mana sources"
            assert df_dict["hand__t1__t2"] == {0: False}, f"{i} should not allow T2 cast"

        df = run_hand_simulation(base_hand[:4] + [PHYREXIAN_TOWER.name, LOTUS_PETAL.name, MEMNITE.name])
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: True}
        assert df_dict["hand__t1__t2"] == {0: False}

    def test_land_check_3(self) -> None:
        df = run_hand_simulation(base_hand[:4] + [BOSEIJU_WHO_SHELTERS_ALL.name, LOTUS_PETAL.name, CHROME_MOX.name])
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: False}
        assert df_dict["hand__t1__t2"] == {0: True}

        df = run_hand_simulation(base_hand[:4] + [LOTUS_PETAL.name, PEAT_BOG.name])
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: False}
        assert df_dict["hand__t1__t2"] == {0: True}

        df = run_hand_simulation(base_hand[:4] + [PEAT_BOG.name, SWAMP.name])
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: False}
        assert df_dict["hand__t1__t2"] == {0: True}

    def test_fetchland_add_swamp_or_snowcovered_to_board_if_still_in_deck_and_fetchland_into_grave(self) -> None:
        for fetchland in [
            "marsh_flats",
            "polluted_delta",
            "prismatic_vista",
            "verdant_catacombs",
            BLOODSTAINED_MIRE.name,
        ]:
            df = run_hand_simulation(base_hand[:4] + [fetchland, "lotus_petal", "chrome_mox"])
            df_dict = df.to_dict()

            assert df_dict["hand__t1"] == {0: True}, f"Should cast Braids T1 with {fetchland}"
            assert df_dict["hand__t1__t2"] == {0: False}

            # Fetchland should be in graveyard after being cracked
            graveyard = df_dict["hand__t1~graveyard"][0]
            assert fetchland in graveyard, f"Fetchland {fetchland} should be in graveyard, got: {graveyard}"
            assert (
                fetchland not in df_dict["hand__t1~battlefield"][0]
            ), f"Fetchland {fetchland} should NOT be on battlefield"

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
        df_dict = df.to_dict()

        assert df_dict["hand__t1"] == {0: False}
        assert df_dict["hand__t1__t2"] == {0: False}
        assert df_dict["hand__t1__t2__t3"] == {0: True}

    def test_land_gemstone(self) -> None:
        pass

    # gemstone
    # lake
