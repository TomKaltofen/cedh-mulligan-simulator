"""Tests for Serum Powder mulligan replacement in HandGenerator."""

import pandas as pd

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from card_database.colorless import SERUM_POWDER
from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.card_registry import build_registry
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider, SpecificHandProvider

_SP_REGISTRY = build_registry(*BRAIDS_REGISTRY.values(), SERUM_POWDER)


def _opts_with_sp(n: int = 30, mulligan_steps: int = 4) -> Options:
    return Options(
        group={"n_simulations": n, "mulligan_steps": mulligan_steps},
        context={"card_registry": _SP_REGISTRY, "commander_cost": BRAIDS_COST},
    )


def _opts_without_sp(n: int = 30, mulligan_steps: int = 4) -> Options:
    return Options(
        group={"n_simulations": n, "mulligan_steps": mulligan_steps},
        context={"card_registry": BRAIDS_REGISTRY, "commander_cost": BRAIDS_COST},
    )


def _run_hand_generator(opts: Options) -> pd.DataFrame:
    result = mlodaAPI.run_all(
        features=[
            Feature("hand", options=opts),
            Feature("simulation_id", options=opts),
            Feature("remaining_library", options=opts),
            Feature("mulligan_count", options=opts),
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    return pd.concat(dfs, axis=1)


def test_sp_generates_extra_rows() -> None:
    """When SP is drawn, an extra row is emitted for the replacement hand.

    With 500 sims and ~7% SP draw rate per step, some sims must have > 4 rows.
    """
    df = _run_hand_generator(_opts_with_sp(n=500))
    rows_per_sim = df.groupby("simulation_id").size()
    assert (rows_per_sim > 4).any(), "Some sims should have extra SP-replacement rows"
    # All sims have at least 4 (the base mulligan steps)
    assert (rows_per_sim >= 4).all()
    # No sim should have more than 5 rows (SP is singleton, so at most 1 extra row)
    assert (rows_per_sim <= 5).all()


def test_sp_replacement_row_does_not_contain_sp() -> None:
    """The SP replacement row (extra row) should not contain serum_powder (it was exiled)."""
    df = _run_hand_generator(_opts_with_sp(n=500))
    # Find sims with extra rows (SP was triggered)
    rows_per_sim = df.groupby("simulation_id").size()
    sp_sims = rows_per_sim[rows_per_sim > 4].index

    for sim_id in sp_sims:
        sim_rows = df[df["simulation_id"] == sim_id].sort_values("mulligan_count")
        # Find duplicate mulligan_count (the SP original + replacement pair)
        mull_counts = sim_rows["mulligan_count"].tolist()
        for i in range(len(mull_counts) - 1):
            if mull_counts[i] == mull_counts[i + 1]:
                # Second row of the pair is the SP replacement
                replacement_hand = sim_rows.iloc[i + 1]["hand"]
                assert "serum_powder" not in replacement_hand, (
                    f"SP replacement hand should not contain serum_powder: {replacement_hand}"
                )


def test_without_serum_powder_library_sizes_unchanged() -> None:
    """Without SP in the registry, library sizes should follow standard London Mulligan pattern."""
    df = _run_hand_generator(_opts_without_sp(n=30))
    expected = {0: 92, 1: 92, 2: 93, 3: 94}
    for _, row in df.iterrows():
        mull_count = row["mulligan_count"]
        assert len(row["remaining_library"]) == expected[mull_count]


def test_sp_replacement_has_smaller_library() -> None:
    """SP replacement rows have smaller libraries (7 cards exiled from the game)."""
    df = _run_hand_generator(_opts_with_sp(n=2000))

    # Find sims with extra rows (SP was triggered)
    rows_per_sim = df.groupby("simulation_id").size()
    sp_sims = rows_per_sim[rows_per_sim > 4].index
    assert len(sp_sims) > 0, "With 2000 sims, some should trigger SP"

    for sim_id in sp_sims:
        sim_rows = df[df["simulation_id"] == sim_id].sort_values("mulligan_count")
        mull_counts = sim_rows["mulligan_count"].tolist()
        for i in range(len(mull_counts) - 1):
            if mull_counts[i] == mull_counts[i + 1]:
                # Original hand row and SP replacement row
                orig_hand_size = len(sim_rows.iloc[i]["hand"])
                orig_lib = len(sim_rows.iloc[i]["remaining_library"])
                repl_lib = len(sim_rows.iloc[i + 1]["remaining_library"])
                # SP replacement library should be hand_size smaller (exiled hand + drew new)
                assert repl_lib == orig_lib - orig_hand_size, (
                    f"SP replacement library ({repl_lib}) should be {orig_hand_size} less than original ({orig_lib})"
                )
                break  # Only one SP use per sim
