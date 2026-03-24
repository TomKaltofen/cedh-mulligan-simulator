"""Tests for HandGenerator feature group."""

import pandas as pd

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider
from tests.feature_groups.test_data_providers import SpecificHandProvider


def _hand_opts(n: int, mulligan_steps: int = 4) -> Options:
    return Options(
        group={"n_simulations": n, "mulligan_steps": mulligan_steps},
        context={"card_registry": BRAIDS_REGISTRY, "commander_cost": BRAIDS_COST},
    )


def test_hand_generator_produces_mulligan_rows() -> None:
    """HandGenerator should produce N x mulligan_steps rows."""
    n = 50
    steps = 4
    opts = _hand_opts(n, steps)
    result = mlodaAPI.run_all(
        features=[
            Feature("hand", options=opts),
            Feature("simulation_id", options=opts),
            Feature("mulligan_count", options=opts),
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    df = pd.concat(dfs, axis=1)
    assert "hand" in df.columns
    assert "simulation_id" in df.columns
    assert "mulligan_count" in df.columns
    assert len(df) == n * steps


def test_hand_generator_mulligan_count_values() -> None:
    """mulligan_count should have values 0..steps-1 for each simulation."""
    n = 20
    steps = 4
    opts = _hand_opts(n, steps)
    result = mlodaAPI.run_all(
        features=[
            Feature("simulation_id", options=opts),
            Feature("mulligan_count", options=opts),
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    df = pd.concat(dfs, axis=1)
    assert set(df["mulligan_count"].unique()) == {0, 1, 2, 3}
    for _sim_id, group in df.groupby("simulation_id"):
        assert sorted(group["mulligan_count"].tolist()) == [0, 1, 2, 3]


def test_hand_generator_hand_size() -> None:
    """London Mulligan hand sizes: 7, 7 (free), 6, 5."""
    n = 20
    steps = 4
    opts = _hand_opts(n, steps)
    result = mlodaAPI.run_all(
        features=[
            Feature("hand", options=opts),
            Feature("mulligan_count", options=opts),
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    df = pd.concat(dfs, axis=1)
    expected_sizes = {0: 7, 1: 7, 2: 6, 3: 5}
    for _, row in df.iterrows():
        mull_count = row["mulligan_count"]
        assert len(row["hand"]) == expected_sizes[mull_count], (
            f"mulligan_count={mull_count} should have {expected_sizes[mull_count]} cards"
        )


def test_hand_generator_remaining_library() -> None:
    """remaining_library column should exist and contain lists of cards."""
    n = 20
    steps = 4
    opts = _hand_opts(n, steps)
    result = mlodaAPI.run_all(
        features=[
            Feature("hand", options=opts),
            Feature("remaining_library", options=opts),
            Feature("mulligan_count", options=opts),
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    df = pd.concat(dfs, axis=1)
    assert "remaining_library" in df.columns
    assert len(df) == n * steps
    # London Mulligan: hand sizes 7, 7, 6, 5 -> library sizes 92, 92, 93, 94
    expected_library_sizes = {0: 92, 1: 92, 2: 93, 3: 94}
    for _, row in df.iterrows():
        library = row["remaining_library"]
        mull_count = row["mulligan_count"]
        assert isinstance(library, list)
        assert len(library) == expected_library_sizes[mull_count], (
            f"mulligan_count={mull_count} should have {expected_library_sizes[mull_count]} cards in library"
        )


def test_hand_generator_singleton() -> None:
    """Each hand should contain no duplicate non-filler cards."""
    result = mlodaAPI.run_all(
        features=[Feature("hand", options=_hand_opts(20))],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    df = result[0]
    for hand in df["hand"]:
        non_filler = [c for c in hand if c != "filler"]
        assert len(non_filler) == len(set(non_filler)), f"Duplicate non-filler cards: {hand}"
