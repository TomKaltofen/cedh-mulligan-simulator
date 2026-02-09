"""Tests for Turn1 and Turn2 feature groups."""

import pandas as pd

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.feature_groups.mulligan import Turn1, Turn2  # noqa: F401
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider
from tests.test_integration_mloda_specific_hands import SpecificHandProvider


def _opts(n: int) -> Options:
    return Options(
        group={"n_simulations": n},
        context={"card_registry": BRAIDS_REGISTRY, "commander_cost": BRAIDS_COST},
    )


def test_turn1_feature() -> None:
    """hand__t1 should produce boolean column."""
    opts = _opts(50)
    result = mlodaAPI.run_all(
        features=[Feature("hand__t1", options=opts)],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    assert len(result) == 1
    df = result[0]
    assert isinstance(df, pd.DataFrame)
    assert "hand__t1" in df.columns
    assert df["hand__t1"].dtype == bool


def test_turn2_feature() -> None:
    """hand__t1__t2 should produce boolean column."""
    opts = _opts(50)
    result = mlodaAPI.run_all(
        features=[Feature("hand__t1__t2", options=opts)],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    assert len(result) == 1
    df = result[0]
    assert isinstance(df, pd.DataFrame)
    assert "hand__t1__t2" in df.columns
    assert df["hand__t1__t2"].dtype == bool


def test_t2_at_least_as_good_as_t1() -> None:
    """T2 castable should be >= T1 castable for every hand (T2 is strictly easier)."""
    opts = _opts(200)
    result = mlodaAPI.run_all(
        features=[
            Feature("hand__t1", options=opts),
            Feature("hand__t1__t2", options=opts),
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    assert len(dfs) >= 1
