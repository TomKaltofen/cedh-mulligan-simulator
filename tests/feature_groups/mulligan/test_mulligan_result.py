"""Tests for MulliganResult feature group."""

import polars as pl

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.feature_groups.mulligan.mulligan_result import _kept_hand_size
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider, SpecificHandProvider
from tests.helpers import concat_results


def _mulligan_opts(n: int) -> Options:
    return Options(
        group={"scenario_id": "test", "n_simulations": n},
        context={"card_registry": BRAIDS_REGISTRY, "commander_cost": BRAIDS_COST},
    )


def _run_mulligan(n: int) -> pl.DataFrame:
    """Run MulliganResult pipeline and merge all result DataFrames."""
    opts = _mulligan_opts(n)
    result = mlodaAPI.run_all(
        features=[
            Feature("MulliganResult", options=opts),
            Feature("simulation_id", options=opts),
            Feature("mulligan_count", options=opts),
            Feature("hand__t1", options=opts),
            Feature("hand__t1__t2", options=opts),
        ],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    return concat_results(result)


def test_mulligan_result_via_mloda() -> None:
    """MulliganResult should produce expected columns via mlodaAPI."""
    n = 100
    df = _run_mulligan(n)
    assert "MulliganResult" in df.columns
    assert "MulliganResult~kept_at" in df.columns
    assert df["MulliganResult"].dtype == pl.Boolean
    # Total rows = N x mulligan_steps (default 4)
    assert len(df) == n * 4


def test_exactly_one_keep_per_simulation() -> None:
    """Exactly one row per simulation_id should have MulliganResult == True."""
    n = 100
    df = _run_mulligan(n)
    kept = df.filter(pl.col("MulliganResult"))
    assert len(kept) == n
    assert kept["simulation_id"].n_unique() == n


def test_kept_at_values() -> None:
    """kept_at values should be valid London Mulligan hand sizes."""
    df = _run_mulligan(200)
    kept = df.filter(pl.col("MulliganResult"))
    kept_at = kept["MulliganResult~kept_at"].to_list()
    assert all(v in {7, 6, 5, 4} for v in kept_at)


def test_kept_hand_size_helper() -> None:
    """_kept_hand_size should return correct London Mulligan sizes."""
    assert _kept_hand_size(0) == 7
    assert _kept_hand_size(1) == 7
    assert _kept_hand_size(2) == 6
    assert _kept_hand_size(3) == 5
    assert _kept_hand_size(4) == 4
