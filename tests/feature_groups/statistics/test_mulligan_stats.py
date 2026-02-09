"""Tests for MeanMulliganDepth and AverageKeptHandSize feature groups."""

import pandas as pd

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.feature_groups.statistics import AverageKeptHandSize, MeanMulliganDepth  # noqa: F401
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider
from tests.test_integration_mloda_specific_hands import SpecificHandProvider


def _opts(n: int) -> Options:
    return Options(
        group={"scenario_id": "test", "n_simulations": n},
        context={"card_registry": BRAIDS_REGISTRY, "commander_cost": BRAIDS_COST},
    )


def _run(n: int) -> pd.DataFrame:
    opts = _opts(n)
    result = mlodaAPI.run_all(
        features=[
            Feature("MeanMulliganDepth", options=opts),
            Feature("AverageKeptHandSize", options=opts),
            Feature("MulliganResult", options=opts),
            Feature("mulligan_count", options=opts),
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    return pd.concat(dfs, axis=1)


def test_columns_exist() -> None:
    """Both stat columns should be present."""
    df = _run(50)
    assert "MeanMulliganDepth" in df.columns
    assert "AverageKeptHandSize" in df.columns


def test_stats_only_on_kept_rows() -> None:
    """Non-kept rows should have NaN."""
    df = _run(50)
    not_kept = df[~df["MulliganResult"]]
    assert not_kept["MeanMulliganDepth"].isna().all()
    assert not_kept["AverageKeptHandSize"].isna().all()


def test_mean_mulligan_depth_is_correct() -> None:
    """Mean depth should match manual calculation."""
    df = _run(200)
    kept = df[df["MulliganResult"]]
    expected = kept["mulligan_count"].mean()
    actual = kept["MeanMulliganDepth"].iloc[0]
    assert abs(actual - expected) < 1e-9


def test_average_kept_hand_size_is_correct() -> None:
    """Average hand size should match manual calculation."""
    df = _run(200)
    kept = df[df["MulliganResult"]]
    expected = kept["MulliganResult~kept_at"].mean()
    actual = kept["AverageKeptHandSize"].iloc[0]
    assert abs(actual - expected) < 1e-9


def test_mean_mulligan_depth_range() -> None:
    """Depth should be in [0, 3] (max 4 mulligan steps → indices 0-3)."""
    df = _run(100)
    kept = df[df["MulliganResult"]]
    val = kept["MeanMulliganDepth"].iloc[0]
    assert 0.0 <= val <= 3.0


def test_average_hand_size_range() -> None:
    """Hand size should be in [4, 7] (London Mulligan)."""
    df = _run(100)
    kept = df[df["MulliganResult"]]
    val = kept["AverageKeptHandSize"].iloc[0]
    assert 4.0 <= val <= 7.0
