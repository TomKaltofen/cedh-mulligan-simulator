"""Tests for MeanMulliganDepth and AverageKeptHandSize feature groups."""

from typing import Any

import polars as pl

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.feature_groups.statistics import AverageKeptHandSize, MeanMulliganDepth  # noqa: F401
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider
from tests.feature_groups.test_data_providers import SpecificHandProvider


def _concat_results(results: list[Any]) -> pl.DataFrame:
    seen_cols: set[str] = set()
    parts: list[pl.DataFrame] = []
    for r in results:
        if isinstance(r, pl.DataFrame):
            new_cols = [c for c in r.columns if c not in seen_cols]
            if new_cols:
                parts.append(r.select(new_cols))
                seen_cols.update(new_cols)
    return pl.concat(parts, how="horizontal")


def _opts(n: int) -> Options:
    return Options(
        group={"scenario_id": "test", "n_simulations": n},
        context={"card_registry": BRAIDS_REGISTRY, "commander_cost": BRAIDS_COST},
    )


def _run(n: int) -> pl.DataFrame:
    opts = _opts(n)
    result = mlodaAPI.run_all(
        features=[
            Feature("MeanMulliganDepth", options=opts),
            Feature("AverageKeptHandSize", options=opts),
            Feature("MulliganResult", options=opts),
            Feature("mulligan_count", options=opts),
        ],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    return _concat_results(result)


def test_columns_exist() -> None:
    """Both stat columns should be present."""
    df = _run(50)
    assert "MeanMulliganDepth" in df.columns
    assert "AverageKeptHandSize" in df.columns


def test_stats_only_on_kept_rows() -> None:
    """Non-kept rows should have null."""
    df = _run(50)
    not_kept = df.filter(~pl.col("MulliganResult"))
    assert not_kept["MeanMulliganDepth"].is_null().all()
    assert not_kept["AverageKeptHandSize"].is_null().all()


def test_mean_mulligan_depth_is_correct() -> None:
    """Mean depth should match manual calculation."""
    df = _run(200)
    kept = df.filter(pl.col("MulliganResult"))
    raw_exp = kept["mulligan_count"].mean()
    expected = float(raw_exp) if isinstance(raw_exp, (int, float)) else 0.0
    actual = float(kept["MeanMulliganDepth"][0] or 0.0)
    assert abs(actual - expected) < 1e-9


def test_average_kept_hand_size_is_correct() -> None:
    """Average hand size should match manual calculation."""
    df = _run(200)
    kept = df.filter(pl.col("MulliganResult"))
    raw_exp = kept["MulliganResult~kept_at"].mean()
    expected = float(raw_exp) if isinstance(raw_exp, (int, float)) else 0.0
    actual = float(kept["AverageKeptHandSize"][0] or 0.0)
    assert abs(actual - expected) < 1e-9


def test_mean_mulligan_depth_range() -> None:
    """Depth should be in [0, 3] (max 4 mulligan steps → indices 0-3)."""
    df = _run(100)
    kept = df.filter(pl.col("MulliganResult"))
    val = float(kept["MeanMulliganDepth"][0] or 0.0)
    assert 0.0 <= val <= 3.0


def test_average_hand_size_range() -> None:
    """Hand size should be in [4, 7] (London Mulligan)."""
    df = _run(100)
    kept = df.filter(pl.col("MulliganResult"))
    val = float(kept["AverageKeptHandSize"][0] or 0.0)
    assert 4.0 <= val <= 7.0
