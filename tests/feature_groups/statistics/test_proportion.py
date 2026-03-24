"""Tests for the Proportion feature group."""

from typing import Any

import polars as pl

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.feature_groups.statistics import Proportion  # noqa: F401
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
            Feature("hand__t1__proportion", options=opts),
            Feature("MulliganResult", options=opts),
            Feature("hand__t1", options=opts),
        ],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    return _concat_results(result)


def test_proportion_column_exists() -> None:
    """hand__t1__proportion should produce a float column."""
    df = _run(50)
    assert "hand__t1__proportion" in df.columns


def test_proportion_only_on_kept_rows() -> None:
    """Non-kept rows should have null for proportion."""
    df = _run(50)
    not_kept = df.filter(~pl.col("MulliganResult"))
    assert not_kept["hand__t1__proportion"].is_null().all()


def test_proportion_value_is_correct() -> None:
    """Proportion should match manually computed value."""
    df = _run(200)
    kept = df.filter(pl.col("MulliganResult"))
    expected = float(kept["hand__t1"].sum() or 0) / len(kept)
    actual = float(kept["hand__t1__proportion"][0] or 0.0)
    assert abs(actual - expected) < 1e-9


def test_proportion_between_zero_and_one() -> None:
    """Proportion must be in [0, 1]."""
    df = _run(100)
    kept = df.filter(pl.col("MulliganResult"))
    vals = kept["hand__t1__proportion"].drop_nulls()
    assert (vals >= 0.0).all()
    assert (vals <= 1.0).all()


def test_two_proportions_batched() -> None:
    """Both proportion features should be present when requested together (batching case)."""
    opts = _opts(100)
    result = mlodaAPI.run_all(
        features=[
            Feature("hand__t1__proportion", options=opts),
            Feature("hand__t1__t2__proportion", options=opts),
            Feature("MulliganResult", options=opts),
        ],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    df = _concat_results(result)
    assert "hand__t1__proportion" in df.columns
    assert "hand__t1__t2__proportion" in df.columns
