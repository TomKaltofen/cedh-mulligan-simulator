"""Tests for the Convergence feature group."""

from typing import Any

import polars as pl

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.feature_groups.statistics import Convergence  # noqa: F401
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
            Feature("hand__t1__convergence", options=opts),
            Feature("MulliganResult", options=opts),
            Feature("hand__t1", options=opts),
            Feature("simulation_id", options=opts),
        ],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    return _concat_results(result)


def test_convergence_column_exists() -> None:
    """hand__t1__convergence should produce a float column."""
    df = _run(50)
    assert "hand__t1__convergence" in df.columns


def test_convergence_only_on_kept_rows() -> None:
    """Non-kept rows should have null."""
    df = _run(50)
    not_kept = df.filter(~pl.col("MulliganResult"))
    assert not_kept["hand__t1__convergence"].is_null().all()


def test_convergence_between_zero_and_one() -> None:
    """Running mean of a boolean column must stay in [0, 1]."""
    df = _run(100)
    kept = df.filter(pl.col("MulliganResult"))
    vals = kept["hand__t1__convergence"].drop_nulls()
    assert (vals >= 0.0).all()
    assert (vals <= 1.0).all()


def test_convergence_last_value_matches_proportion() -> None:
    """The final running-mean value should equal the overall proportion."""
    df = _run(200)
    kept = df.filter(pl.col("MulliganResult")).sort("simulation_id")
    total = float(kept["hand__t1"].sum() or 0)
    expected = total / len(kept)
    last_val = float(kept["hand__t1__convergence"][-1] or 0.0)
    assert abs(last_val - expected) < 1e-9
