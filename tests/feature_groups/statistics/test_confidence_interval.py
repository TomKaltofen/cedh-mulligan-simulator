"""Tests for the CILower / CIUpper feature groups."""

import polars as pl

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.feature_groups.statistics import CILower, CIUpper  # noqa: F401
from cedh_mulligan_simulator.feature_groups.statistics.confidence_interval import _wilson_interval
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider, SpecificHandProvider
from tests.helpers import concat_results


def _opts(n: int) -> Options:
    return Options(
        group={"scenario_id": "test", "n_simulations": n},
        context={"card_registry": BRAIDS_REGISTRY, "commander_cost": BRAIDS_COST},
    )


def _run(n: int) -> pl.DataFrame:
    opts = _opts(n)
    result = mlodaAPI.run_all(
        features=[
            Feature("hand__t1__ci_lower", options=opts),
            Feature("hand__t1__ci_upper", options=opts),
            Feature("MulliganResult", options=opts),
            Feature("hand__t1", options=opts),
        ],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    return concat_results(result)


def test_ci_columns_exist() -> None:
    """CI lower and upper columns should be present."""
    df = _run(50)
    assert "hand__t1__ci_lower" in df.columns
    assert "hand__t1__ci_upper" in df.columns


def test_ci_only_on_kept_rows() -> None:
    """Non-kept rows should have null for CI bounds."""
    df = _run(50)
    not_kept = df.filter(~pl.col("MulliganResult"))
    assert not_kept["hand__t1__ci_lower"].is_null().all()
    assert not_kept["hand__t1__ci_upper"].is_null().all()


def test_ci_lower_le_upper() -> None:
    """Lower bound must be <= upper bound."""
    df = _run(200)
    kept = df.filter(pl.col("MulliganResult"))
    assert (kept["hand__t1__ci_lower"] <= kept["hand__t1__ci_upper"]).all()


def test_ci_bounds_contain_proportion() -> None:
    """The actual proportion should lie within the CI bounds."""
    df = _run(200)
    kept = df.filter(pl.col("MulliganResult"))
    proportion = float(kept["hand__t1"].sum() or 0) / len(kept)
    lower = float(kept["hand__t1__ci_lower"][0] or 0.0)
    upper = float(kept["hand__t1__ci_upper"][0] or 0.0)
    assert lower <= proportion <= upper


def test_wilson_interval_unit() -> None:
    """Direct unit test for _wilson_interval."""
    lower, point, upper = _wilson_interval(50, 100)
    assert abs(point - 0.5) < 1e-9
    assert lower < 0.5 < upper
    assert lower >= 0.0
    assert upper <= 1.0


def test_wilson_interval_zero_n() -> None:
    """Edge case: n=0 should return (0, 0, 0)."""
    assert _wilson_interval(0, 0) == (0.0, 0.0, 0.0)


def test_wilson_interval_all_successes() -> None:
    """When all succeed, upper should be ~1.0."""
    lower, point, upper = _wilson_interval(100, 100)
    assert abs(point - 1.0) < 1e-9
    assert abs(upper - 1.0) < 1e-9
    assert lower > 0.9
