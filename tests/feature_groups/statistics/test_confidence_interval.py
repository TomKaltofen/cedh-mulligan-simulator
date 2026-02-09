"""Tests for the CILower / CIUpper feature groups."""

import pandas as pd

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.feature_groups.statistics import CILower, CIUpper  # noqa: F401
from cedh_mulligan_simulator.feature_groups.statistics.confidence_interval import _wilson_interval
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
            Feature("hand__t1__ci_lower", options=opts),
            Feature("hand__t1__ci_upper", options=opts),
            Feature("MulliganResult", options=opts),
            Feature("hand__t1", options=opts),
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    return pd.concat(dfs, axis=1)


def test_ci_columns_exist() -> None:
    """CI lower and upper columns should be present."""
    df = _run(50)
    assert "hand__t1__ci_lower" in df.columns
    assert "hand__t1__ci_upper" in df.columns


def test_ci_only_on_kept_rows() -> None:
    """Non-kept rows should have NaN for CI bounds."""
    df = _run(50)
    not_kept = df[~df["MulliganResult"]]
    assert not_kept["hand__t1__ci_lower"].isna().all()
    assert not_kept["hand__t1__ci_upper"].isna().all()


def test_ci_lower_le_upper() -> None:
    """Lower bound must be <= upper bound."""
    df = _run(200)
    kept = df[df["MulliganResult"]]
    assert (kept["hand__t1__ci_lower"] <= kept["hand__t1__ci_upper"]).all()


def test_ci_bounds_contain_proportion() -> None:
    """The actual proportion should lie within the CI bounds."""
    df = _run(200)
    kept = df[df["MulliganResult"]]
    proportion = kept["hand__t1"].sum() / len(kept)
    lower = kept["hand__t1__ci_lower"].iloc[0]
    upper = kept["hand__t1__ci_upper"].iloc[0]
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
