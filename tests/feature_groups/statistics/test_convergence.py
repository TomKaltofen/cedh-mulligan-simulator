"""Tests for the Convergence feature group."""

import pandas as pd

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.feature_groups.statistics import Convergence  # noqa: F401
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
            Feature("hand__t1__convergence", options=opts),
            Feature("MulliganResult", options=opts),
            Feature("hand__t1", options=opts),
            Feature("simulation_id", options=opts),
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    return pd.concat(dfs, axis=1)


def test_convergence_column_exists() -> None:
    """hand__t1__convergence should produce a float column."""
    df = _run(50)
    assert "hand__t1__convergence" in df.columns


def test_convergence_only_on_kept_rows() -> None:
    """Non-kept rows should have NaN."""
    df = _run(50)
    not_kept = df[~df["MulliganResult"]]
    assert not_kept["hand__t1__convergence"].isna().all()


def test_convergence_between_zero_and_one() -> None:
    """Running mean of a boolean column must stay in [0, 1]."""
    df = _run(100)
    kept = df[df["MulliganResult"]]
    vals = kept["hand__t1__convergence"].dropna()
    assert (vals >= 0.0).all()
    assert (vals <= 1.0).all()


def test_convergence_last_value_matches_proportion() -> None:
    """The final running-mean value should equal the overall proportion."""
    df = _run(200)
    kept = df[df["MulliganResult"]].sort_values("simulation_id")
    expected = kept["hand__t1"].sum() / len(kept)
    last_val = kept["hand__t1__convergence"].iloc[-1]
    assert abs(last_val - expected) < 1e-9
