"""Tests for the Proportion feature group."""

import pandas as pd

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.feature_groups.statistics import Proportion  # noqa: F401
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
            Feature("hand__t1__proportion", options=opts),
            Feature("MulliganResult", options=opts),
            Feature("hand__t1", options=opts),
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    return pd.concat(dfs, axis=1)


def test_proportion_column_exists() -> None:
    """hand__t1__proportion should produce a float column."""
    df = _run(50)
    assert "hand__t1__proportion" in df.columns


def test_proportion_only_on_kept_rows() -> None:
    """Non-kept rows should have NaN for proportion."""
    df = _run(50)
    not_kept = df[~df["MulliganResult"]]
    assert not_kept["hand__t1__proportion"].isna().all()


def test_proportion_value_is_correct() -> None:
    """Proportion should match manually computed value."""
    df = _run(200)
    kept = df[df["MulliganResult"]]
    expected = kept["hand__t1"].sum() / len(kept)
    actual = kept["hand__t1__proportion"].iloc[0]
    assert abs(actual - expected) < 1e-9


def test_proportion_between_zero_and_one() -> None:
    """Proportion must be in [0, 1]."""
    df = _run(100)
    kept = df[df["MulliganResult"]]
    vals = kept["hand__t1__proportion"]
    assert (vals >= 0.0).all()
    assert (vals <= 1.0).all()
