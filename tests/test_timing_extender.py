"""Tests for TimingExtender."""

import logging

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.extenders import TimingExtender
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider
from tests.test_integration_mloda_specific_hands import SpecificHandProvider


def _hand_opts(n: int = 10, mulligan_steps: int = 4) -> Options:
    return Options(
        group={"n_simulations": n, "mulligan_steps": mulligan_steps},
        context={"card_registry": BRAIDS_REGISTRY, "commander_cost": BRAIDS_COST},
    )


def test_timing_extender_logs_feature_calculation_time(caplog: pytest.LogCaptureFixture) -> None:
    """TimingExtender should log timing information for feature calculations."""
    opts = _hand_opts(n=5)

    with caplog.at_level(logging.INFO):
        result = mlodaAPI.run_all(
            features=[Feature("hand", options=opts)],
            compute_frameworks={PandasDataFrame},
            function_extender={TimingExtender()},
            plugin_collector=PluginCollector.disabled_feature_groups(
                {HandMulliganTestDataProvider, SpecificHandProvider}
            ),
        )

    # Verify we got a result
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    assert len(dfs) > 0

    # Verify timing was logged (look for pattern with seconds)
    timing_logs = [record for record in caplog.records if "s" in record.message and ":" in record.message]
    assert len(timing_logs) > 0, f"Expected timing logs, got: {[r.message for r in caplog.records]}"


def test_timing_extender_does_not_affect_results() -> None:
    """TimingExtender should not affect the calculation results."""
    opts = _hand_opts(n=10)

    # Run without extender
    result_without = mlodaAPI.run_all(
        features=[Feature("hand", options=opts)],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )

    # Run with extender
    result_with = mlodaAPI.run_all(
        features=[Feature("hand", options=opts)],
        compute_frameworks={PandasDataFrame},
        function_extender={TimingExtender()},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )

    # Both should produce DataFrames with the same structure
    df_without = [r for r in result_without if isinstance(r, pd.DataFrame)][0]
    df_with = [r for r in result_with if isinstance(r, pd.DataFrame)][0]

    assert "hand" in df_without.columns
    assert "hand" in df_with.columns
    assert len(df_without) == len(df_with)
