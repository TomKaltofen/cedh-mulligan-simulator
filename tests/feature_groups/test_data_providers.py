"""Shared test data providers for integration tests.

These FeatureGroup subclasses provide mock data for testing feature groups
that depend on 'hand' and 'MulliganResult' features without going through
the full HandGenerator pipeline.

Usage:
    1. Import the provider you need
    2. Set provider._test_data = your_dataframe before calling mlodaAPI.run_all
    3. Use PluginCollector to disable HandGenerator and other providers
"""

from typing import Any, Optional

import polars as pl

from mloda.provider import DataCreator, FeatureGroup, FeatureSet


class HandMulliganTestDataProvider(FeatureGroup):
    """Provides test data for hand and MulliganResult features.

    Set _test_data to a DataFrame with 'hand' and 'MulliganResult' columns
    before calling mlodaAPI.run_all.
    """

    _test_data: Optional[pl.DataFrame] = None

    @classmethod
    def input_data(cls) -> Optional[Any]:
        return DataCreator({"hand", "MulliganResult"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return cls._test_data
