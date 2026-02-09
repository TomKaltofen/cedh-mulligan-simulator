"""Aggregate mulligan statistics — mean depth and average kept hand size."""

from typing import Any, Optional, Set

import pandas as pd

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options


class MeanMulliganDepth(FeatureGroup):
    """Average number of mulligans before keeping.

    Usage: ``Feature("MeanMulliganDepth")``

    Computes the mean ``mulligan_count`` across kept hands and broadcasts
    the scalar to every kept row.  Non-kept rows receive ``NaN``.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature("MulliganResult", options=options),
            Feature("mulligan_count", options=options),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pd.DataFrame = data
        kept = df["MulliganResult"].astype(bool)
        mean_depth = float(df.loc[kept, "mulligan_count"].mean()) if kept.any() else 0.0

        df["MeanMulliganDepth"] = float("nan")
        df.loc[kept, "MeanMulliganDepth"] = mean_depth
        return df


class AverageKeptHandSize(FeatureGroup):
    """Average hand size at the moment of keeping.

    Usage: ``Feature("AverageKeptHandSize")``

    Computes the mean ``kept_at`` across kept hands and broadcasts
    the scalar to every kept row.  Non-kept rows receive ``NaN``.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature("MulliganResult", options=options),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pd.DataFrame = data
        kept = df["MulliganResult"].astype(bool)
        avg_size = float(df.loc[kept, "MulliganResult~kept_at"].mean()) if kept.any() else 0.0

        df["AverageKeptHandSize"] = float("nan")
        df.loc[kept, "AverageKeptHandSize"] = avg_size
        return df
