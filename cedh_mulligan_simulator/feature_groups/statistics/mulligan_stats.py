"""Aggregate mulligan statistics — mean depth and average kept hand size."""

from typing import Any, Optional, Set

import polars as pl

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options


class MeanMulliganDepth(FeatureGroup):
    """Average number of mulligans before keeping.

    Usage: ``Feature("MeanMulliganDepth")``

    Computes the mean ``mulligan_count`` across kept hands and broadcasts
    the scalar to every kept row.  Non-kept rows receive ``None``.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature("MulliganResult", options=options),
            Feature("mulligan_count", options=options),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pl.DataFrame = data
        kept = df["MulliganResult"].cast(pl.Boolean)
        if kept.any():
            raw = df.filter(kept)["mulligan_count"].mean()
            mean_depth = float(raw) if isinstance(raw, (int, float)) else 0.0
        else:
            mean_depth = 0.0

        return df.with_columns(pl.when(kept).then(mean_depth).otherwise(None).alias("MeanMulliganDepth"))


class AverageKeptHandSize(FeatureGroup):
    """Average hand size at the moment of keeping.

    Usage: ``Feature("AverageKeptHandSize")``

    Computes the mean ``kept_at`` across kept hands and broadcasts
    the scalar to every kept row.  Non-kept rows receive ``None``.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature("MulliganResult", options=options),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pl.DataFrame = data
        kept = df["MulliganResult"].cast(pl.Boolean)
        if kept.any():
            raw = df.filter(kept)["MulliganResult~kept_at"].mean()
            avg_size = float(raw) if isinstance(raw, (int, float)) else 0.0
        else:
            avg_size = 0.0

        return df.with_columns(pl.when(kept).then(avg_size).otherwise(None).alias("AverageKeptHandSize"))
