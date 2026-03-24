"""Confidence interval feature groups using the Wilson score method."""

import math
from typing import Any, Optional, Set, Tuple

import polars as pl

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options

_DEFAULT_Z = 1.96  # 95 % confidence


def _wilson_interval(successes: int, n: int, z: float = _DEFAULT_Z) -> Tuple[float, float, float]:
    """Return ``(lower, point, upper)`` for a Wilson score interval.

    The Wilson score interval provides better coverage than the normal
    approximation, especially for proportions near 0 or 1 and small *n*.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)

    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator
    return (max(centre - margin, 0.0), p, min(centre + margin, 1.0))


class CILower(FeatureGroup):
    """Lower bound of the Wilson score 95 % confidence interval.

    Usage: ``Feature("hand__t1__ci_lower")``

    Computes the lower bound for a boolean source feature among kept hands.
    Set ``ci_z`` in options to override the default z-value (1.96 = 95 %).
    """

    @classmethod
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, data_access_collection: Any = None) -> bool:
        return str(feature_name).endswith("__ci_lower")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        source = feature_name.name.replace("__ci_lower", "")
        return {Feature(source, options=options), Feature("MulliganResult", options=options)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pl.DataFrame = data
        one_feature = features.name_of_one_feature
        if one_feature is None:
            raise ValueError("CILower: no feature name found")
        fname: str = one_feature.name
        source = fname.replace("__ci_lower", "")
        z = float(features.get_options_key("ci_z") or _DEFAULT_Z)

        kept = df["MulliganResult"].cast(pl.Boolean)
        n = int(kept.sum() or 0)
        raw = df.filter(kept)[source].sum()
        successes = int(raw or 0)
        lower, _, _ = _wilson_interval(successes, n, z)

        return df.with_columns(pl.when(kept).then(lower).otherwise(None).alias(fname))


class CIUpper(FeatureGroup):
    """Upper bound of the Wilson score 95 % confidence interval.

    Usage: ``Feature("hand__t1__ci_upper")``
    """

    @classmethod
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, data_access_collection: Any = None) -> bool:
        return str(feature_name).endswith("__ci_upper")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        source = feature_name.name.replace("__ci_upper", "")
        return {Feature(source, options=options), Feature("MulliganResult", options=options)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pl.DataFrame = data
        one_feature = features.name_of_one_feature
        if one_feature is None:
            raise ValueError("CIUpper: no feature name found")
        fname: str = one_feature.name
        source = fname.replace("__ci_upper", "")
        z = float(features.get_options_key("ci_z") or _DEFAULT_Z)

        kept = df["MulliganResult"].cast(pl.Boolean)
        n = int(kept.sum() or 0)
        raw = df.filter(kept)[source].sum()
        successes = int(raw or 0)
        _, _, upper = _wilson_interval(successes, n, z)

        return df.with_columns(pl.when(kept).then(upper).otherwise(None).alias(fname))
