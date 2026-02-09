"""Convergence — chained feature: ``<source>__convergence``."""

from typing import Any, Optional, Set

import pandas as pd

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options


class Convergence(FeatureGroup):
    """Running cumulative mean of a boolean feature across kept hands.

    Usage: ``Feature("hand__t1__convergence")``

    Computes the expanding mean of the source column across kept hands
    ordered by ``simulation_id``.  Useful for checking whether the
    estimate has stabilised as *N* grows.

    Non-kept rows receive ``NaN``.
    """

    @classmethod
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, data_access_collection: Any = None) -> bool:
        return str(feature_name).endswith("__convergence")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        source = feature_name.name.replace("__convergence", "")
        return {
            Feature(source, options=options),
            Feature("MulliganResult", options=options),
            Feature("simulation_id", options=options),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pd.DataFrame = data
        one_feature = features.name_of_one_feature
        if one_feature is None:
            raise ValueError("Convergence: no feature name found")
        fname: str = one_feature.name
        source = fname.replace("__convergence", "")

        kept_mask = df["MulliganResult"].astype(bool)
        kept = df.loc[kept_mask].sort_values("simulation_id")
        running_mean: pd.Series[Any] = kept[source].expanding().mean()

        df[fname] = float("nan")
        df.loc[kept.index, fname] = running_mean.values
        return df
