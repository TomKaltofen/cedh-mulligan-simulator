"""Proportion — chained feature: ``<source>__proportion``."""

from typing import Any, Optional, Set

import pandas as pd

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options


class Proportion(FeatureGroup):
    """Success rate of a boolean feature among kept hands.

    Usage: ``Feature("hand__t1__proportion")``

    Computes the proportion of ``True`` values for the source column
    among rows where ``MulliganResult`` is ``True``.  The scalar result
    is broadcast to every kept row; non-kept rows receive ``NaN``.
    """

    @classmethod
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, data_access_collection: Any = None) -> bool:
        return str(feature_name).endswith("__proportion")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        source = feature_name.name.replace("__proportion", "")
        return {Feature(source, options=options), Feature("MulliganResult", options=options)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pd.DataFrame = data
        one_feature = features.name_of_one_feature
        if one_feature is None:
            raise ValueError("Proportion: no feature name found")
        fname: str = one_feature.name
        source = fname.replace("__proportion", "")

        kept_mask = df["MulliganResult"].astype(bool)
        n_kept = int(kept_mask.sum())
        proportion = float(df.loc[kept_mask, source].sum()) / n_kept if n_kept > 0 else 0.0

        df[fname] = float("nan")
        df.loc[kept_mask, fname] = proportion
        return df
