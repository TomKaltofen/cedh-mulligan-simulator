"""Proportion: chained feature ``<source>__proportion``."""

from typing import Any, Optional, Set

import polars as pl

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options


class Proportion(FeatureGroup):
    """Success rate of a boolean feature among kept hands.

    Usage: ``Feature("hand__t1__proportion")``

    Computes the proportion of ``True`` values for the source column
    among rows where ``MulliganResult`` is ``True``.  The scalar result
    is broadcast to every kept row; non-kept rows receive ``None``.
    """

    @classmethod
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, data_access_collection: Any = None) -> bool:
        return str(feature_name).endswith("__proportion")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        source = feature_name.name.replace("__proportion", "")
        return {Feature(source, options=options), Feature("MulliganResult", options=options)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pl.DataFrame = data
        kept_mask = df["MulliganResult"].cast(pl.Boolean)
        n_kept = int(kept_mask.sum() or 0)

        new_cols = []
        for fname in features.get_all_names():
            source = fname.replace("__proportion", "")
            if n_kept > 0:
                raw = df.filter(kept_mask)[source].sum()
                proportion = float(raw or 0) / n_kept
            else:
                proportion = 0.0
            new_cols.append(pl.when(kept_mask).then(proportion).otherwise(None).alias(fname))

        return df.with_columns(new_cols)
