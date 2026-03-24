"""Convergence — chained feature: ``<source>__convergence``."""

from typing import Any, Optional, Set

import polars as pl

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options


class Convergence(FeatureGroup):
    """Running cumulative mean of a boolean feature across kept hands.

    Usage: ``Feature("hand__t1__convergence")``

    Computes the expanding mean of the source column across kept hands
    ordered by ``simulation_id``.  Useful for checking whether the
    estimate has stabilised as *N* grows.

    Non-kept rows receive ``null``.
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
        df: pl.DataFrame = data
        one_feature = features.name_of_one_feature
        if one_feature is None:
            raise ValueError("Convergence: no feature name found")
        fname: str = one_feature.name
        source = fname.replace("__convergence", "")

        df = df.with_row_index("__row_idx")
        kept_mask = df["MulliganResult"].cast(pl.Boolean)
        kept = df.filter(kept_mask).sort("simulation_id")

        source_list = kept[source].cast(pl.Float64).to_list()
        running: list[float] = []
        total = 0.0
        for val in source_list:
            total += float(val) if val is not None else 0.0
            running.append(total / (len(running) + 1))

        row_idxs = kept["__row_idx"].to_list()
        running_df = pl.DataFrame({"__row_idx": row_idxs, fname: running})
        df = df.join(running_df, on="__row_idx", how="left").drop("__row_idx")
        return df
