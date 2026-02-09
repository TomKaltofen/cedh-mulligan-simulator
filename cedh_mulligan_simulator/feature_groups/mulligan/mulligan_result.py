"""MulliganResult — aggregates turn evaluations into keep/mulligan decisions."""

from typing import Any, Optional, Set

import pandas as pd

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options


def _kept_hand_size(mulligan_count: int) -> int:
    """London Mulligan hand size: 7, 7(free), 6, 5, 4, ..."""
    return max(0, 7 - max(0, mulligan_count - 1))


class MulliganResult(FeatureGroup):
    """Per simulation: walk mulligan steps, keep first where Turn evaluations are True.

    Options:
        group["max_keep_turn"]: Maximum turn to consider for keep decision (default: 2).
            1 = keep only T1 castable hands
            2 = keep T1 or T2 castable hands
            3 = keep T1, T2, or T3 castable hands
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        max_turn = options.group.get("max_keep_turn", 2) if options.group else 2
        features: Set[Feature] = {
            Feature("hand__t1", options=options),
            Feature("simulation_id", options=options),
            Feature("mulligan_count", options=options),
        }
        if max_turn >= 2:
            features.add(Feature("hand__t1__t2", options=options))
        if max_turn >= 3:
            features.add(Feature("hand__t1__t2__t3", options=options))
        return features

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pd.DataFrame = data
        kept: list[bool] = [False] * len(df)
        kept_at: list[int] = [0] * len(df)

        # Determine which turn columns are available
        turn_cols = ["hand__t1"]
        if "hand__t1__t2" in df.columns:
            turn_cols.append("hand__t1__t2")
        if "hand__t1__t2__t3" in df.columns:
            turn_cols.append("hand__t1__t2__t3")

        for _sim_id, group in df.groupby("simulation_id"):
            sorted_group = group.sort_values("mulligan_count")
            keep_idx: Optional[int] = None

            for idx, row in sorted_group.iterrows():
                if any(row[col] for col in turn_cols):
                    keep_idx = int(idx)
                    break

            if keep_idx is None:
                keep_idx = int(sorted_group.index[-1])

            kept[keep_idx] = True
            kept_at[keep_idx] = _kept_hand_size(int(df.loc[keep_idx, "mulligan_count"]))

        df["MulliganResult"] = kept
        df["MulliganResult~kept_at"] = kept_at
        return df
