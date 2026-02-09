"""CardCooccurrence feature group — finds which card pairs appear together most in kept hands."""

from collections import Counter
from pathlib import Path
from typing import Any, Optional, Set

import pandas as pd

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options

from cedh_mulligan_simulator.card_registry import DEFAULT_CARD_REGISTRY


class CardCooccurrence(FeatureGroup):
    """Derived FG: saves top-50 card pair co-occurrences in kept hands to CSV.

    Usage:
        Feature("CardCooccurrence", options={"group": {"experiment_id": "my_exp"}, "context": {"plot_dir": "plots"}})

    Output column contains CSV path for kept hands, None for non-kept.
    CSV columns: card_a, card_b, count, frequency
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature("hand", options=options),
            Feature("MulliganResult", options=options),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pd.DataFrame = data

        # Extract options
        experiment_id = (
            features.get_options_key("experiment_id") or features.get_options_key("scenario_id") or "default"
        )
        plot_dir = features.get_options_key("plot_dir") or "plots"
        card_registry = features.get_options_key("card_registry") or DEFAULT_CARD_REGISTRY

        # Compute co-occurrences for kept hands (handle both boolean True and string "keep")
        keep_mask = (df["MulliganResult"] == True) | (df["MulliganResult"] == "keep")  # noqa: E712
        kept_df = df[keep_mask].copy()

        if len(kept_df) == 0:
            # No kept hands — return empty column
            df["CardCooccurrence"] = pd.array([None] * len(df), dtype="object")
            return df

        # Count all pairs in kept hands
        pair_counter: Counter[tuple[str, str]] = Counter()

        for hand in kept_df["hand"]:
            # Exclude filler cards
            non_filler = [card for card in hand if card != "filler" and card in card_registry]
            # Generate canonical pairs
            unique_cards = sorted(set(non_filler))
            for i, card_a in enumerate(unique_cards):
                for card_b in unique_cards[i + 1 :]:
                    pair_counter[(card_a, card_b)] += 1

        # Get top 50 pairs
        top_pairs = pair_counter.most_common(50)

        # Save to CSV
        output_path = Path(plot_dir) / experiment_id / "card_cooccurrence.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_kept = len(kept_df)
        rows = [
            {
                "card_a": card_a,
                "card_b": card_b,
                "count": count,
                "frequency": count / total_kept,
            }
            for (card_a, card_b), count in top_pairs
        ]

        pd.DataFrame(rows).to_csv(output_path, index=False)

        # Create output column: path for kept hands, None otherwise
        df["CardCooccurrence"] = pd.array([None] * len(df), dtype="object")
        df.loc[keep_mask, "CardCooccurrence"] = str(output_path)

        return df
