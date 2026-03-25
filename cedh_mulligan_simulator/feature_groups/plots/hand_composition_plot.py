"""HandCompositionPlot feature group — 2×2 histogram grid of card type distributions in kept hands."""

from pathlib import Path
from typing import Any, Optional, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options

from cedh_mulligan_simulator.card_registry import DEFAULT_CARD_REGISTRY, CardRegistry


class HandCompositionPlot(FeatureGroup):
    """Derived FG: saves 2×2 histogram grid of card type counts in kept hands.

    Usage:
        Feature("HandCompositionPlot", options={"group": {"experiment_id": "my_exp"}, "context": {"plot_dir": "plots"}})

    Output column contains PNG path for kept hands, None for non-kept.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature("hand", options=options),
            Feature("MulliganResult", options=options),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pl.DataFrame = data

        # Extract options
        experiment_id = (
            features.get_options_key("experiment_id") or features.get_options_key("scenario_id") or "default"
        )
        plot_dir = features.get_options_key("plot_dir") or "plots"
        card_registry = features.get_options_key("card_registry") or DEFAULT_CARD_REGISTRY

        keep_mask = df["MulliganResult"].cast(pl.Boolean)
        kept_df = df.filter(keep_mask)

        if len(kept_df) == 0:
            return df.with_columns(pl.lit(None).cast(pl.Utf8).alias("HandCompositionPlot"))

        # Compute type counts for kept hands
        card_types = ["land", "artifact", "creature", "ritual"]
        counts = {
            card_type: kept_df["hand"]
            .map_elements(
                lambda hand, ct=card_type: _count_type(hand, card_registry, ct),  # type: ignore[misc]
                return_dtype=pl.Int64,
            )
            .to_numpy()
            for card_type in card_types
        }

        # Create 2×2 histogram grid
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"Hand Composition Distribution — {experiment_id}", fontsize=14)

        for idx, card_type in enumerate(card_types):
            ax = axes[idx // 2, idx % 2]
            ax.hist(counts[card_type], bins=range(0, 8), edgecolor="black", alpha=0.7)
            ax.set_xlabel(f"{card_type.capitalize()} Count")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{card_type.capitalize()} Distribution")
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Save to PNG
        output_path = Path(plot_dir) / experiment_id / "hand_composition.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return df.with_columns(
            pl.when(keep_mask).then(pl.lit(str(output_path))).otherwise(None).alias("HandCompositionPlot")
        )


def _count_type(hand: list[str], registry: CardRegistry, card_type: str) -> int:
    """Count how many cards in hand match the given type (excluding filler)."""
    count = 0
    for card_name in hand:
        if card_name != "filler":
            card = registry.get(card_name)
            if card is not None and card.type == card_type:
                count += 1
    return count
