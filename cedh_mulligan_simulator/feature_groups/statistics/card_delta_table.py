"""CardDeltaTable — derived feature that quantifies card impact via scenario comparison."""

from pathlib import Path
from typing import Any, Optional, Set

import pandas as pd

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options


class CardDeltaTable(FeatureGroup):
    """Quantifies marginal card impact by comparing baseline vs removal scenarios.

    Usage: ``Feature("CardDeltaTable", options=opts)``

    Depends on:
        - ``scenario_id`` — Scenario name (e.g., "baseline", "no_jeweled_lotus")
        - ``hand__t1__proportion`` — T1 castability rate
        - ``hand__t1__t2__proportion`` — T2 castability rate

    Creates a CSV table with columns:
        - ``card_removed`` — Card name (parsed from scenario_id)
        - ``baseline_t1_rate`` — Baseline T1 %
        - ``scenario_t1_rate`` — Scenario T1 %
        - ``t1_delta`` — Percentage point drop
        - ``t1_pct_change`` — Percent change
        - ``baseline_t2_rate``, ``scenario_t2_rate``, ``t2_delta``, ``t2_pct_change`` — Same for T2

    Sorted by ``t1_delta`` descending (biggest impact first).

    Options (group):
        ``experiment_id`` — subdirectory name; falls back to ``scenario_id``, then ``"default"``

    Options (context):
        ``plot_dir`` — base directory override (e.g. ``tmp_path`` for tests)

    Output column contains the CSV path for kept rows, ``NaN`` for non-kept rows.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature("scenario_id", options=options),
            Feature("hand__t1__proportion", options=options),
            Feature("hand__t1__t2__proportion", options=options),
            Feature("MulliganResult", options=options),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        df: pd.DataFrame = data

        experiment_id: str = (
            features.get_options_key("experiment_id") or features.get_options_key("scenario_id") or "default"
        )

        plot_dir_override: Optional[str] = features.get_options_key("plot_dir")
        base_dir = Path(plot_dir_override) if plot_dir_override else Path("plots")
        out_dir = base_dir / experiment_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "card_delta_table.csv"

        kept_mask = df["MulliganResult"].astype(bool)
        kept = df.loc[kept_mask].copy()

        # Group by scenario and aggregate proportions
        scenario_stats = (
            kept.groupby("scenario_id")
            .agg({"hand__t1__proportion": "first", "hand__t1__t2__proportion": "first"})
            .reset_index()
        )

        # Identify baseline scenario
        baseline_row = scenario_stats[scenario_stats["scenario_id"] == "baseline"]
        if baseline_row.empty:
            # Fallback: first alphabetically
            scenario_stats = scenario_stats.sort_values("scenario_id")
            baseline_row = scenario_stats.iloc[[0]]

        baseline_t1 = float(baseline_row["hand__t1__proportion"].iloc[0])
        baseline_t2 = float(baseline_row["hand__t1__t2__proportion"].iloc[0])

        # Parse card names and compute deltas
        rows = []
        for _, row in scenario_stats.iterrows():
            scenario_id = str(row["scenario_id"])
            if scenario_id == baseline_row["scenario_id"].iloc[0]:
                continue  # Skip baseline itself

            # Parse card name from scenario_id
            card_name = cls._parse_card_name(scenario_id)

            scenario_t1 = float(row["hand__t1__proportion"])
            scenario_t2 = float(row["hand__t1__t2__proportion"])

            t1_delta = baseline_t1 - scenario_t1
            t1_pct_change = (t1_delta / baseline_t1 * 100.0) if baseline_t1 > 0 else 0.0

            t2_delta = baseline_t2 - scenario_t2
            t2_pct_change = (t2_delta / baseline_t2 * 100.0) if baseline_t2 > 0 else 0.0

            rows.append(
                {
                    "card_removed": card_name,
                    "baseline_t1_rate": baseline_t1,
                    "scenario_t1_rate": scenario_t1,
                    "t1_delta": t1_delta,
                    "t1_pct_change": t1_pct_change,
                    "baseline_t2_rate": baseline_t2,
                    "scenario_t2_rate": scenario_t2,
                    "t2_delta": t2_delta,
                    "t2_pct_change": t2_pct_change,
                }
            )

        delta_df = pd.DataFrame(rows).sort_values("t1_delta", ascending=False)
        delta_df.to_csv(out_path, index=False)

        df["CardDeltaTable"] = pd.array([None] * len(df), dtype="object")
        df.loc[kept_mask, "CardDeltaTable"] = str(out_path)
        return df

    @staticmethod
    def _parse_card_name(scenario_id: str) -> str:
        """Parse card name from scenario_id (e.g., 'no_jeweled_lotus' -> 'Jeweled Lotus')."""
        if not scenario_id.startswith("no_"):
            return scenario_id

        name_part = scenario_id[3:]  # Remove 'no_' prefix

        # Special cases - replace before processing words
        special_cases = {
            "0mana": "0-Mana",
            "1cmc": "1-CMC",
        }

        # Check for special cases first
        for pattern, replacement in special_cases.items():
            if pattern in name_part:
                name_part = name_part.replace(pattern, replacement)
                # Convert remaining underscores to spaces and capitalize remaining words
                words = name_part.split("_")
                # Special case words keep their capitalization
                result_words = []
                for word in words:
                    if any(special in word for special in special_cases.values()):
                        result_words.append(word)
                    else:
                        result_words.append(word.capitalize())
                return " ".join(result_words)

        # Convert underscores to spaces and capitalize words (no special cases)
        words = name_part.split("_")
        return " ".join(word.capitalize() for word in words)
