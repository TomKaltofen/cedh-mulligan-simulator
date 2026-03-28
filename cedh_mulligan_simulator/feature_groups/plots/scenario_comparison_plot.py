"""ScenarioComparisonPlot: visual dashboard showing T1/T2 rates across scenarios."""

from pathlib import Path
from typing import Any, Optional, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from mloda.provider import FeatureGroup, FeatureSet  # noqa: E402
from mloda.user import Feature, FeatureName, Options  # noqa: E402


class ScenarioComparisonPlot(FeatureGroup):
    """Saves a grouped bar chart comparing T1/T2 castability across scenarios.

    Usage: ``Feature("ScenarioComparisonPlot", options=opts)``

    Depends on:
        - ``scenario_id``: Scenario name (e.g., "baseline", "no_jeweled_lotus")
        - ``hand__t1__proportion``: T1 castability rate
        - ``hand__t1__t2__proportion``: T2 castability rate
        - ``MulliganResult``: kept/mulligan status

    Creates a grouped bar chart with:
        - X-axis: Scenario names
        - Y-axis: Success rate (0-100%)
        - Two bars per scenario: T1 (blue), T2 (orange)
        - Horizontal baseline reference line (if baseline scenario exists)

    Options (group):
        ``experiment_id``: subdirectory name; falls back to ``scenario_id``, then ``"default"``

    Options (context):
        ``plot_dir``: base directory override (e.g. ``tmp_path`` for tests)

    Output column contains the PNG path for kept rows, ``None`` for non-kept rows.
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
        df: pl.DataFrame = data

        experiment_id: str = (
            features.get_options_key("experiment_id") or features.get_options_key("scenario_id") or "default"
        )

        plot_dir_override: Optional[str] = features.get_options_key("plot_dir")
        base_dir = Path(plot_dir_override) if plot_dir_override else Path("plots")
        out_dir = base_dir / experiment_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "scenario_comparison.png"

        kept_mask = df["MulliganResult"].cast(pl.Boolean)
        kept = df.filter(kept_mask)

        # Group by scenario and aggregate proportions
        scenario_stats = (
            kept.group_by("scenario_id")
            .agg(
                [
                    pl.col("hand__t1__proportion").first(),
                    pl.col("hand__t1__t2__proportion").first(),
                ]
            )
            .sort("scenario_id")
        )

        # Extract data for plotting
        scenarios = scenario_stats["scenario_id"].to_list()
        t1_rates = (scenario_stats["hand__t1__proportion"] * 100).to_list()
        t2_rates = (scenario_stats["hand__t1__t2__proportion"] * 100).to_list()

        # Find baseline for reference line
        baseline_t1: Optional[float] = None
        if "baseline" in scenarios:
            baseline_idx = scenarios.index("baseline")
            baseline_t1 = t1_rates[baseline_idx]

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))

        x = range(len(scenarios))
        width = 0.35

        ax.bar([i - width / 2 for i in x], t1_rates, width, label="T1 Castability", color="#2E86DE")
        ax.bar([i + width / 2 for i in x], t2_rates, width, label="T2 Castability", color="#EE5A6F")

        # Add baseline reference line if exists
        if baseline_t1 is not None:
            ax.axhline(y=baseline_t1, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline T1")

        ax.set_xlabel("Scenario", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(f"Scenario Comparison: {experiment_id}", fontsize=14, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

        return df.with_columns(
            pl.when(kept_mask).then(pl.lit(str(out_path))).otherwise(None).alias("ScenarioComparisonPlot")
        )
