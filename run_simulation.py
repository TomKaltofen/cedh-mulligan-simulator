"""Run the cEDH Mulligan Simulator — multi-scenario comparison with statistics and plots."""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from mloda.user import Domain, Feature, Options, PluginCollector, mlodaAPI
from mloda.provider import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.card_registry import CardRegistry
from cedh_mulligan_simulator.extenders import TimingExtender
from cedh_mulligan_simulator.feature_groups.mulligan import (
    HandGenerator,
    MulliganResult,
    Turn1,
    Turn2,
)
from cedh_mulligan_simulator.feature_groups.statistics import (
    AverageKeptHandSize,
    CardCooccurrence,
    MeanMulliganDepth,
    Proportion,
)
from cedh_mulligan_simulator.feature_groups.plots import (
    ConvergencePlot,
    HandCompositionPlot,
)

N_SIMULATIONS = 1000

# Box-drawing constants
BOX_WIDTH = 60
BOX_HORIZ = "="
BOX_THIN = "-"


def remove_card(registry: CardRegistry, card_name: str) -> CardRegistry:
    """Create a new registry with the specified card removed."""
    return {k: v for k, v in registry.items() if k != card_name}


def make_scenario_providers(scenario_id: str) -> set[type[FeatureGroup]]:
    """Create domain-specific FeatureGroup wrappers with unique class names.

    Uses type() to dynamically create classes with names that include the scenario_id
    (e.g., "HandGenerator_baseline"). This ensures mloda treats each scenario's
    feature groups as distinct, since mloda groups results by class name.
    """

    def make_domain_class(base: type[FeatureGroup], feature_names: set[str] | None = None) -> type[FeatureGroup]:
        """Create a domain-specific subclass with unique name."""

        def get_domain(cls: type[FeatureGroup]) -> Domain:
            return Domain(scenario_id)

        attrs: dict[str, Any] = {"get_domain": classmethod(get_domain)}
        if feature_names:

            def feature_names_supported(cls: type[FeatureGroup]) -> set[str]:
                return feature_names

            attrs["feature_names_supported"] = classmethod(feature_names_supported)

        # Create class with unique name: "HandGenerator_baseline"
        return type(f"{base.__name__}_{scenario_id}", (base,), attrs)

    return {
        make_domain_class(HandGenerator, {"hand", "simulation_id", "mulligan_count", "scenario_id"}),
        make_domain_class(Turn1),
        make_domain_class(Turn2),
        make_domain_class(MulliganResult, {"MulliganResult"}),
        make_domain_class(Proportion),
        make_domain_class(MeanMulliganDepth, {"MeanMulliganDepth"}),
        make_domain_class(AverageKeptHandSize, {"AverageKeptHandSize"}),
        make_domain_class(CardCooccurrence, {"CardCooccurrence"}),
        make_domain_class(ConvergencePlot, {"ConvergencePlot"}),
        make_domain_class(HandCompositionPlot, {"HandCompositionPlot"}),
    }


# Define scenarios for comparison
SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "baseline",
        "name": "Baseline",
        "registry": BRAIDS_REGISTRY,
        "cost": BRAIDS_COST,
    },
    {
        "id": "no_lions_eye_diamond",
        "name": "No LED",
        "registry": remove_card(BRAIDS_REGISTRY, "lions_eye_diamond"),
        "cost": BRAIDS_COST,
    },
]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Return (lower, point, upper) for a Wilson score interval."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator
    return (max(centre - margin, 0.0), p, min(centre + margin, 1.0))


def print_header(title: str, n_simulations: int, n_scenarios: int) -> None:
    """Print main header with box drawing."""
    print(BOX_HORIZ * BOX_WIDTH)
    print(f"  {title}")
    print(f"  {n_simulations:,} simulations per scenario x {n_scenarios} scenarios")
    print(BOX_HORIZ * BOX_WIDTH)
    print()


def print_section(title: str) -> None:
    """Print section header."""
    print(f"{BOX_THIN * 3} {title} {BOX_THIN * 3}")
    print()


def format_pct(value: float) -> str:
    """Format as percentage."""
    return f"{value * 100:.1f}%"


def format_ci(low: float, high: float) -> str:
    """Format confidence interval."""
    return f"[{low * 100:.1f}%, {high * 100:.1f}%]"


def format_delta(value: float) -> str:
    """Format delta (signed percentage points)."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}pp"


# Type alias for scenario metrics
ScenarioMetrics = Dict[str, Union[float, str]]


def build_all_features(scenarios: List[Dict[str, Any]], experiment_id: str) -> list[Feature | str]:
    """Build feature list for all scenarios with domain isolation."""
    all_features: list[Feature | str] = []

    for scenario in scenarios:
        opts = Options(
            group={"scenario_id": scenario["id"], "n_simulations": N_SIMULATIONS, "experiment_id": experiment_id},
            context={"card_registry": scenario["registry"], "commander_cost": scenario["cost"]},
        )
        domain = scenario["id"]

        all_features.extend(
            [
                # Core
                Feature("MulliganResult", options=opts, domain=domain),
                Feature("hand", options=opts, domain=domain),
                Feature("simulation_id", options=opts, domain=domain),
                Feature("mulligan_count", options=opts, domain=domain),
                Feature("scenario_id", options=opts, domain=domain),
                # Turn evaluations
                Feature("hand__t1", options=opts, domain=domain),
                Feature("hand__t1__t2", options=opts, domain=domain),
                # Proportions (for multi-scenario comparison)
                Feature("hand__t1__proportion", options=opts, domain=domain),
                Feature("hand__t1__t2__proportion", options=opts, domain=domain),
                # Aggregate stats
                Feature("MeanMulliganDepth", options=opts, domain=domain),
                Feature("AverageKeptHandSize", options=opts, domain=domain),
                # Card co-occurrence
                Feature("CardCooccurrence", options=opts, domain=domain),
                # Plots (per-scenario)
                Feature("ConvergencePlot", options=opts, domain=domain),
                Feature("HandCompositionPlot", options=opts, domain=domain),
            ]
        )

    return all_features


def print_scenario_results(scenario: Dict[str, Any], df: pd.DataFrame) -> ScenarioMetrics:
    """Print results for a single scenario and return key metrics."""
    print(f"{BOX_THIN * 3} Scenario: {scenario['name']} {BOX_THIN * 3}")
    print()

    kept = df[df["MulliganResult"]]
    n = len(kept)

    # Compute statistics directly from data
    t1_successes = int(kept["hand__t1"].sum())
    t2_successes = int(kept["hand__t1__t2"].sum())

    t1_ci_low, t1_prop, t1_ci_high = wilson_interval(t1_successes, n)
    t2_ci_low, t2_prop, t2_ci_high = wilson_interval(t2_successes, n)

    # Extract scalar statistics from first kept row
    first_kept = kept.iloc[0]
    mean_depth = float(first_kept["MeanMulliganDepth"])
    avg_hand_size = float(first_kept["AverageKeptHandSize"])

    # Exclusive keep categories (each hand counted once)
    t2_only = kept["hand__t1__t2"] & ~kept["hand__t1"]
    forced = ~kept["hand__t1__t2"] & ~kept["hand__t1"]

    t2_only_pct = t2_only.sum() / n
    forced_pct = forced.sum() / n

    print(f"  {'Metric':<20} {'Rate':>10} {'95% CI':>20}")
    print(f"  {BOX_THIN * 50}")
    print(f"  {'T1 Castable':<20} {format_pct(t1_prop):>10} {format_ci(t1_ci_low, t1_ci_high):>20}")
    print(f"  {'T2 Castable':<20} {format_pct(t2_prop):>10} {format_ci(t2_ci_low, t2_ci_high):>20}")
    print(f"  {'T2 Only':<20} {format_pct(t2_only_pct):>10} {'':<20}")
    print(f"  {'Forced Keep':<20} {format_pct(forced_pct):>10} {'':<20}")
    print()

    print(f"  {'Metric':<28} {'Value':>12}")
    print(f"  {BOX_THIN * 42}")
    print(f"  {'Mean Mulligan Depth':<28} {mean_depth:>12.2f}")
    print(f"  {'Average Kept Hand Size':<28} {avg_hand_size:>11.1f} cards")
    print()

    # Get plot paths
    convergence_path = first_kept.get("ConvergencePlot", None)
    composition_path = first_kept.get("HandCompositionPlot", None)
    cooccurrence_path = first_kept.get("CardCooccurrence", None)

    print(f"  {'Convergence Plot:':<20} {convergence_path or 'Not generated'}")
    print(f"  {'Hand Composition:':<20} {composition_path or 'Not generated'}")
    print(f"  {'Card Co-occurrence:':<20} {cooccurrence_path or 'Not generated'}")
    print()

    return {"t1_prop": t1_prop, "t2_prop": t2_prop, "name": str(scenario["name"]), "id": str(scenario["id"])}


def create_scenario_comparison_plot(metrics: List[ScenarioMetrics], experiment_id: str) -> Optional[str]:
    """Create a grouped bar chart comparing T1/T2 castability across scenarios."""
    if len(metrics) < 1:
        return None

    out_dir = Path("plots") / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scenario_comparison.png"

    scenarios = [str(m["name"]) for m in metrics]
    t1_rates = [float(m["t1_prop"]) * 100 for m in metrics]
    t2_rates = [float(m["t2_prop"]) * 100 for m in metrics]

    # Find baseline for reference line
    baseline_t1: Optional[float] = None
    for m in metrics:
        if m["id"] == "baseline":
            baseline_t1 = float(m["t1_prop"]) * 100
            break

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
    ax.set_title(f"Scenario Comparison — {experiment_id}", fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)

    return str(out_path)


def create_card_delta_table(metrics: List[ScenarioMetrics], experiment_id: str) -> Optional[str]:
    """Create CSV showing impact of removing cards."""
    if len(metrics) < 2:
        return None

    # Find baseline
    baseline: Optional[ScenarioMetrics] = None
    for m in metrics:
        if m["id"] == "baseline":
            baseline = m
            break

    if baseline is None:
        # Use first as baseline
        baseline = metrics[0]

    baseline_t1 = float(baseline["t1_prop"])
    baseline_t2 = float(baseline["t2_prop"])

    rows = []
    for m in metrics:
        if m["id"] == baseline["id"]:
            continue

        scenario_id = str(m["id"])
        # Parse card name from scenario_id
        card_name = scenario_id
        if scenario_id.startswith("no_"):
            card_name = scenario_id[3:].replace("_", " ").title()

        scenario_t1 = float(m["t1_prop"])
        scenario_t2 = float(m["t2_prop"])

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

    if not rows:
        return None

    out_dir = Path("plots") / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "card_delta_table.csv"

    delta_df = pd.DataFrame(rows).sort_values("t1_delta", ascending=False)
    delta_df.to_csv(out_path, index=False)

    return str(out_path)


def print_delta_summary(metrics: List[ScenarioMetrics]) -> None:
    """Print card impact summary comparing scenarios to baseline."""
    print_section("Card Impact Summary")

    baseline: Optional[ScenarioMetrics] = None
    for m in metrics:
        if m["id"] == "baseline":
            baseline = m
            break

    if baseline is None:
        print("  No baseline scenario found.")
        print()
        return

    print(f"  {'Card Removed':<25} {'T1 Delta':>12} {'T2 Delta':>12}")
    print(f"  {BOX_THIN * 50}")

    for m in metrics:
        if m["id"] == "baseline":
            continue

        # Parse card name from scenario id
        scenario_id = str(m["id"])
        card_name = scenario_id.replace("no_", "").replace("_", " ").title()
        t1_delta = float(m["t1_prop"]) - float(baseline["t1_prop"])
        t2_delta = float(m["t2_prop"]) - float(baseline["t2_prop"])

        print(f"  {card_name:<25} {format_delta(t1_delta):>12} {format_delta(t2_delta):>12}")

    print()


def main() -> None:
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)

    experiment_id = "comparison"

    print_header("Braids Mulligan Simulator — Multi-Scenario Comparison", N_SIMULATIONS, len(SCENARIOS))

    # Build domain providers for all scenarios
    all_providers: set[type[FeatureGroup]] = set()
    for scenario in SCENARIOS:
        all_providers |= make_scenario_providers(scenario["id"])

    # Build features for all scenarios
    all_features = build_all_features(SCENARIOS, experiment_id)

    # Single mloda call with all providers
    results = mlodaAPI.run_all(
        features=all_features,
        compute_frameworks={PandasDataFrame},
        function_extender={TimingExtender()},
        plugin_collector=PluginCollector.enabled_feature_groups(all_providers),
    )

    # Results come back from mloda in an order determined by feature resolution.
    # DataFrames with scenario_id column tell us which scenario they belong to.
    # DataFrames without scenario_id are paired by column signature (same feature type).
    #
    # Strategy: Group by (column_signature) to find pairs/groups of same feature type,
    # then use scenario_id within each group to assign, or positional order otherwise.
    result_dfs = [r for r in results if isinstance(r, pd.DataFrame)]
    n_scenarios = len(SCENARIOS)

    from collections import defaultdict

    # Group DataFrames by their column signature
    sig_groups: Dict[tuple[str, ...], List[tuple[int, pd.DataFrame]]] = defaultdict(list)
    for i, df in enumerate(result_dfs):
        sig = tuple(sorted(df.columns))
        sig_groups[sig].append((i, df))

    # Build scenario results
    scenario_result_lists: Dict[str, List[pd.DataFrame]] = {s["id"]: [] for s in SCENARIOS}

    for sig, items in sig_groups.items():
        # Within this group, identify scenarios
        if len(items) == n_scenarios:
            # Check if any have scenario_id
            items_with_sid = [(i, df) for i, df in items if "scenario_id" in df.columns]
            if items_with_sid:
                # Use scenario_id to assign
                for _, df in items:
                    if "scenario_id" in df.columns:
                        sid = str(df["scenario_id"].iloc[0])
                        scenario_result_lists[sid].append(df)
            else:
                # No scenario_id in this group - use original result order
                # Map to scenarios based on position in original results
                sorted_items = sorted(items, key=lambda x: x[0])  # Sort by original index
                scenario_ids = list(scenario_result_lists.keys())
                for j, (_, df) in enumerate(sorted_items):
                    sid = scenario_ids[j % n_scenarios]
                    scenario_result_lists[sid].append(df)

    all_metrics: List[ScenarioMetrics] = []
    for scenario in SCENARIOS:
        dfs = scenario_result_lists[scenario["id"]]
        if not dfs:
            print(f"Warning: No results for scenario {scenario['id']}")
            continue
        # Merge all DataFrames for this scenario by index (axis=1), deduplicate columns
        merged = pd.concat(dfs, axis=1)
        df = merged.loc[:, ~merged.columns.duplicated()]
        metrics = print_scenario_results(scenario, df)
        all_metrics.append(metrics)

    # Print delta summary
    print_delta_summary(all_metrics)

    # Generate comparison outputs directly (not through mloda)
    scenario_plot_path = create_scenario_comparison_plot(all_metrics, experiment_id)
    delta_table_path = create_card_delta_table(all_metrics, experiment_id)

    # Print comparison outputs
    print_section("Generated Comparison Outputs")
    print(f"  {'Scenario Comparison Plot:':<28} {scenario_plot_path or 'Not generated'}")
    print(f"  {'Card Delta Table:':<28} {delta_table_path or 'Not generated'}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    main()
