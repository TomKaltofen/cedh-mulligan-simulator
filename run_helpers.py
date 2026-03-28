"""Shared helpers for simulation run scripts."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import polars as pl

from mloda.user import Domain, Feature, Options, PluginCollector, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from cedh_mulligan_simulator.card_registry import CardRegistry
from cedh_mulligan_simulator.extenders import TimingExtender
from cedh_mulligan_simulator.feature_groups.statistics.confidence_interval import wilson_interval

# Box-drawing constants
BOX_WIDTH = 60
BOX_HORIZ = "="
BOX_THIN = "-"

# Type aliases
ScenarioMetrics = Dict[str, Union[float, str]]
ProviderSpec = Tuple[type, Optional[Set[str]]]


def remove_card(registry: CardRegistry, card_name: str) -> CardRegistry:
    """Create a new registry with the specified card removed."""
    return {k: v for k, v in registry.items() if k != card_name}


def remove_cards(registry: CardRegistry, *card_names: str) -> CardRegistry:
    """Create a new registry with all specified cards removed."""
    names = set(card_names)
    return {k: v for k, v in registry.items() if k not in names}


def format_pct(value: float) -> str:
    """Format as percentage."""
    return f"{value * 100:.1f}%"


def format_ci(low: float, high: float) -> str:
    """Format confidence interval."""
    return f"[{low * 100:.1f}%, {high * 100:.1f}%]"


def format_delta(value: float, precision: int = 1) -> str:
    """Format delta (signed percentage points)."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.{precision}f}pp"


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


def make_scenario_providers(
    scenario_id: str,
    feature_group_specs: List[ProviderSpec],
) -> Set[type]:
    """Create domain-specific FeatureGroup wrappers with unique class names.

    Uses type() to dynamically create classes with names that include the scenario_id
    (e.g., "HandGenerator_baseline"). This ensures mloda treats each scenario's
    feature groups as distinct, since mloda groups results by class name.
    """

    def make_domain_class(base: type, feature_names: Optional[Set[str]] = None) -> type:
        """Create a domain-specific subclass with unique name."""

        def get_domain(cls: type) -> Domain:
            return Domain(scenario_id)

        attrs: Dict[str, Any] = {"get_domain": classmethod(get_domain)}
        if feature_names:

            def feature_names_supported(cls: type) -> Set[str]:
                return feature_names

            attrs["feature_names_supported"] = classmethod(feature_names_supported)

        return type(f"{base.__name__}_{scenario_id}", (base,), attrs)

    return {make_domain_class(base, names) for base, names in feature_group_specs}


def build_all_features(
    scenarios: List[Dict[str, Any]],
    experiment_id: str,
    n_simulations: int,
    feature_names: List[str],
) -> List[Any]:
    """Build feature list for all scenarios with domain isolation."""
    all_features: List[Any] = []
    for scenario in scenarios:
        opts = Options(
            group={"scenario_id": scenario["id"], "n_simulations": n_simulations, "experiment_id": experiment_id},
            context={"card_registry": scenario["registry"], "commander_cost": scenario["cost"]},
        )
        domain = scenario["id"]
        all_features.extend(Feature(name, options=opts, domain=domain) for name in feature_names)
    return all_features


def collect_scenario_results(
    results: List[Any],
    scenarios: List[Dict[str, Any]],
) -> Dict[str, pl.DataFrame]:
    """Group mloda results by scenario and merge into one DataFrame per scenario."""
    result_dfs = [r for r in results if isinstance(r, pl.DataFrame)]
    n_scenarios = len(scenarios)

    sig_groups: Dict[Tuple[str, ...], List[Tuple[int, pl.DataFrame]]] = defaultdict(list)
    for i, df in enumerate(result_dfs):
        sig = tuple(sorted(df.columns))
        sig_groups[sig].append((i, df))

    scenario_result_lists: Dict[str, List[pl.DataFrame]] = {s["id"]: [] for s in scenarios}

    for _sig, items in sig_groups.items():
        if len(items) == n_scenarios:
            items_with_sid = [(i, df) for i, df in items if "scenario_id" in df.columns]
            if items_with_sid:
                for _, df in items:
                    if "scenario_id" in df.columns:
                        sid = str(df["scenario_id"][0])
                        scenario_result_lists[sid].append(df)
            else:
                sorted_items = sorted(items, key=lambda x: x[0])
                scenario_ids = list(scenario_result_lists.keys())
                for j, (_, df) in enumerate(sorted_items):
                    sid = scenario_ids[j % n_scenarios]
                    scenario_result_lists[sid].append(df)

    merged: Dict[str, pl.DataFrame] = {}
    for scenario in scenarios:
        dfs = scenario_result_lists[scenario["id"]]
        if not dfs:
            print(f"Warning: No results for scenario {scenario['id']}")
            continue
        seen: set[str] = set()
        parts: list[pl.DataFrame] = []
        for sub_df in dfs:
            new_cols = [c for c in sub_df.columns if c not in seen]
            if new_cols:
                parts.append(sub_df.select(new_cols))
                seen.update(new_cols)
        merged[scenario["id"]] = pl.concat(parts, how="horizontal")

    return merged


def print_scenario_results(scenario: Dict[str, Any], df: pl.DataFrame, show_plots: bool = False) -> ScenarioMetrics:
    """Print results for a single scenario and return key metrics."""
    print(f"{BOX_THIN * 3} Scenario: {scenario['name']} {BOX_THIN * 3}")
    print()

    kept = df.filter(pl.col("MulliganResult").cast(pl.Boolean))
    n = len(kept)

    t1_successes = int(kept["hand__t1"].sum() or 0)
    t2_successes = int(kept["hand__t1__t2"].sum() or 0)

    t1_ci_low, t1_prop, t1_ci_high = wilson_interval(t1_successes, n)
    t2_ci_low, t2_prop, t2_ci_high = wilson_interval(t2_successes, n)

    mean_depth = float(kept["MeanMulliganDepth"][0] or 0.0)
    avg_hand_size = float(kept["AverageKeptHandSize"][0] or 0.0)

    t2_only = kept["hand__t1__t2"] & ~kept["hand__t1"]
    forced = ~kept["hand__t1__t2"] & ~kept["hand__t1"]
    t2_only_pct = int(t2_only.sum() or 0) / n
    forced_pct = int(forced.sum() or 0) / n

    print(f"  {'Metric':<22} {'Rate':>10} {'95% CI':>22}")
    print(f"  {BOX_THIN * 56}")
    print(f"  {'T1 Castable':<22} {format_pct(t1_prop):>10} {format_ci(t1_ci_low, t1_ci_high):>22}")
    print(f"  {'T2 Castable':<22} {format_pct(t2_prop):>10} {format_ci(t2_ci_low, t2_ci_high):>22}")
    print(f"  {'T2 Only':<22} {format_pct(t2_only_pct):>10}")
    print(f"  {'Forced Keep':<22} {format_pct(forced_pct):>10}")
    print()

    print(f"  {'Metric':<28} {'Value':>12}")
    print(f"  {BOX_THIN * 42}")
    print(f"  {'Mean Mulligan Depth':<28} {mean_depth:>12.2f}")
    print(f"  {'Average Kept Hand Size':<28} {avg_hand_size:>11.1f} cards")
    print()

    if show_plots:
        convergence_path = kept["ConvergencePlot"][0] if "ConvergencePlot" in kept.columns else None
        composition_path = kept["HandCompositionPlot"][0] if "HandCompositionPlot" in kept.columns else None
        cooccurrence_path = kept["CardCooccurrence"][0] if "CardCooccurrence" in kept.columns else None
        print(f"  {'Convergence Plot:':<20} {convergence_path or 'Not generated'}")
        print(f"  {'Hand Composition:':<20} {composition_path or 'Not generated'}")
        print(f"  {'Card Co-occurrence:':<20} {cooccurrence_path or 'Not generated'}")
        print()

    return {"t1_prop": t1_prop, "t2_prop": t2_prop, "name": str(scenario["name"]), "id": str(scenario["id"])}


def print_delta_summary(metrics: List[ScenarioMetrics], precision: int = 1) -> None:
    """Print delta vs baseline for all non-baseline scenarios."""
    if len(metrics) < 2:
        return

    baseline: Optional[ScenarioMetrics] = next((m for m in metrics if m["id"] == "baseline"), None)
    if baseline is None:
        print("  No baseline scenario found.")
        print()
        return

    print_section("Delta vs Baseline")
    print(f"  {'Scenario':<35} {'T1 Delta':>12} {'T2 Delta':>12}")
    print(f"  {BOX_THIN * 60}")

    for m in metrics:
        if m["id"] == "baseline":
            continue
        t1_d = float(m["t1_prop"]) - float(baseline["t1_prop"])
        t2_d = float(m["t2_prop"]) - float(baseline["t2_prop"])
        print(f"  {str(m['name']):<35} {format_delta(t1_d, precision):>12} {format_delta(t2_d, precision):>12}")
    print()


def run_experiment(
    scenarios: List[Dict[str, Any]],
    provider_specs: List[ProviderSpec],
    feature_names: List[str],
    n_simulations: int,
    experiment_id: str,
    title: str,
    show_plots: bool = False,
    delta_precision: int = 1,
) -> List[ScenarioMetrics]:
    """Run a full simulation experiment: build providers, run mloda, print results."""
    print_header(title, n_simulations, len(scenarios))

    all_providers: Set[type] = set()
    for scenario in scenarios:
        all_providers |= make_scenario_providers(scenario["id"], provider_specs)

    all_features = build_all_features(scenarios, experiment_id, n_simulations, feature_names)

    results: List[Any] = mlodaAPI.run_all(
        features=all_features,
        compute_frameworks={PolarsDataFrame},
        function_extender={TimingExtender()},
        plugin_collector=PluginCollector.enabled_feature_groups(all_providers),
    )

    scenario_dfs = collect_scenario_results(results, scenarios)

    all_metrics: List[ScenarioMetrics] = []
    for scenario in scenarios:
        if scenario["id"] not in scenario_dfs:
            continue
        df = scenario_dfs[scenario["id"]]
        metrics = print_scenario_results(scenario, df, show_plots=show_plots)
        all_metrics.append(metrics)

    print_delta_summary(all_metrics, precision=delta_precision)

    return all_metrics
