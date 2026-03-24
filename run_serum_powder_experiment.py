"""Serum Powder Experiment — compare Braids mulligan with and without Serum Powder."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl

from mloda.user import Domain, Feature, Options, PluginCollector, mlodaAPI
from mloda.provider import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from card_database.colorless import SERUM_POWDER
from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.card_registry import CardRegistry, Mana, build_registry, land
from cedh_mulligan_simulator.extenders import TimingExtender
from cedh_mulligan_simulator.feature_groups.mulligan import HandGenerator, MulliganResult, Turn1, Turn2
from cedh_mulligan_simulator.feature_groups.statistics import AverageKeptHandSize, MeanMulliganDepth

N_SIMULATIONS = 50000

BOX_WIDTH = 60
BOX_HORIZ = "="
BOX_THIN = "-"

# Serum Powder replaces Dismember (generic removal, weakest synergy with sacrifice themes).
REPLACED_CARD = "dismember"


def _remove_card(registry: CardRegistry, card_name: str) -> CardRegistry:
    return {k: v for k, v in registry.items() if k != card_name}


_BRAIDS_WITHOUT_DISMEMBER = _remove_card(BRAIDS_REGISTRY, REPLACED_CARD)
_BRAIDS_WITH_SERUM_POWDER = build_registry(*_BRAIDS_WITHOUT_DISMEMBER.values(), SERUM_POWDER)
_EXTRA_SWAMP = land("swamp_3", t1_mana=Mana(1, black=1), t2_mana=Mana(1, black=1))
_BRAIDS_WITH_SWAMP = build_registry(*_BRAIDS_WITHOUT_DISMEMBER.values(), _EXTRA_SWAMP)

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "baseline",
        "name": "Baseline (with Dismember)",
        "registry": BRAIDS_REGISTRY,
        "cost": BRAIDS_COST,
    },
    {
        "id": "serum_powder",
        "name": "Serum Powder (replaces Dismember)",
        "registry": _BRAIDS_WITH_SERUM_POWDER,
        "cost": BRAIDS_COST,
    },
    {
        "id": "swamp",
        "name": "Swamp (replaces Dismember)",
        "registry": _BRAIDS_WITH_SWAMP,
        "cost": BRAIDS_COST,
    },
]


def make_scenario_providers(scenario_id: str) -> set[type[FeatureGroup]]:
    def make_domain_class(base: type[FeatureGroup], feature_names: set[str] | None = None) -> type[FeatureGroup]:
        def get_domain(cls: type[FeatureGroup]) -> Domain:
            return Domain(scenario_id)

        attrs: dict[str, Any] = {"get_domain": classmethod(get_domain)}
        if feature_names:

            def feature_names_supported(cls: type[FeatureGroup]) -> set[str]:
                return feature_names

            attrs["feature_names_supported"] = classmethod(feature_names_supported)

        return type(f"{base.__name__}_{scenario_id}", (base,), attrs)

    return {
        make_domain_class(HandGenerator, {"hand", "simulation_id", "mulligan_count", "scenario_id"}),
        make_domain_class(Turn1),
        make_domain_class(Turn2),
        make_domain_class(MulliganResult, {"MulliganResult"}),
        make_domain_class(MeanMulliganDepth, {"MeanMulliganDepth"}),
        make_domain_class(AverageKeptHandSize, {"AverageKeptHandSize"}),
    }


def build_all_features(scenarios: List[Dict[str, Any]], experiment_id: str) -> list[Feature | str]:
    all_features: list[Feature | str] = []
    for scenario in scenarios:
        opts = Options(
            group={"scenario_id": scenario["id"], "n_simulations": N_SIMULATIONS, "experiment_id": experiment_id},
            context={"card_registry": scenario["registry"], "commander_cost": scenario["cost"]},
        )
        domain = scenario["id"]
        all_features.extend(
            [
                Feature("MulliganResult", options=opts, domain=domain),
                Feature("hand", options=opts, domain=domain),
                Feature("simulation_id", options=opts, domain=domain),
                Feature("mulligan_count", options=opts, domain=domain),
                Feature("scenario_id", options=opts, domain=domain),
                Feature("hand__t1", options=opts, domain=domain),
                Feature("hand__t1__t2", options=opts, domain=domain),
                Feature("MeanMulliganDepth", options=opts, domain=domain),
                Feature("AverageKeptHandSize", options=opts, domain=domain),
            ]
        )
    return all_features


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator
    return (max(centre - margin, 0.0), p, min(centre + margin, 1.0))


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_ci(low: float, high: float) -> str:
    return f"[{low * 100:.1f}%, {high * 100:.1f}%]"


def format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f}pp"


ScenarioMetrics = Dict[str, Union[float, str]]


def print_scenario_results(scenario: Dict[str, Any], df: pl.DataFrame) -> ScenarioMetrics:
    print(f"{BOX_THIN * 3} {scenario['name']} {BOX_THIN * 3}")
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
    print(f"  {'Mean Mulligan Depth':<28} {mean_depth:>8.2f}")
    print(f"  {'Avg Kept Hand Size':<28} {avg_hand_size:>7.1f} cards")
    print()

    return {"t1_prop": t1_prop, "t2_prop": t2_prop, "name": str(scenario["name"]), "id": str(scenario["id"])}


def print_delta(metrics: List[ScenarioMetrics]) -> None:
    if len(metrics) < 2:
        return

    baseline: Optional[ScenarioMetrics] = next((m for m in metrics if m["id"] == "baseline"), None)
    if baseline is None:
        return

    baseline_t1 = float(baseline["t1_prop"])
    baseline_t2 = float(baseline["t2_prop"])

    print("--- Delta vs Baseline ---")
    print()
    print(f"  {'Scenario':<35} {'T1 Delta':>10} {'T2 Delta':>10}")
    print(f"  {BOX_THIN * 57}")
    for m in metrics:
        if m["id"] == "baseline":
            continue
        t1_d = float(m["t1_prop"]) - baseline_t1
        t2_d = float(m["t2_prop"]) - baseline_t2
        print(f"  {str(m['name']):<35} {format_delta(t1_d):>10} {format_delta(t2_d):>10}")
    print()


def main() -> None:
    experiment_id = "serum_powder"

    print(BOX_HORIZ * BOX_WIDTH)
    print("  Serum Powder Experiment")
    print(f"  {N_SIMULATIONS:,} simulations per scenario")
    print(BOX_HORIZ * BOX_WIDTH)
    print()

    all_providers: set[type[FeatureGroup]] = set()
    for scenario in SCENARIOS:
        all_providers |= make_scenario_providers(scenario["id"])

    all_features = build_all_features(SCENARIOS, experiment_id)

    results = mlodaAPI.run_all(
        features=all_features,
        compute_frameworks={PolarsDataFrame},
        function_extender={TimingExtender()},
        plugin_collector=PluginCollector.enabled_feature_groups(all_providers),
    )

    result_dfs = [r for r in results if isinstance(r, pl.DataFrame)]
    n_scenarios = len(SCENARIOS)

    from collections import defaultdict

    sig_groups: Dict[tuple[str, ...], List[tuple[int, pl.DataFrame]]] = defaultdict(list)
    for i, df in enumerate(result_dfs):
        sig = tuple(sorted(df.columns))
        sig_groups[sig].append((i, df))

    scenario_result_lists: Dict[str, List[pl.DataFrame]] = {s["id"]: [] for s in SCENARIOS}

    for sig, items in sig_groups.items():
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

    all_metrics: List[ScenarioMetrics] = []
    for scenario in SCENARIOS:
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
        df = pl.concat(parts, how="horizontal")
        metrics = print_scenario_results(scenario, df)
        all_metrics.append(metrics)

    print_delta(all_metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    main()
