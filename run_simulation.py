"""Run the cEDH Mulligan Simulator: multi-scenario comparison with statistics and plots."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from card_database.black import DEEPWOOD_LEGATE, OFFALSNOUT
from card_database.colorless import JEWELED_LOTUS
from card_database.lands import SWAMP_2
from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.card_registry import Card, Mana, build_registry
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
from run_helpers import (
    ProviderSpec,
    ScenarioMetrics,
    print_section,
    remove_card,
    remove_cards,
    run_experiment,
)

N_SIMULATIONS = 2000

# Replacement swamp entries used when cards are removed from the registry.
# We use unique names so they don't collide with existing registry keys.
_SWAMP_A = SWAMP_2

# 1cmc creature replacements for the 0-mana creature swap scenario.
_1CMC_CREATURES = [
    Card(name="signal_pest", type="creature", cost=Mana(1), cmc=1),
    Card(name="vault_skirge", type="creature", cost=Mana(1, black=1), cmc=1),
    Card(name="gingerbrute", type="creature", cost=Mana(1), cmc=1),
    Card(name="hope_of_ghirapur", type="creature", cost=Mana(1), cmc=1),
]

# Registry with Jeweled Lotus added (not in Braids main 99, used as A/B test target).
_BRAIDS_WITH_JLO = build_registry(*BRAIDS_REGISTRY.values(), JEWELED_LOTUS)

# Registry with Deepwood Legate added.
_BRAIDS_WITH_DEEPWOOD = build_registry(*BRAIDS_REGISTRY.values(), DEEPWOOD_LEGATE)

# Registry with Offalsnout added.
_BRAIDS_WITH_OFFALSNOUT = build_registry(*BRAIDS_REGISTRY.values(), OFFALSNOUT)


def _0mana_to_1cmc(registry: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the four 0-cost creatures with 1-CMC creatures."""
    base = remove_cards(registry, "memnite", "ornithopter", "phyrexian_walker", "shield_sphere")
    return build_registry(*base.values(), *_1CMC_CREATURES)


# Define scenarios for comparison
SCENARIOS: List[Dict[str, Any]] = [
    # -- Baseline ----------------------------------------------------------------
    {
        "id": "baseline",
        "name": "Baseline",
        "registry": BRAIDS_REGISTRY,
        "cost": BRAIDS_COST,
    },
    # -- Fast mana: individual card removal --------------------------------------
    {
        "id": "no_lions_eye_diamond",
        "name": "No LED",
        "registry": remove_card(BRAIDS_REGISTRY, "lions_eye_diamond"),
        "cost": BRAIDS_COST,
    },
    {
        "id": "no_jeweled_lotus",
        "name": "No JLO",
        "registry": remove_card(_BRAIDS_WITH_JLO, "jeweled_lotus"),
        "cost": BRAIDS_COST,
    },
    {
        "id": "no_gemstone_caverns",
        "name": "No Gemstone Caverns",
        "registry": remove_card(BRAIDS_REGISTRY, "gemstone_caverns"),
        "cost": BRAIDS_COST,
    },
    {
        "id": "no_jeweled_amulet",
        "name": "No Jeweled Amulet",
        "registry": remove_card(BRAIDS_REGISTRY, "jeweled_amulet"),
        "cost": BRAIDS_COST,
    },
    # -- Lands -------------------------------------------------------------------
    {
        "id": "no_peat_bog",
        "name": "No Peat Bog",
        "registry": remove_card(BRAIDS_REGISTRY, "peat_bog"),
        "cost": BRAIDS_COST,
    },
    {
        "id": "peat_bog_to_swamp",
        "name": "Peat Bog -> Swamp",
        "registry": build_registry(*remove_card(BRAIDS_REGISTRY, "peat_bog").values(), _SWAMP_A),
        "cost": BRAIDS_COST,
    },
    # -- Creatures ---------------------------------------------------------------
    {
        "id": "no_0mana_creatures",
        "name": "No 0-Mana Creatures",
        "registry": remove_cards(BRAIDS_REGISTRY, "memnite", "ornithopter", "phyrexian_walker", "shield_sphere"),
        "cost": BRAIDS_COST,
    },
    {
        "id": "1cmc_creatures",
        "name": "1-CMC Creatures",
        "registry": _0mana_to_1cmc(BRAIDS_REGISTRY),
        "cost": BRAIDS_COST,
    },
    {
        "id": "no_grief",
        "name": "No Grief",
        "registry": remove_card(BRAIDS_REGISTRY, "grief"),
        "cost": BRAIDS_COST,
    },
    {
        "id": "no_deepwood_legate",
        "name": "No Deepwood Legate",
        "registry": remove_card(_BRAIDS_WITH_DEEPWOOD, "deepwood_legate"),
        "cost": BRAIDS_COST,
    },
    {
        "id": "no_offalsnout",
        "name": "No Offalsnout",
        "registry": remove_card(_BRAIDS_WITH_OFFALSNOUT, "offalsnout"),
        "cost": BRAIDS_COST,
    },
    # -- Equipment ---------------------------------------------------------------
    {
        "id": "no_drum",
        "name": "No Springleaf Drum",
        "registry": remove_card(BRAIDS_REGISTRY, "springleaf_drum"),
        "cost": BRAIDS_COST,
    },
    {
        "id": "no_mantle",
        "name": "No Paradise Mantle",
        "registry": remove_card(BRAIDS_REGISTRY, "paradise_mantle"),
        "cost": BRAIDS_COST,
    },
]

PROVIDER_SPECS: List[ProviderSpec] = [
    (HandGenerator, {"hand", "simulation_id", "mulligan_count", "scenario_id"}),
    (Turn1, None),
    (Turn2, None),
    (MulliganResult, {"MulliganResult"}),
    (Proportion, None),
    (MeanMulliganDepth, {"MeanMulliganDepth"}),
    (AverageKeptHandSize, {"AverageKeptHandSize"}),
    (CardCooccurrence, {"CardCooccurrence"}),
    (ConvergencePlot, {"ConvergencePlot"}),
    (HandCompositionPlot, {"HandCompositionPlot"}),
]

FEATURE_NAMES = [
    "MulliganResult",
    "hand",
    "simulation_id",
    "mulligan_count",
    "scenario_id",
    "hand__t1",
    "hand__t1__t2",
    "hand__t1__proportion",
    "hand__t1__t2__proportion",
    "MeanMulliganDepth",
    "AverageKeptHandSize",
    "CardCooccurrence",
    "ConvergencePlot",
    "HandCompositionPlot",
]


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
    ax.set_title(f"Scenario Comparison: {experiment_id}", fontsize=14, fontweight="bold")
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

    pl.DataFrame(rows).sort("t1_delta", descending=True).write_csv(str(out_path))

    return str(out_path)


def main() -> None:
    experiment_id = "comparison"

    all_metrics = run_experiment(
        scenarios=SCENARIOS,
        provider_specs=PROVIDER_SPECS,
        feature_names=FEATURE_NAMES,
        n_simulations=N_SIMULATIONS,
        experiment_id=experiment_id,
        title="Braids Mulligan Simulator: Multi-Scenario Comparison",
        show_plots=True,
    )

    # Generate comparison outputs (unique to multi-scenario comparison)
    scenario_plot_path = create_scenario_comparison_plot(all_metrics, experiment_id)
    delta_table_path = create_card_delta_table(all_metrics, experiment_id)

    print_section("Generated Comparison Outputs")
    print(f"  {'Scenario Comparison Plot:':<28} {scenario_plot_path or 'Not generated'}")
    print(f"  {'Card Delta Table:':<28} {delta_table_path or 'Not generated'}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    main()
