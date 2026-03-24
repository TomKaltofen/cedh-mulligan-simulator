"""Integration tests for Serum Powder: full HandGenerator -> Turn -> MulliganResult pipeline."""

import polars as pl

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from card_database.colorless import SERUM_POWDER
from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.card_registry import CardRegistry, build_registry
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider, SpecificHandProvider

_N = 2000
_BRAIDS_WITH_SP = build_registry(*BRAIDS_REGISTRY.values(), SERUM_POWDER)


def _run_pipeline(registry: CardRegistry, n: int = _N) -> pl.DataFrame:
    """Run the full mulligan pipeline and return a single merged DataFrame."""
    opts = Options(
        group={"n_simulations": n, "mulligan_steps": 4},
        context={"card_registry": registry, "commander_cost": BRAIDS_COST},
    )
    results = mlodaAPI.run_all(
        features=[
            Feature("hand", options=opts),
            Feature("simulation_id", options=opts),
            Feature("mulligan_count", options=opts),
            Feature("MulliganResult", options=opts),
            Feature("hand__t1", options=opts),
            Feature("hand__t1__t2", options=opts),
        ],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    seen: set[str] = set()
    parts: list[pl.DataFrame] = []
    for r in results:
        if isinstance(r, pl.DataFrame):
            new_cols = [c for c in r.columns if c not in seen]
            if new_cols:
                parts.append(r.select(new_cols))
                seen.update(new_cols)
    return pl.concat(parts, how="horizontal")


class TestSerumPowderIntegration:
    def test_pipeline_runs_with_sp(self) -> None:
        """Full pipeline with SP in registry should produce rows and MulliganResult."""
        df = _run_pipeline(_BRAIDS_WITH_SP, n=200)
        # With SP, some sims have extra rows (5 instead of 4)
        assert len(df) >= 200 * 4
        assert "MulliganResult" in df.columns
        kept = df.filter(pl.col("MulliganResult").cast(pl.Boolean))
        assert len(kept) > 0, "At least some simulations should find a keepable hand"

    def test_serum_powder_improves_or_matches_mulligan_rate(self) -> None:
        """SP scenario should have T1+T2 castability >= baseline (or within sampling noise).

        With 2000 sims the 95% CI half-width is ~2pp, so we use a 3pp buffer to avoid
        flakiness while still catching regressions.
        """
        df_baseline = _run_pipeline(BRAIDS_REGISTRY)
        df_sp = _run_pipeline(_BRAIDS_WITH_SP)

        kept_baseline = df_baseline.filter(pl.col("MulliganResult").cast(pl.Boolean))
        kept_sp = df_sp.filter(pl.col("MulliganResult").cast(pl.Boolean))

        t2_baseline_vals: list[bool] = kept_baseline["hand__t1__t2"].to_list()
        t2_sp_vals: list[bool] = kept_sp["hand__t1__t2"].to_list()
        t2_baseline = sum(t2_baseline_vals) / len(t2_baseline_vals) if t2_baseline_vals else 0.0
        t2_sp = sum(t2_sp_vals) / len(t2_sp_vals) if t2_sp_vals else 0.0

        # SP gives a free hand replacement so should not hurt castability.
        # Allow 3pp slack for Monte Carlo variance.
        assert t2_sp >= t2_baseline - 0.03, f"SP T2 rate {t2_sp:.3f} is more than 3pp below baseline {t2_baseline:.3f}"
