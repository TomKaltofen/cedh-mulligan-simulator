"""ConvergencePlot — derived feature that saves a convergence line chart as PNG."""

from pathlib import Path
from typing import Any, Optional, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from mloda.provider import FeatureGroup, FeatureSet  # noqa: E402
from mloda.user import Feature, FeatureName, Options  # noqa: E402


class ConvergencePlot(FeatureGroup):
    """Saves a convergence plot PNG showing T1/T2 castability vs simulation count.

    Usage: ``Feature("ConvergencePlot", options=opts)``

    Depends on the raw boolean columns ``hand__t1`` and ``hand__t1__t2``, plus
    ``MulliganResult`` and ``simulation_id``.  Computes running means internally
    and renders a two-line convergence chart.

    Options (group):
        ``experiment_id`` — subdirectory name; falls back to ``scenario_id``, then ``"default"``

    Options (context):
        ``plot_dir`` — base directory override (e.g. ``tmp_path`` for tests)

    Output column contains the file path for kept rows, ``None`` for non-kept rows.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature("hand__t1", options=options),
            Feature("hand__t1__t2", options=options),
            Feature("MulliganResult", options=options),
            Feature("simulation_id", options=options),
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
        out_path = out_dir / "convergence.png"

        kept_mask = df["MulliganResult"].cast(pl.Boolean)
        kept = df.filter(kept_mask).sort("simulation_id")

        t1_list = kept["hand__t1"].cast(pl.Float64).to_list()
        t2_list = kept["hand__t1__t2"].cast(pl.Float64).to_list()
        n = len(t1_list)
        t1_running = [0.0] * n
        t2_running = [0.0] * n
        s1, s2 = 0.0, 0.0
        for i in range(n):
            s1 += float(t1_list[i]) if t1_list[i] is not None else 0.0
            s2 += float(t2_list[i]) if t2_list[i] is not None else 0.0
            t1_running[i] = s1 / (i + 1)
            t2_running[i] = s2 / (i + 1)

        x = range(1, len(kept) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, t1_running, label="T1 Castability", linewidth=1.5)
        ax.plot(x, t2_running, label="T2 Castability", linewidth=1.5)
        ax.set_xlabel("Simulation count (kept hands)")
        ax.set_ylabel("Running mean")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Convergence — {experiment_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

        return df.with_columns(pl.when(kept_mask).then(pl.lit(str(out_path))).otherwise(None).alias("ConvergencePlot"))
