"""Tests for the ConvergencePlot feature group."""

from pathlib import Path
from typing import Any

import polars as pl

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from card_registries.mono.black.braids import BRAIDS_COST, BRAIDS_REGISTRY
from cedh_mulligan_simulator.feature_groups.plots import ConvergencePlot  # noqa: F401
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider
from tests.feature_groups.test_data_providers import SpecificHandProvider


def _concat_results(results: list[Any]) -> pl.DataFrame:
    seen_cols: set[str] = set()
    parts: list[pl.DataFrame] = []
    for r in results:
        if isinstance(r, pl.DataFrame):
            new_cols = [c for c in r.columns if c not in seen_cols]
            if new_cols:
                parts.append(r.select(new_cols))
                seen_cols.update(new_cols)
    return pl.concat(parts, how="horizontal")


def _opts(n: int, tmp_path: Path, experiment_id: str | None = None) -> Options:
    group = {"scenario_id": "test", "n_simulations": n}
    if experiment_id is not None:
        group["experiment_id"] = experiment_id
    return Options(
        group=group,
        context={"card_registry": BRAIDS_REGISTRY, "commander_cost": BRAIDS_COST, "plot_dir": str(tmp_path)},
    )


def _run(n: int, tmp_path: Path, experiment_id: str | None = None) -> pl.DataFrame:
    opts = _opts(n, tmp_path, experiment_id)
    result = mlodaAPI.run_all(
        features=[
            Feature("ConvergencePlot", options=opts),
            Feature("MulliganResult", options=opts),
            Feature("simulation_id", options=opts),
        ],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandMulliganTestDataProvider, SpecificHandProvider}),
    )
    return _concat_results(result)


def test_column_exists(tmp_path: Path) -> None:
    """ConvergencePlot column should exist in the output DataFrame."""
    df = _run(20, tmp_path)
    assert "ConvergencePlot" in df.columns


def test_png_file_exists(tmp_path: Path) -> None:
    """A convergence.png file should be created on disk."""
    _run(20, tmp_path)
    png = tmp_path / "test" / "convergence.png"
    assert png.exists()


def test_png_magic_bytes(tmp_path: Path) -> None:
    """The saved file should start with the PNG magic bytes."""
    _run(20, tmp_path)
    png = tmp_path / "test" / "convergence.png"
    header = png.read_bytes()[:4]
    assert header == b"\x89PNG"


def test_non_kept_rows_are_nan(tmp_path: Path) -> None:
    """Non-kept rows should have null in the ConvergencePlot column."""
    df = _run(20, tmp_path)
    not_kept = df.filter(~pl.col("MulliganResult"))
    assert not_kept["ConvergencePlot"].is_null().all()


def test_path_contains_scenario_id(tmp_path: Path) -> None:
    """The output path should include the scenario_id as a subdirectory."""
    df = _run(20, tmp_path)
    kept = df.filter(pl.col("MulliganResult"))
    path_val = kept["ConvergencePlot"][0]
    assert "test" in str(path_val)


def test_experiment_id_overrides_scenario_id(tmp_path: Path) -> None:
    """When experiment_id is set, it should be used instead of scenario_id."""
    df = _run(20, tmp_path, experiment_id="my_experiment")
    kept = df.filter(pl.col("MulliganResult"))
    path_val = kept["ConvergencePlot"][0]
    assert "my_experiment" in str(path_val)
    png = tmp_path / "my_experiment" / "convergence.png"
    assert png.exists()
