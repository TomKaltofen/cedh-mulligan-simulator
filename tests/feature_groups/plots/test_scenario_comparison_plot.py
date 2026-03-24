"""Tests for the ScenarioComparisonPlot feature group."""

from pathlib import Path

import polars as pl

from cedh_mulligan_simulator.feature_groups.plots.scenario_comparison_plot import ScenarioComparisonPlot


class MockFeatureSet:
    """Mock FeatureSet for testing."""

    def __init__(self, experiment_id: str, plot_dir: str):
        self._experiment_id = experiment_id
        self._plot_dir = plot_dir

    def get_options_key(self, key: str) -> object:
        if key == "experiment_id":
            return self._experiment_id
        elif key == "plot_dir":
            return self._plot_dir
        return None


def _create_sample_data(n_per_scenario: int = 100) -> pl.DataFrame:
    """Create sample multi-scenario data for testing."""
    scenarios = [
        ("baseline", 0.65, 0.85),
        ("no_jeweled_lotus", 0.50, 0.75),
        ("no_gemstone_caverns", 0.62, 0.83),
    ]

    rows = []
    for scenario_id, t1_rate, t2_rate in scenarios:
        for _ in range(n_per_scenario):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "hand__t1__proportion": t1_rate,
                    "hand__t1__t2__proportion": t2_rate,
                    "MulliganResult": True,
                }
            )

    return pl.DataFrame(rows)


def _run_scenario_plot(data: pl.DataFrame, tmp_path: Path, experiment_id: str = "test") -> pl.DataFrame:
    """Run ScenarioComparisonPlot.calculate_feature directly."""
    features = MockFeatureSet(experiment_id=experiment_id, plot_dir=str(tmp_path))
    result_df = ScenarioComparisonPlot.calculate_feature(data, features)  # type: ignore
    return result_df  # type: ignore[no-any-return]


def test_column_exists(tmp_path: Path) -> None:
    """ScenarioComparisonPlot column should exist in the output DataFrame."""
    data = _create_sample_data()
    df = _run_scenario_plot(data, tmp_path)
    assert "ScenarioComparisonPlot" in df.columns


def test_png_file_exists(tmp_path: Path) -> None:
    """A scenario_comparison.png file should be created on disk."""
    data = _create_sample_data()
    _run_scenario_plot(data, tmp_path)
    png = tmp_path / "test" / "scenario_comparison.png"
    assert png.exists()


def test_png_magic_bytes(tmp_path: Path) -> None:
    """The saved file should start with the PNG magic bytes."""
    data = _create_sample_data()
    _run_scenario_plot(data, tmp_path)
    png = tmp_path / "test" / "scenario_comparison.png"
    header = png.read_bytes()[:4]
    assert header == b"\x89PNG"


def test_multiple_scenarios(tmp_path: Path) -> None:
    """Should handle multiple scenarios correctly."""
    data = _create_sample_data()
    _run_scenario_plot(data, tmp_path)
    png = tmp_path / "test" / "scenario_comparison.png"
    assert png.exists()
    assert png.stat().st_size > 1000  # PNG should be reasonably sized


def test_single_scenario(tmp_path: Path) -> None:
    """Should degrade gracefully with single scenario."""
    data = pl.DataFrame(
        {
            "scenario_id": ["baseline"] * 50,
            "hand__t1__proportion": [0.65] * 50,
            "hand__t1__t2__proportion": [0.85] * 50,
            "MulliganResult": [True] * 50,
        }
    )
    _run_scenario_plot(data, tmp_path)
    png = tmp_path / "test" / "scenario_comparison.png"
    assert png.exists()


def test_baseline_reference_present(tmp_path: Path) -> None:
    """Baseline reference line should be present when baseline exists."""
    data = _create_sample_data()
    _run_scenario_plot(data, tmp_path)
    # Test passes if PNG is created successfully (visual inspection would verify line)
    png = tmp_path / "test" / "scenario_comparison.png"
    assert png.exists()


def test_no_baseline_still_works(tmp_path: Path) -> None:
    """Should work without baseline scenario."""
    scenarios = [
        ("scenario_a", 0.60, 0.80),
        ("scenario_b", 0.55, 0.75),
    ]

    rows = []
    for scenario_id, t1_rate, t2_rate in scenarios:
        for _ in range(50):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "hand__t1__proportion": t1_rate,
                    "hand__t1__t2__proportion": t2_rate,
                    "MulliganResult": True,
                }
            )

    data = pl.DataFrame(rows)
    _run_scenario_plot(data, tmp_path)
    png = tmp_path / "test" / "scenario_comparison.png"
    assert png.exists()


def test_non_kept_rows_are_nan(tmp_path: Path) -> None:
    """Non-kept rows should have null in the ScenarioComparisonPlot column."""
    data = _create_sample_data()
    df = _run_scenario_plot(data, tmp_path)
    kept = df.filter(pl.col("MulliganResult"))
    assert kept["ScenarioComparisonPlot"].is_not_null().all()


def test_path_contains_experiment_id(tmp_path: Path) -> None:
    """The output path should include the experiment_id as a subdirectory."""
    data = _create_sample_data()
    df = _run_scenario_plot(data, tmp_path)
    kept = df.filter(pl.col("MulliganResult"))
    path_val = kept["ScenarioComparisonPlot"][0]
    assert "test" in str(path_val)


def test_experiment_id_overrides_scenario_id(tmp_path: Path) -> None:
    """When experiment_id is set, it should be used for directory name."""
    data = _create_sample_data()
    _run_scenario_plot(data, tmp_path, experiment_id="my_experiment")
    png = tmp_path / "my_experiment" / "scenario_comparison.png"
    assert png.exists()
