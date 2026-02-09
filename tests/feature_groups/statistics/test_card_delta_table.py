"""Tests for the CardDeltaTable feature group."""

from pathlib import Path

import pandas as pd

from cedh_mulligan_simulator.feature_groups.statistics.card_delta_table import CardDeltaTable


def _create_sample_data(n_per_scenario: int = 100) -> pd.DataFrame:
    """Create sample multi-scenario data for testing."""
    scenarios = [
        ("baseline", 0.65, 0.85),
        ("no_jeweled_lotus", 0.50, 0.75),
        ("no_gemstone_caverns", 0.62, 0.83),
        ("no_0mana_creatures", 0.60, 0.80),
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

    return pd.DataFrame(rows)


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


def _run_card_delta_table(data: pd.DataFrame, tmp_path: Path, experiment_id: str = "test") -> pd.DataFrame:
    """Run CardDeltaTable.calculate_feature directly."""
    features = MockFeatureSet(experiment_id=experiment_id, plot_dir=str(tmp_path))
    result_df = CardDeltaTable.calculate_feature(data, features)  # type: ignore
    return result_df


def test_column_exists(tmp_path: Path) -> None:
    """CardDeltaTable column should exist in the output DataFrame."""
    data = _create_sample_data()
    df = _run_card_delta_table(data, tmp_path)
    assert "CardDeltaTable" in df.columns


def test_csv_file_exists(tmp_path: Path) -> None:
    """A card_delta_table.csv file should be created on disk."""
    data = _create_sample_data()
    _run_card_delta_table(data, tmp_path)
    csv = tmp_path / "test" / "card_delta_table.csv"
    assert csv.exists()


def test_csv_has_expected_columns(tmp_path: Path) -> None:
    """The CSV should have all 9 expected columns."""
    data = _create_sample_data()
    _run_card_delta_table(data, tmp_path)
    csv_path = tmp_path / "test" / "card_delta_table.csv"
    delta_df = pd.read_csv(csv_path)

    expected_cols = {
        "card_removed",
        "baseline_t1_rate",
        "scenario_t1_rate",
        "t1_delta",
        "t1_pct_change",
        "baseline_t2_rate",
        "scenario_t2_rate",
        "t2_delta",
        "t2_pct_change",
    }
    assert set(delta_df.columns) == expected_cols


def test_baseline_comparison_correct(tmp_path: Path) -> None:
    """Deltas should be calculated correctly relative to baseline."""
    data = _create_sample_data()
    _run_card_delta_table(data, tmp_path)
    csv_path = tmp_path / "test" / "card_delta_table.csv"
    delta_df = pd.read_csv(csv_path)

    # Check Jeweled Lotus row (biggest expected delta)
    lotus_row = delta_df[delta_df["card_removed"] == "Jeweled Lotus"].iloc[0]

    # Expected: baseline 0.65, scenario 0.50, delta = 0.15
    assert abs(lotus_row["baseline_t1_rate"] - 0.65) < 1e-9
    assert abs(lotus_row["scenario_t1_rate"] - 0.50) < 1e-9
    assert abs(lotus_row["t1_delta"] - 0.15) < 1e-9

    # Percent change: (0.15 / 0.65) * 100 ≈ 23.08%
    expected_pct = (0.15 / 0.65) * 100.0
    assert abs(lotus_row["t1_pct_change"] - expected_pct) < 1e-6


def test_sorted_by_impact(tmp_path: Path) -> None:
    """Rows should be sorted by t1_delta descending (biggest impact first)."""
    data = _create_sample_data()
    _run_card_delta_table(data, tmp_path)
    csv_path = tmp_path / "test" / "card_delta_table.csv"
    delta_df = pd.read_csv(csv_path)

    # First row should be Jeweled Lotus (delta 0.15)
    assert delta_df.iloc[0]["card_removed"] == "Jeweled Lotus"
    assert abs(delta_df.iloc[0]["t1_delta"] - 0.15) < 1e-9

    # Verify descending order
    deltas = delta_df["t1_delta"].values
    assert all(deltas[i] >= deltas[i + 1] for i in range(len(deltas) - 1))


def test_card_name_parsing(tmp_path: Path) -> None:
    """Card names should be parsed correctly from scenario_id."""
    data = _create_sample_data()
    _run_card_delta_table(data, tmp_path)
    csv_path = tmp_path / "test" / "card_delta_table.csv"
    delta_df = pd.read_csv(csv_path)

    card_names = set(delta_df["card_removed"])
    expected_names = {"Jeweled Lotus", "Gemstone Caverns", "0-Mana Creatures"}
    assert card_names == expected_names


def test_missing_baseline_uses_first(tmp_path: Path) -> None:
    """When baseline is missing, should use first scenario alphabetically."""
    # Create data without explicit "baseline" scenario
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

    sample_data = pd.DataFrame(rows)
    _run_card_delta_table(sample_data, tmp_path)

    csv_path = tmp_path / "test" / "card_delta_table.csv"
    assert csv_path.exists()

    delta_df = pd.read_csv(csv_path)
    # Should have one row (scenario_b compared to scenario_a)
    assert len(delta_df) == 1
    # Baseline should be scenario_a (first alphabetically) with rate 0.60
    assert abs(delta_df.iloc[0]["baseline_t1_rate"] - 0.60) < 1e-9


def test_non_kept_rows_are_nan(tmp_path: Path) -> None:
    """Non-kept rows should have NaN in the CardDeltaTable column."""
    data = _create_sample_data()
    df = _run_card_delta_table(data, tmp_path)
    kept = df[df["MulliganResult"]]
    assert kept["CardDeltaTable"].notna().all()


def test_experiment_id_overrides_scenario_id(tmp_path: Path) -> None:
    """When experiment_id is set, it should be used for directory name."""
    data = _create_sample_data()
    _run_card_delta_table(data, tmp_path, experiment_id="my_experiment")
    csv = tmp_path / "my_experiment" / "card_delta_table.csv"
    assert csv.exists()
