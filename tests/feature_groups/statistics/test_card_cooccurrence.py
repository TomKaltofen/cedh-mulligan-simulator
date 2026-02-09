"""Tests for CardCooccurrence feature group."""

from pathlib import Path

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from cedh_mulligan_simulator.card_registry import Card, CardRegistry, Mana
from cedh_mulligan_simulator.feature_groups.mulligan.hand_generator import HandGenerator
from cedh_mulligan_simulator.feature_groups.mulligan.mulligan_result import MulliganResult
from cedh_mulligan_simulator.feature_groups.statistics.card_cooccurrence import CardCooccurrence  # noqa: F401
from tests.feature_groups.test_data_providers import HandMulliganTestDataProvider
from tests.test_integration_mloda_specific_hands import SpecificHandProvider


@pytest.fixture
def sample_registry() -> CardRegistry:
    """Minimal registry for testing."""
    return {
        "Swamp": Card(name="Swamp", type="land"),
        "Dark Ritual": Card(name="Dark Ritual", type="ritual", cost=Mana(1, black=1)),
        "Mana Crypt": Card(name="Mana Crypt", type="artifact", cost=Mana(0)),
        "Chrome Mox": Card(name="Chrome Mox", type="artifact", cost=Mana(0)),
        "Ophiomancer": Card(name="Ophiomancer", type="creature", cost=Mana(3, black=1)),
        "Necromancy": Card(name="Necromancy", type="enchantment", cost=Mana(3, black=1)),
        "filler": Card(name="filler", type="filler"),
    }


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Sample hands with kept/mulligan decisions."""
    return pd.DataFrame(
        {
            "hand": [
                ["Swamp", "Dark Ritual", "Mana Crypt", "filler", "filler", "filler", "filler"],  # keep
                ["Chrome Mox", "Swamp", "Ophiomancer", "filler", "filler", "filler", "filler"],  # mulligan
                ["Dark Ritual", "Mana Crypt", "Necromancy", "filler", "filler", "filler", "filler"],  # keep
                ["Swamp", "Mana Crypt", "Ophiomancer", "filler", "filler", "filler", "filler"],  # keep
            ],
            "MulliganResult": ["keep", "mulligan", "keep", "keep"],
        }
    )


def _run_cooccurrence(
    data: pd.DataFrame, tmp_path: Path, sample_registry: CardRegistry, exp_id: str = "test_exp"
) -> pd.DataFrame:
    """Helper to run CardCooccurrence feature and return result DataFrame."""
    HandMulliganTestDataProvider._test_data = data
    opts = Options(
        group={"experiment_id": exp_id},
        context={"plot_dir": str(tmp_path), "card_registry": sample_registry},
    )
    result = mlodaAPI.run_all(
        features=[
            Feature("hand"),
            Feature("MulliganResult"),
            Feature("CardCooccurrence", options=opts),
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandGenerator, MulliganResult, SpecificHandProvider}),
    )
    dfs = [r for r in result if isinstance(r, pd.DataFrame)]
    return pd.concat(dfs, axis=1)


def test_column_exists(tmp_path: Path, sample_data: pd.DataFrame, sample_registry: CardRegistry) -> None:
    """Column is created."""
    df = _run_cooccurrence(sample_data, tmp_path, sample_registry)
    assert "CardCooccurrence" in df.columns


def test_csv_file_exists(tmp_path: Path, sample_data: pd.DataFrame, sample_registry: CardRegistry) -> None:
    """CSV file is created on disk."""
    _run_cooccurrence(sample_data, tmp_path, sample_registry)
    csv_path = tmp_path / "test_exp" / "card_cooccurrence.csv"
    assert csv_path.exists()


def test_csv_has_expected_header(tmp_path: Path, sample_data: pd.DataFrame, sample_registry: CardRegistry) -> None:
    """CSV has the expected columns."""
    _run_cooccurrence(sample_data, tmp_path, sample_registry)
    csv_path = tmp_path / "test_exp" / "card_cooccurrence.csv"
    csv_df = pd.read_csv(csv_path)
    assert list(csv_df.columns) == ["card_a", "card_b", "count", "frequency"]


def test_csv_has_at_most_50_rows(tmp_path: Path, sample_registry: CardRegistry) -> None:
    """CSV contains at most 50 data rows (top pairs)."""
    # Create many unique cards in kept hands
    many_cards = [f"Card{i}" for i in range(100)]
    for card in many_cards:
        sample_registry[card] = Card(name=card, type="test")

    # Each hand has 7 unique cards
    hands = [
        [f"Card{i}", f"Card{i + 1}", f"Card{i + 2}", f"Card{i + 3}", f"Card{i + 4}", f"Card{i + 5}", f"Card{i + 6}"]
        for i in range(0, 93, 7)
    ]

    data = pd.DataFrame(
        {
            "hand": hands,
            "MulliganResult": ["keep"] * len(hands),
        }
    )

    _run_cooccurrence(data, tmp_path, sample_registry)
    csv_path = tmp_path / "test_exp" / "card_cooccurrence.csv"
    csv_df = pd.read_csv(csv_path)
    assert len(csv_df) <= 50


def test_non_kept_rows_are_nan(tmp_path: Path, sample_data: pd.DataFrame, sample_registry: CardRegistry) -> None:
    """Non-kept rows have NaN (None) in the column."""
    df = _run_cooccurrence(sample_data, tmp_path, sample_registry)
    # Row 1 is mulligan
    assert pd.isna(df.loc[1, "CardCooccurrence"])
    # Rows 0, 2, 3 are kept
    assert not pd.isna(df.loc[0, "CardCooccurrence"])
    assert not pd.isna(df.loc[2, "CardCooccurrence"])
    assert not pd.isna(df.loc[3, "CardCooccurrence"])


def test_experiment_id_overrides_scenario_id(
    tmp_path: Path, sample_data: pd.DataFrame, sample_registry: CardRegistry
) -> None:
    """experiment_id in group options overrides scenario_id in path."""
    _run_cooccurrence(sample_data, tmp_path, sample_registry, exp_id="test_exp")
    # Should use experiment_id, not scenario_id
    csv_path = tmp_path / "test_exp" / "card_cooccurrence.csv"
    assert csv_path.exists()
    assert not (tmp_path / "ignored_scenario" / "card_cooccurrence.csv").exists()


def test_all_frequencies_in_range(tmp_path: Path, sample_data: pd.DataFrame, sample_registry: CardRegistry) -> None:
    """All frequencies are in [0, 1]."""
    _run_cooccurrence(sample_data, tmp_path, sample_registry)
    csv_path = tmp_path / "test_exp" / "card_cooccurrence.csv"
    csv_df = pd.read_csv(csv_path)
    assert (csv_df["frequency"] >= 0).all()
    assert (csv_df["frequency"] <= 1).all()
