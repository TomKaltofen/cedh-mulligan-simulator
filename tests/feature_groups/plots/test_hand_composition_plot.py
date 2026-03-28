"""Tests for HandCompositionPlot feature group."""

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.user import Feature, Options, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from cedh_mulligan_simulator.card_registry import Card, CardRegistry, Mana
from cedh_mulligan_simulator.feature_groups.mulligan.hand_generator import HandGenerator
from cedh_mulligan_simulator.feature_groups.mulligan.mulligan_result import MulliganResult
from cedh_mulligan_simulator.feature_groups.plots.hand_composition_plot import HandCompositionPlot  # noqa: F401
from cedh_mulligan_simulator.feature_groups.statistics.card_type_count import _count_type
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
def sample_data() -> pl.DataFrame:
    """Sample hands with kept/mulligan decisions."""
    return pl.DataFrame(
        {
            "hand": [
                ["Swamp", "Dark Ritual", "Mana Crypt", "filler", "filler", "filler", "filler"],  # keep
                ["Chrome Mox", "Swamp", "Ophiomancer", "filler", "filler", "filler", "filler"],  # mulligan
                ["Dark Ritual", "Mana Crypt", "Necromancy", "filler", "filler", "filler", "filler"],  # keep
                ["Swamp", "Mana Crypt", "Ophiomancer", "filler", "filler", "filler", "filler"],  # keep
            ],
            "MulliganResult": [True, False, True, True],
        }
    )


def _run_plot(
    data: pl.DataFrame, tmp_path: Path, sample_registry: CardRegistry, exp_id: str = "test_exp"
) -> pl.DataFrame:
    """Helper to run HandCompositionPlot feature and return result DataFrame."""
    HandMulliganTestDataProvider._test_data = data
    opts = Options(
        group={"experiment_id": exp_id},
        context={"plot_dir": str(tmp_path), "card_registry": sample_registry},
    )
    result = mlodaAPI.run_all(
        features=[
            Feature("hand"),
            Feature("MulliganResult"),
            Feature("HandCompositionPlot", options=opts),
        ],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandGenerator, MulliganResult, SpecificHandProvider}),
    )
    return _concat_results(result)


def test_column_exists(tmp_path: Path, sample_data: pl.DataFrame, sample_registry: CardRegistry) -> None:
    """Column is created."""
    df = _run_plot(sample_data, tmp_path, sample_registry)
    assert "HandCompositionPlot" in df.columns


def test_png_file_exists(tmp_path: Path, sample_data: pl.DataFrame, sample_registry: CardRegistry) -> None:
    """PNG file is created on disk."""
    _run_plot(sample_data, tmp_path, sample_registry)
    png_path = tmp_path / "test_exp" / "hand_composition.png"
    assert png_path.exists()


def test_png_magic_bytes(tmp_path: Path, sample_data: pl.DataFrame, sample_registry: CardRegistry) -> None:
    """PNG file has correct magic bytes."""
    _run_plot(sample_data, tmp_path, sample_registry)
    png_path = tmp_path / "test_exp" / "hand_composition.png"
    with open(png_path, "rb") as f:
        magic = f.read(8)
    assert magic == b"\x89PNG\r\n\x1a\n"


def test_non_kept_rows_are_nan(tmp_path: Path, sample_data: pl.DataFrame, sample_registry: CardRegistry) -> None:
    """Non-kept rows have null in the column."""
    df = _run_plot(sample_data, tmp_path, sample_registry)
    # Row 1 is mulligan
    assert df["HandCompositionPlot"][1] is None
    # Rows 0, 2, 3 are kept
    assert df["HandCompositionPlot"][0] is not None
    assert df["HandCompositionPlot"][2] is not None
    assert df["HandCompositionPlot"][3] is not None


def test_path_contains_scenario_id(tmp_path: Path, sample_data: pl.DataFrame, sample_registry: CardRegistry) -> None:
    """Output path contains scenario_id when no experiment_id is provided."""
    HandMulliganTestDataProvider._test_data = sample_data
    opts = Options(
        group={"scenario_id": "test_scenario"},
        context={"plot_dir": str(tmp_path), "card_registry": sample_registry},
    )
    result = mlodaAPI.run_all(
        features=[
            Feature("hand"),
            Feature("MulliganResult"),
            Feature("HandCompositionPlot", options=opts),
        ],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups({HandGenerator, MulliganResult, SpecificHandProvider}),
    )
    df = _concat_results(result)
    kept_path = df["HandCompositionPlot"][0]
    assert kept_path is not None
    assert "test_scenario" in kept_path


def test_experiment_id_overrides_scenario_id(
    tmp_path: Path, sample_data: pl.DataFrame, sample_registry: CardRegistry
) -> None:
    """experiment_id in group options overrides scenario_id in path."""
    _run_plot(sample_data, tmp_path, sample_registry, exp_id="test_exp")
    # Should use experiment_id, not scenario_id
    png_path = tmp_path / "test_exp" / "hand_composition.png"
    assert png_path.exists()
    assert not (tmp_path / "ignored_scenario" / "hand_composition.png").exists()


def test_count_type_helper(sample_registry: CardRegistry) -> None:
    """Unit test for _count_type helper function."""
    hand = ["Swamp", "Dark Ritual", "Mana Crypt", "Ophiomancer", "Necromancy"]
    assert _count_type(hand, sample_registry, "land") == 1
    assert _count_type(hand, sample_registry, "artifact") == 1
    assert _count_type(hand, sample_registry, "creature") == 1
    assert _count_type(hand, sample_registry, "ritual") == 1
    assert _count_type(hand, sample_registry, "enchantment") == 1  # Necromancy
    assert _count_type(hand, sample_registry, "nonexistent") == 0


def test_filler_cards_not_counted(sample_registry: CardRegistry) -> None:
    """Filler cards are excluded from type counts."""
    hand = ["filler", "filler", "Swamp", "filler", "Dark Ritual"]
    assert _count_type(hand, sample_registry, "land") == 1
    assert _count_type(hand, sample_registry, "ritual") == 1
    # Total non-filler cards with known types
    total = sum(_count_type(hand, sample_registry, t) for t in ["land", "artifact", "creature", "ritual"])
    # Should be exactly 2 (Swamp land + Dark Ritual ritual), not counting filler
    assert total == 2
