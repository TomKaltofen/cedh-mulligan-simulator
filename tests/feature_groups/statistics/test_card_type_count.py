"""Tests for CardTypeCount feature group."""

import polars as pl
import pytest

from cedh_mulligan_simulator.card_registry import Card, CardRegistry, Mana
from cedh_mulligan_simulator.feature_groups.statistics.card_type_count import CardTypeCount, _count_type


@pytest.fixture
def sample_registry() -> CardRegistry:
    """Minimal registry for testing."""
    return {
        "Swamp": Card(name="Swamp", type="land"),
        "Dark Ritual": Card(name="Dark Ritual", type="ritual", cost=Mana(1, black=1)),
        "Mana Crypt": Card(name="Mana Crypt", type="artifact", cost=Mana(0)),
        "Chrome Mox": Card(name="Chrome Mox", type="artifact", cost=Mana(0)),
        "Ophiomancer": Card(name="Ophiomancer", type="creature", cost=Mana(3, black=1)),
        "filler": Card(name="filler", type="filler"),
    }


@pytest.fixture
def sample_hands() -> pl.DataFrame:
    """Three sample hands with known composition."""
    return pl.DataFrame(
        {
            "hand": [
                ["Swamp", "Dark Ritual", "Mana Crypt", "Ophiomancer", "filler", "filler", "filler"],
                ["Chrome Mox", "Swamp", "Swamp", "filler", "filler", "filler", "filler"],
                ["Dark Ritual", "Dark Ritual", "filler", "filler", "filler", "filler", "filler"],
            ]
        }
    )


class MockFeatures:
    """Mock FeatureSet for testing."""

    def __init__(self, feature_names: list[str], card_registry: CardRegistry):
        self.feature_names = feature_names
        self.card_registry = card_registry
        self.features = [MockFeature(name) for name in feature_names]

    def get_options_key(self, key: str) -> object:
        if key == "card_registry":
            return self.card_registry
        return None


class MockFeature:
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name


def test_column_exists(sample_hands: pl.DataFrame, sample_registry: CardRegistry) -> None:
    """Column is created."""
    features = MockFeatures(["land__type_count"], sample_registry)
    df = CardTypeCount.calculate_feature(sample_hands, features)  # type: ignore[arg-type]
    assert "land__type_count" in df.columns


def test_all_rows_have_values(sample_hands: pl.DataFrame, sample_registry: CardRegistry) -> None:
    """No null values — all rows computed."""
    features = MockFeatures(["land__type_count"], sample_registry)
    df = CardTypeCount.calculate_feature(sample_hands, features)  # type: ignore[arg-type]
    assert not df["land__type_count"].is_null().any()


def test_counts_non_negative(sample_hands: pl.DataFrame, sample_registry: CardRegistry) -> None:
    """All counts are >= 0."""
    features = MockFeatures(["artifact__type_count"], sample_registry)
    df = CardTypeCount.calculate_feature(sample_hands, features)  # type: ignore[arg-type]
    assert (df["artifact__type_count"] >= 0).all()


def test_counts_within_hand_size(sample_hands: pl.DataFrame, sample_registry: CardRegistry) -> None:
    """Counts never exceed hand size."""
    features = MockFeatures(["creature__type_count"], sample_registry)
    df = CardTypeCount.calculate_feature(sample_hands, features)  # type: ignore[arg-type]
    hand_sizes = sample_hands["hand"].map_elements(len, return_dtype=pl.Int64)
    assert (df["creature__type_count"] <= hand_sizes).all()


def test_multiple_types(sample_hands: pl.DataFrame, sample_registry: CardRegistry) -> None:
    """Can request multiple card types simultaneously."""
    features = MockFeatures(["artifact__type_count", "creature__type_count"], sample_registry)
    df = CardTypeCount.calculate_feature(sample_hands, features)  # type: ignore[arg-type]
    assert "artifact__type_count" in df.columns
    assert "creature__type_count" in df.columns


def test_manual_verification(sample_hands: pl.DataFrame, sample_registry: CardRegistry) -> None:
    """Verify counts match expected values for known hands."""
    features = MockFeatures(
        ["land__type_count", "artifact__type_count", "creature__type_count", "ritual__type_count"],
        sample_registry,
    )
    df = CardTypeCount.calculate_feature(sample_hands, features)  # type: ignore[arg-type]

    # Hand 0: 1 land (Swamp), 1 artifact (Mana Crypt), 1 creature (Ophiomancer), 1 ritual (Dark Ritual)
    assert df["land__type_count"][0] == 1
    assert df["artifact__type_count"][0] == 1
    assert df["creature__type_count"][0] == 1
    assert df["ritual__type_count"][0] == 1

    # Hand 1: 2 lands (Swamp, Swamp), 1 artifact (Chrome Mox)
    assert df["land__type_count"][1] == 2
    assert df["artifact__type_count"][1] == 1
    assert df["creature__type_count"][1] == 0
    assert df["ritual__type_count"][1] == 0

    # Hand 2: 2 rituals (Dark Ritual, Dark Ritual)
    assert df["land__type_count"][2] == 0
    assert df["artifact__type_count"][2] == 0
    assert df["creature__type_count"][2] == 0
    assert df["ritual__type_count"][2] == 2


def test_count_type_helper(sample_registry: CardRegistry) -> None:
    """Unit test for _count_type helper function."""
    hand = ["Swamp", "Dark Ritual", "Mana Crypt", "Ophiomancer", "filler"]
    assert _count_type(hand, sample_registry, "land") == 1
    assert _count_type(hand, sample_registry, "artifact") == 1
    assert _count_type(hand, sample_registry, "creature") == 1
    assert _count_type(hand, sample_registry, "ritual") == 1
    assert _count_type(hand, sample_registry, "filler") == 1
    assert _count_type(hand, sample_registry, "nonexistent") == 0
