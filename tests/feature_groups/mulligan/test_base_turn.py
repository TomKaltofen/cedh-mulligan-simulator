"""Tests for TurnFeatureBase shared utilities."""

from typing import Any

from mloda.user import FeatureName, Options

from cedh_mulligan_simulator.card_registry import CardRegistry, DEFAULT_CARD_REGISTRY, ManaRequirement
from cedh_mulligan_simulator.feature_groups.mulligan.base_turn import TurnFeatureBase


# Test stub - concrete implementation of TurnFeatureBase for testing
class StubTurn(TurnFeatureBase):
    """Test implementation of TurnFeatureBase."""

    PREFIX_PATTERN = r".*__test$"

    @classmethod
    def _simulate_turn(
        cls,
        data: Any,
        feature_name: str,
        root: str,
        registry: CardRegistry,
        commander_cost: ManaRequirement,
        draw_per_turn: bool = True,
    ) -> None:
        """Stub implementation - does nothing."""
        pass


def test_input_features_extracts_source() -> None:
    """input_features() should extract source from chained name."""
    base = StubTurn()
    opts = Options()

    # Test FeatureName object
    feature_name = FeatureName("hand__t1")
    result = base.input_features(opts, feature_name)

    assert result is not None
    assert len(result) == 1
    feature = list(result)[0]
    assert feature.name == "hand"


def test_input_features_multiple_chains() -> None:
    """input_features() should handle multiple __ separators."""
    base = StubTurn()
    opts = Options()

    # hand__t1__t2 → should extract hand__t1
    feature_name = FeatureName("hand__t1__t2")
    result = base.input_features(opts, feature_name)

    assert result is not None
    feature = list(result)[0]
    assert feature.name == "hand__t1"


def test_input_features_propagates_options() -> None:
    """input_features() should propagate options to the source feature."""
    base = StubTurn()
    custom_cost = ManaRequirement(total=4, black=3)
    opts = Options(context={"commander_cost": custom_cost})

    feature_name = FeatureName("hand__t1")
    result = base.input_features(opts, feature_name)

    assert result is not None
    feature = list(result)[0]
    assert feature.options == opts


def test_get_registry_and_cost_defaults() -> None:
    """_get_registry_and_cost() should return defaults when options not set."""

    # Create a mock FeatureSet-like object
    class MockFeatureSet:
        def get_options_key(self, key: str) -> object:
            return None

    features = MockFeatureSet()

    registry, cost = TurnFeatureBase._get_registry_and_cost(features)  # type: ignore

    assert registry == DEFAULT_CARD_REGISTRY
    assert cost == ManaRequirement()


def test_get_registry_and_cost_from_options() -> None:
    """_get_registry_and_cost() should extract from options when set."""
    custom_cost = ManaRequirement(total=4, black=3)

    # Create a mock FeatureSet-like object with options
    class MockFeatureSet:
        def __init__(self, options: dict[str, object]) -> None:
            self._options = options

        def get_options_key(self, key: str) -> object:
            return self._options.get(key)

    features = MockFeatureSet({"commander_cost": custom_cost})

    registry, cost = TurnFeatureBase._get_registry_and_cost(features)  # type: ignore

    assert cost == custom_cost
    assert registry == DEFAULT_CARD_REGISTRY  # Registry still uses default


def test_calculate_feature_template_method() -> None:
    """calculate_feature() should call _simulate_turn with correct parameters."""
    # Track what parameters were passed to _simulate_turn
    captured_params: dict[str, Any] = {}

    class TestTurnWithCapture(TurnFeatureBase):
        PREFIX_PATTERN = r".*__test$"

        @classmethod
        def _simulate_turn(
            cls,
            data: Any,
            feature_name: str,
            root: str,
            registry: CardRegistry,
            commander_cost: ManaRequirement,
            draw_per_turn: bool = True,
        ) -> None:
            """Capture parameters for testing."""
            captured_params["data"] = data
            captured_params["feature_name"] = feature_name
            captured_params["root"] = root
            captured_params["registry"] = registry
            captured_params["commander_cost"] = commander_cost
            captured_params["draw_per_turn"] = draw_per_turn

    # Create mock FeatureSet
    class MockFeatureName:
        name = "hand__t1__test"

    class MockFeatureSet:
        name_of_one_feature = MockFeatureName()

        def get_options_key(self, key: str) -> object:
            return None

    # Run calculate_feature
    data = {"hand__t1": ["some", "data"]}
    features = MockFeatureSet()
    result = TestTurnWithCapture.calculate_feature(data, features)  # type: ignore

    # Verify _simulate_turn was called with correct parameters
    assert result == data  # Should return the data dict
    assert captured_params["data"] == data
    assert captured_params["feature_name"] == "hand__t1__test"
    assert captured_params["root"] == "hand"
    assert captured_params["registry"] == DEFAULT_CARD_REGISTRY
    assert captured_params["commander_cost"] == ManaRequirement()
    assert captured_params["draw_per_turn"] is True  # Default value
