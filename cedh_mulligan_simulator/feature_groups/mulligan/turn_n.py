"""Turn feature groups — Turn1, Turn2, Turn3 with shared logic via TurnFeatureBase."""

from cedh_mulligan_simulator.feature_groups.mulligan.base_turn import TurnFeatureBase


class Turn1(TurnFeatureBase):
    """Evaluates whether each hand can cast the commander on Turn 1.

    Usage: ``Feature("hand__t1")``
    """

    PREFIX_PATTERN = r".*__t1$"
    TURN_NUMBER = 1


class Turn2(TurnFeatureBase):
    """Evaluates whether each hand can cast the commander on Turn 2.

    Usage: ``Feature("hand__t1__t2")``
    """

    PREFIX_PATTERN = r".*__t2$"
    TURN_NUMBER = 2


class Turn3(TurnFeatureBase):
    """Evaluates whether each hand can cast the commander on Turn 3.

    Usage: ``Feature("hand__t1__t2__t3")``
    """

    PREFIX_PATTERN = r".*__t3$"
    TURN_NUMBER = 3
