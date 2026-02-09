"""Turn simulation result dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional

from cedh_mulligan_simulator.card_registry import Mana


@dataclass
class Battlefield:
    """Permanents on the battlefield (not graveyard or exile)."""

    lands: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    creatures: List[str] = field(default_factory=list)
    equipment_attached: List[str] = field(default_factory=list)

    @property
    def all_permanents(self) -> List[str]:
        """All cards on the battlefield."""
        return self.lands + self.artifacts + self.creatures + self.equipment_attached


@dataclass
class GameState:
    """Complete game state across all zones."""

    hand: List[str] = field(default_factory=list)
    library: List[str] = field(default_factory=list)
    battlefield: Battlefield = field(default_factory=Battlefield)
    graveyard: List[str] = field(default_factory=list)
    exile: List[str] = field(default_factory=list)


@dataclass
class TurnResult:
    """Complete result of simulating one turn."""

    turn_number: int
    state_before: GameState
    state_after: GameState
    drawn_card: Optional[str]
    land_played: Optional[str]
    cards_played: List[str]
    mana_remaining: Mana
    can_cast_commander: bool
