"""Registry file code generator.

Writes card_registries/<output_name>.py from a list of DeckRegistryEntry objects.
Known cards are re-imported from card_database; new cards are defined inline.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from cedh_mulligan_simulator.card_registry import Card, Mana, ManaRequirement
from deck_importer.builder import DeckRegistryEntry


# ── Mana repr helpers ──────────────────────────────────────────────────────


def _mana_repr(mana: Mana) -> str:
    """Produce Python source text for a Mana literal."""
    # Check for any_color(N) pattern: all non-zero colour fields equal total
    if (
        mana.total > 0
        and mana.white == mana.total
        and mana.blue == mana.total
        and mana.black == mana.total
        and mana.red == mana.total
        and mana.green == mana.total
    ):
        return f"any_color({mana.total})"

    if mana.total == 0 and mana.white == 0 and mana.blue == 0 and mana.black == 0 and mana.red == 0 and mana.green == 0:
        return "Mana()"

    parts: List[str] = [str(mana.total)]
    if mana.white > 0:
        parts.append(f"white={mana.white}")
    if mana.blue > 0:
        parts.append(f"blue={mana.blue}")
    if mana.black > 0:
        parts.append(f"black={mana.black}")
    if mana.red > 0:
        parts.append(f"red={mana.red}")
    if mana.green > 0:
        parts.append(f"green={mana.green}")
    return f"Mana({', '.join(parts)})"


def _mana_req_repr(req: ManaRequirement) -> str:
    """Produce Python source text for a ManaRequirement literal."""
    parts: List[str] = []
    parts.append(f"total={req.total}")
    if req.white > 0:
        parts.append(f"white={req.white}")
    if req.blue > 0:
        parts.append(f"blue={req.blue}")
    if req.black > 0:
        parts.append(f"black={req.black}")
    if req.red > 0:
        parts.append(f"red={req.red}")
    if req.green > 0:
        parts.append(f"green={req.green}")
    return f"ManaRequirement({', '.join(parts)})"


# ── Card line renderer ─────────────────────────────────────────────────────


def _render_new_card(entry: DeckRegistryEntry) -> str:
    """Render a newly inferred Card as a Python assignment line."""
    card = entry.card
    var = card.name.upper()

    line = _render_card_expr(card)
    assignment = f"{var} = {line}"
    if entry.needs_review:
        assignment += f"  # NEEDS_REVIEW: {entry.review_reason}"
    return assignment


def _render_card_expr(card: Card) -> str:
    """Return the right-hand side expression for a Card definition."""
    ctype = card.type

    if ctype == "land":
        return _render_land(card)
    if ctype in ("creature", "artifact_creature"):
        return _render_creature(card)
    if ctype == "ritual":
        return _render_ritual(card)
    if ctype == "tutor":
        return _render_tutor(card)
    if ctype == "artifact":
        return _render_artifact(card)
    # Generic fallback
    return _render_generic(card)


def _render_land(card: Card) -> str:
    args: List[str] = [f'"{card.name}"']
    if card.t1_mana is not None:
        args.append(f"t1_mana={_mana_repr(card.t1_mana)}")
    if card.t2_mana is not None:
        args.append(f"t2_mana={_mana_repr(card.t2_mana)}")
    if card.requires is not None:
        args.append(f'requires="{card.requires}"')
    return f"land({', '.join(args)})"


def _render_creature(card: Card) -> str:
    args: List[str] = [f'"{card.name}"']
    if card.cost is not None:
        args.append(f"cost={_mana_repr(card.cost)}")
    if card.cmc is not None:
        args.append(f"cmc={card.cmc}")
    if card.evoke_cost is not None:
        args.append(f"evoke_cost={_mana_repr(card.evoke_cost)}")
    if card.produces is not None:
        args.append(f"produces={_mana_repr(card.produces)}")
    return f"creature({', '.join(args)})"


def _render_ritual(card: Card) -> str:
    args: List[str] = [f'"{card.name}"']
    if card.cost is not None:
        args.append(f"cost={_mana_repr(card.cost)}")
    if card.produces is not None:
        args.append(f"produces={_mana_repr(card.produces)}")
    return f"ritual({', '.join(args)})"


def _render_tutor(card: Card) -> str:
    args: List[str] = [f'"{card.name}"']
    if card.cost is not None:
        args.append(f"cost={_mana_repr(card.cost)}")
    if card.cmc is not None:
        args.append(f"cmc={card.cmc}")
    if card.destination is not None:
        args.append(f'destination="{card.destination}"')
    return f"tutor({', '.join(args)})"


def _render_artifact(card: Card) -> str:
    cost = card.cost
    if cost is not None and cost.total == 0:
        args: List[str] = [f'"{card.name}"']
        if card.produces is not None:
            args.append(f"produces={_mana_repr(card.produces)}")
        if card.requires is not None:
            args.append(f'requires="{card.requires}"')
        return f"artifact_0({', '.join(args)})"
    if cost is not None and cost.total == 1:
        args = [f'"{card.name}"']
        if card.produces is not None:
            args.append(f"produces={_mana_repr(card.produces)}")
        return f"artifact_1({', '.join(args)})"
    return _render_generic(card)


def _render_generic(card: Card) -> str:
    args: List[str] = [f'name="{card.name}"', f'type="{card.type}"']
    if card.cost is not None:
        args.append(f"cost={_mana_repr(card.cost)}")
    if card.cmc is not None:
        args.append(f"cmc={card.cmc}")
    return f"Card({', '.join(args)})"


# ── Import collection helpers ──────────────────────────────────────────────


def _collect_imports(
    entries: List[DeckRegistryEntry],
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Separate entries into (known-card imports, new-card-var-names).

    Returns:
        imports: ``{module_path: [VAR_NAME, ...]}``
        new_vars: list of snake_names for cards to define inline
    """
    imports: Dict[str, List[str]] = defaultdict(list)
    new_vars: List[str] = []

    for entry in entries:
        if entry.source_module is not None and entry.var_name is not None:
            imports[entry.source_module].append(entry.var_name)
        else:
            new_vars.append(entry.snake_name)

    return dict(imports), new_vars


def _detect_needed_helpers(entries: List[DeckRegistryEntry]) -> List[str]:
    """Return list of factory helper names needed for new card definitions."""
    helpers_needed = set()
    for entry in entries:
        if entry.source_module is not None:
            continue
        ctype = entry.card.type
        if ctype == "land":
            helpers_needed.add("land")
        elif ctype in ("creature", "artifact_creature"):
            helpers_needed.add("creature")
        elif ctype == "ritual":
            helpers_needed.add("ritual")
        elif ctype == "tutor":
            helpers_needed.add("tutor")
        elif ctype == "artifact":
            cost = entry.card.cost
            if cost is not None and cost.total == 0:
                helpers_needed.add("artifact_0")
            elif cost is not None and cost.total == 1:
                helpers_needed.add("artifact_1")
            else:
                helpers_needed.add("Card")
        else:
            helpers_needed.add("Card")

    # Always need these for the registry constant and cost
    helpers_needed.add("ManaRequirement")
    helpers_needed.add("build_registry")

    # any_color needed if any land uses it
    for entry in entries:
        if entry.source_module is not None:
            continue
        card = entry.card
        for attr in (card.t1_mana, card.t2_mana, card.produces):
            if attr is not None and _is_any_color(attr):
                helpers_needed.add("any_color")

    return sorted(helpers_needed)


def _is_any_color(mana: Mana) -> bool:
    return (
        mana.total > 0
        and mana.white == mana.total
        and mana.blue == mana.total
        and mana.black == mana.total
        and mana.red == mana.total
        and mana.green == mana.total
    )


# ── Main generator ─────────────────────────────────────────────────────────


def generate_registry_file(
    entries: List[DeckRegistryEntry],
    deck_name: str,
    commander_cost: Optional[ManaRequirement],
    output_path: str,
) -> None:
    """Write a card_registries/<deck_name>.py file.

    Args:
        entries: Resolved deck registry entries.
        deck_name: Snake_case deck identifier (e.g. ``"braids_reanimator"``).
        commander_cost: Parsed commander mana cost for the COST constant.
        output_path: Absolute path to write the generated file.
    """
    lines: List[str] = []
    upper = deck_name.upper()

    # Header
    lines.append(f'"""Auto-generated registry for {deck_name}. Review cards marked NEEDS_REVIEW."""')
    lines.append("")

    # Imports
    imports, new_vars = _collect_imports(entries)
    helpers = _detect_needed_helpers(entries)

    # cedh_mulligan_simulator.card_registry imports
    core_imports = [
        h
        for h in helpers
        if h
        in (
            "Card",
            "Mana",
            "ManaRequirement",
            "any_color",
            "build_registry",
            "artifact_0",
            "artifact_1",
            "creature",
            "equipment_1",
            "land",
            "ritual",
            "sacrifice_outlet",
            "tutor",
        )
    ]
    if core_imports:
        lines.append("from cedh_mulligan_simulator.card_registry import (")
        for name in sorted(core_imports):
            lines.append(f"    {name},")
        lines.append(")")
        lines.append("")

    # card_database imports (sorted by module for readability)
    for module_path in sorted(imports.keys()):
        var_names = sorted(imports[module_path])
        if not var_names:
            continue
        lines.append(f"from {module_path} import (")
        for vname in var_names:
            lines.append(f"    {vname},")
        lines.append(")")
        lines.append("")

    # Inline new card definitions
    new_entries = [e for e in entries if e.snake_name in new_vars]
    if new_entries:
        lines.append("")
        lines.append("# NEW CARDS (auto-inferred from Scryfall, verify simulation properties)")
        for entry in new_entries:
            lines.append(_render_new_card(entry))
        lines.append("")

    # Commander cost
    if commander_cost is not None:
        lines.append("")
        lines.append(f"{upper}_COST = {_mana_req_repr(commander_cost)}")
    else:
        lines.append("")
        lines.append(f"{upper}_COST = ManaRequirement()  # TODO: set commander cost")

    # Registry
    all_var_names: List[str] = []
    for entry in entries:
        if entry.var_name is not None:
            all_var_names.append(entry.var_name)
        else:
            all_var_names.append(entry.card.name.upper())

    lines.append(f"{upper}_REGISTRY = build_registry(")
    for var in all_var_names:
        lines.append(f"    {var},")
    lines.append(")")
    lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
