"""CLI entry point — import a deck list and generate a card registry.

Usage::

    python import_deck.py decks/braids_reanimator.txt \\
        --commander "Braids, Arisen Nightmare" \\
        [--output braids_reanimator] \\
        [--include-sideboard]
"""

import argparse
import os
import sys
from typing import List

from deck_importer.builder import build_deck_registry, resolve_commander_cost, scan_card_database
from deck_importer.generator import generate_registry_file
from deck_importer.parser import DeckList, parse_deck_list


def _print_summary(
    entries_total: int,
    db_count: int,
    inferred_count: int,
    review_count: int,
    review_lines: List[str],
    output_path: str,
) -> None:
    print(f"\n{'=' * 60}")
    print("Deck Import Summary")
    print(f"{'=' * 60}")
    print(f"  Total cards resolved : {entries_total}")
    print(f"  From card_database   : {db_count}")
    print(f"  Auto-inferred        : {inferred_count}")
    print(f"  Need manual review   : {review_count}")
    if review_lines:
        print("\nCards requiring manual review:")
        for line in review_lines:
            print(f"  {line}")
    print(f"\nGenerated: {output_path}")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a Moxfield deck list and generate a simulation card registry.")
    parser.add_argument("deck_file", type=str, help="Path to plain-text deck list file")
    parser.add_argument(
        "--commander", type=str, default=None, help='Commander display name, e.g. "Braids, Arisen Nightmare"'
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file stem (default: deck file stem). Written to card_registries/<output>.py",
    )
    parser.add_argument(
        "--include-sideboard",
        action="store_true",
        help="Also include sideboard cards in the generated registry",
    )

    args = parser.parse_args()

    deck_file: str = args.deck_file
    commander_arg: str = args.commander if args.commander is not None else ""
    output_stem: str = args.output if args.output is not None else ""
    include_sideboard: bool = args.include_sideboard

    # Determine output stem from deck file name if not provided
    if not output_stem:
        output_stem = os.path.splitext(os.path.basename(deck_file))[0]

    # Read deck file
    if not os.path.exists(deck_file):
        print(f"Error: deck file not found: {deck_file}", file=sys.stderr)
        sys.exit(1)

    with open(deck_file, encoding="utf-8") as fh:
        deck_text = fh.read()

    # Parse deck list
    deck_list: DeckList = parse_deck_list(deck_text, commander=commander_arg if commander_arg else None)

    if include_sideboard:
        deck_list.mainboard.extend(deck_list.sideboard)
        deck_list.sideboard = []

    print(f"Parsed {len(deck_list.mainboard)} mainboard entries, {len(deck_list.sideboard)} sideboard entries.")

    # Resolve commander cost
    commander_cost = None
    if deck_list.commander:
        db_lookup = scan_card_database()
        commander_cost = resolve_commander_cost(deck_list.commander, db_lookup)
        print(f"Commander cost: {commander_cost}")

    # Build registry entries
    print("Resolving cards (Scryfall lookups may take a moment)...")
    entries = build_deck_registry(deck_list)

    # Collect stats
    db_count = sum(1 for e in entries if e.source_module is not None)
    inferred_count = sum(1 for e in entries if e.source_module is None and e.card.type != "unknown")
    review_count = sum(1 for e in entries if e.needs_review)
    review_lines = [f"{e.snake_name}: {e.review_reason}" for e in entries if e.needs_review]

    # Generate file
    output_path = os.path.join("card_registries", f"{output_stem}.py")
    generate_registry_file(entries, output_stem, commander_cost, output_path)

    _print_summary(
        entries_total=len(entries),
        db_count=db_count,
        inferred_count=inferred_count,
        review_count=review_count,
        review_lines=review_lines,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
