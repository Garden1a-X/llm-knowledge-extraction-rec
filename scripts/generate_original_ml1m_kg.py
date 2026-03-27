#!/usr/bin/env python3
"""
Generate original ML-1M knowledge graph from item metadata.
Uses title, genres, and year from ml-1m.item file.
"""

import argparse
from pathlib import Path
from typing import List, Tuple


def parse_item_file(item_file: Path) -> List[Tuple[str, str, List[str], str]]:
    """
    Parse ml-1m.item file.

    Returns:
        List of (item_id, title, genres_list, year)
    """
    items = []

    with open(item_file, 'r', encoding='utf-8') as f:
        # Skip header
        header = f.readline()

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 4:
                print(f"Warning: Skipping malformed line: {line}")
                continue

            item_id = parts[0]
            title = parts[1]
            genres_str = parts[2]
            year = parts[3]

            # Parse genres (may be pipe-separated)
            genres = genres_str.split('|') if genres_str else []

            items.append((item_id, title, genres, year))

    return items


def generate_kg_triples(items: List[Tuple[str, str, List[str], str]]) -> List[Tuple[str, str, str]]:
    """
    Generate KG triples from item metadata.

    Relations:
    - has_genre: item -> genre
    - released_in_year: item -> year

    Returns:
        List of (head_id, relation_id, tail_id) triples
    """
    triples = []

    for item_id, title, genres, year in items:
        # Add genre relations
        for genre in genres:
            # Normalize genre to lowercase and replace spaces/apostrophes
            genre_normalized = genre.lower().replace("'", "").replace(" ", "_")
            triples.append((item_id, "has_genre", genre_normalized))

        # Add year relation
        if year and year != '':
            # Convert year to string (may be float like 1995.0)
            year_str = str(int(float(year)))
            triples.append((item_id, "released_in_year", year_str))

    return triples


def save_kg(triples: List[Tuple[str, str, str]], output_file: Path):
    """
    Save KG triples to file in RecBole format.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")

        # Write triples
        for head, relation, tail in triples:
            f.write(f"{head}\t{relation}\t{tail}\n")

    print(f"Saved {len(triples)} triples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate original ML-1M KG from item metadata'
    )
    parser.add_argument('--item_file', type=str, required=True,
                       help='Path to ml-1m.item file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output KG file (e.g., ml-1m.kg)')

    args = parser.parse_args()

    item_file = Path(args.item_file)
    output_file = Path(args.output)

    if not item_file.exists():
        print(f"Error: Item file not found: {item_file}")
        return

    print("="*70)
    print("Generating Original ML-1M Knowledge Graph")
    print("="*70)
    print(f"\nInput: {item_file}")
    print(f"Output: {output_file}")
    print()

    # Parse item file
    print("Parsing item file...")
    items = parse_item_file(item_file)
    print(f"  ✓ Parsed {len(items)} items")

    # Generate KG triples
    print("\nGenerating KG triples...")
    triples = generate_kg_triples(items)
    print(f"  ✓ Generated {len(triples)} triples")

    # Statistics
    num_genre_triples = sum(1 for _, r, _ in triples if r == "has_genre")
    num_year_triples = sum(1 for _, r, _ in triples if r == "released_in_year")

    print(f"\nStatistics:")
    print(f"  Genre triples: {num_genre_triples}")
    print(f"  Year triples: {num_year_triples}")
    print(f"  Total triples: {len(triples)}")

    # Get unique entities
    unique_genres = set(tail for _, r, tail in triples if r == "has_genre")
    unique_years = set(tail for _, r, tail in triples if r == "released_in_year")

    print(f"\nUnique entities:")
    print(f"  Genres: {len(unique_genres)}")
    print(f"  Years: {len(unique_years)}")
    print(f"  Example genres: {sorted(list(unique_genres))[:5]}")
    print(f"  Year range: {min(unique_years)} - {max(unique_years)}")

    # Save KG
    print(f"\nSaving KG...")
    save_kg(triples, output_file)

    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
