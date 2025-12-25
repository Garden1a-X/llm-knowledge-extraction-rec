#!/usr/bin/env python3
"""Test MovieLens 1M data loading."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import MovieLensLoader


def main():
    # Data directory (adjust if needed)
    data_dir = "/data/xuao/llm-knowledge-extraction-rec/data/raw/ml-1m"

    print("="*60)
    print("MovieLens 1M Data Loading Test")
    print("="*60)

    # Initialize loader
    loader = MovieLensLoader(data_dir)

    # Load all data
    print("\nLoading movies...")
    movies = loader.load_movies()
    print(f" Loaded {len(movies)} movies")
    print(f"  Sample: {movies.iloc[0]['title']}")

    print("\nLoading ratings...")
    ratings = loader.load_ratings()
    print(f" Loaded {len(ratings)} ratings")
    print(f"  Rating range: {ratings['rating'].min()}-{ratings['rating'].max()}")

    print("\nLoading users...")
    users = loader.load_users()
    print(f" Loaded {len(users)} users")

    print("\nChecking posters...")
    posters = loader.get_available_posters()
    print(f" Found {len(posters)} posters")

    # Get statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    stats = loader.get_stats()
    for key, value in stats.items():
        if key == 'sparsity':
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\n All tests passed!")


if __name__ == "__main__":
    main()
