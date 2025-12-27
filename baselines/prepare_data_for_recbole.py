#!/usr/bin/env python3
"""
Prepare MovieLens 1M data for RecBole.

RecBole expects data in the following format:
- dataset_name.inter: user-item interactions
- Optional: dataset_name.user, dataset_name.item for features

Format:
user_id:token  item_id:token  rating:float  timestamp:float
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import argparse
from src.data.loader import MovieLensLoader


def prepare_recbole_data(
    ml_data_dir: str,
    output_dir: str,
    dataset_name: str = 'ml-1m'
):
    """
    Convert MovieLens 1M to RecBole format.

    Args:
        ml_data_dir: Path to MovieLens 1M raw data
        output_dir: Output directory for RecBole data
        dataset_name: Dataset name (used as filename prefix)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MovieLens data from {ml_data_dir}...")
    loader = MovieLensLoader(ml_data_dir)

    # Load data
    ratings = loader.load_ratings()
    movies = loader.load_movies()
    users = loader.load_users()

    print(f"  Ratings: {len(ratings)}")
    print(f"  Users: {len(users)}")
    print(f"  Movies: {len(movies)}")

    # ========================================================================
    # 1. Create .inter file (interactions)
    # ========================================================================
    print(f"\nCreating {dataset_name}.inter file...")

    # RecBole format: user_id:token  item_id:token  rating:float  timestamp:float
    inter_df = ratings[['user_id', 'movie_id', 'rating', 'timestamp']].copy()
    inter_df.columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']

    inter_path = output_dir / f'{dataset_name}.inter'
    inter_df.to_csv(inter_path, sep='\t', index=False)
    print(f"  Saved to {inter_path}")
    print(f"  Shape: {inter_df.shape}")

    # ========================================================================
    # 2. Create .user file (user features) - Optional
    # ========================================================================
    print(f"\nCreating {dataset_name}.user file...")

    # RecBole format: user_id:token  gender:token  age:token  occupation:token
    user_df = users[['user_id', 'gender', 'age', 'occupation']].copy()
    user_df.columns = ['user_id:token', 'gender:token', 'age:token', 'occupation:token']

    user_path = output_dir / f'{dataset_name}.user'
    user_df.to_csv(user_path, sep='\t', index=False)
    print(f"  Saved to {user_path}")
    print(f"  Shape: {user_df.shape}")

    # ========================================================================
    # 3. Create .item file (item features) - Optional
    # ========================================================================
    print(f"\nCreating {dataset_name}.item file...")

    # RecBole format: item_id:token  title:token  genres:token_seq  year:float
    item_df = movies[['movie_id', 'title', 'genres', 'year']].copy()
    item_df.columns = ['item_id:token', 'title:token', 'genres:token_seq', 'year:float']

    item_path = output_dir / f'{dataset_name}.item'
    item_df.to_csv(item_path, sep='\t', index=False)
    print(f"  Saved to {item_path}")
    print(f"  Shape: {item_df.shape}")

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*60}")
    print("RecBole data preparation completed!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - {dataset_name}.inter (required)")
    print(f"  - {dataset_name}.user (optional)")
    print(f"  - {dataset_name}.item (optional)")
    print()
    print("Usage in RecBole:")
    print(f"  config_dict = {{'data_path': '{output_dir}'}}")
    print(f"  config_dict = {{'dataset': '{dataset_name}'}}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare MovieLens data for RecBole')
    parser.add_argument('--ml_data_dir', type=str, required=True,
                        help='Path to MovieLens 1M raw data directory')
    parser.add_argument('--output_dir', type=str, default='data/recbole/ml-1m',
                        help='Output directory for RecBole format data')
    parser.add_argument('--dataset_name', type=str, default='ml-1m',
                        help='Dataset name (used as filename prefix)')

    args = parser.parse_args()

    prepare_recbole_data(
        ml_data_dir=args.ml_data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )


if __name__ == '__main__':
    main()
