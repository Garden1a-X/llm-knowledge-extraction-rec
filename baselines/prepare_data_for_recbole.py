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
import json
from src.data.loader import MovieLensLoader


def apply_k_core_filtering(ratings: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Apply k-core filtering to the ratings data.

    Iteratively removes users and items with fewer than k interactions
    until all remaining users and items have at least k interactions.

    Args:
        ratings: DataFrame with user_id, movie_id, rating, timestamp
        k: Minimum number of interactions (default: 5)

    Returns:
        Filtered ratings DataFrame
    """
    print(f"\nApplying {k}-core filtering...")
    print(f"  Before filtering: {len(ratings)} ratings, "
          f"{ratings['user_id'].nunique()} users, "
          f"{ratings['movie_id'].nunique()} items")

    iteration = 0
    while True:
        iteration += 1
        prev_size = len(ratings)

        # Count interactions per user and item
        user_counts = ratings['user_id'].value_counts()
        item_counts = ratings['movie_id'].value_counts()

        # Keep users and items with at least k interactions
        valid_users = user_counts[user_counts >= k].index
        valid_items = item_counts[item_counts >= k].index

        # Filter ratings
        ratings = ratings[
            ratings['user_id'].isin(valid_users) &
            ratings['movie_id'].isin(valid_items)
        ]

        # Check convergence
        if len(ratings) == prev_size:
            print(f"  Converged after {iteration} iterations")
            break

        print(f"  Iteration {iteration}: {len(ratings)} ratings remaining")

    print(f"  After filtering: {len(ratings)} ratings, "
          f"{ratings['user_id'].nunique()} users, "
          f"{ratings['movie_id'].nunique()} items")
    print(f"  Removed: {prev_size - len(ratings)} ratings "
          f"({100 * (prev_size - len(ratings)) / prev_size:.2f}%)")

    return ratings


def create_id_mappings(users: pd.DataFrame, movies: pd.DataFrame):
    """
    Create continuous ID mappings for users and items.

    Maps original IDs to continuous integers starting from 1.
    (RecBole uses 1-indexed IDs internally, 0 is reserved for padding)

    Args:
        users: DataFrame with user_id column
        movies: DataFrame with movie_id column

    Returns:
        Tuple of (user_id_map, item_id_map) where each map is a dict:
        {
            'original_to_new': {original_id: new_id, ...},
            'new_to_original': {new_id: original_id, ...}
        }
    """
    # Get unique IDs and sort them for consistency
    unique_user_ids = sorted(users['user_id'].unique())
    unique_movie_ids = sorted(movies['movie_id'].unique())

    # Create mappings (start from 1)
    user_id_map = {
        'original_to_new': {old_id: new_id for new_id, old_id in enumerate(unique_user_ids, start=1)},
        'new_to_original': {new_id: old_id for new_id, old_id in enumerate(unique_user_ids, start=1)}
    }

    item_id_map = {
        'original_to_new': {old_id: new_id for new_id, old_id in enumerate(unique_movie_ids, start=1)},
        'new_to_original': {new_id: old_id for new_id, old_id in enumerate(unique_movie_ids, start=1)}
    }

    print(f"\n  ID Mappings created:")
    print(f"    Users: {len(unique_user_ids)} IDs mapped to [1, {len(unique_user_ids)}]")
    print(f"    Items: {len(unique_movie_ids)} IDs mapped to [1, {len(unique_movie_ids)}]")

    return user_id_map, item_id_map


def prepare_recbole_data(
    ml_data_dir: str,
    output_dir: str,
    dataset_name: str = 'ml-1m',
    min_interactions: int = 0
):
    """
    Convert MovieLens 1M to RecBole format with continuous ID mapping.

    Args:
        ml_data_dir: Path to MovieLens 1M raw data
        output_dir: Output directory for RecBole data
        dataset_name: Dataset name (used as filename prefix)
        min_interactions: Minimum interactions for k-core filtering (0 = no filtering)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MovieLens data from {ml_data_dir}...")
    loader = MovieLensLoader(ml_data_dir)

    # Load data
    ratings = loader.load_ratings()
    movies = loader.load_movies()
    users = loader.load_users()

    print(f"  Original data:")
    print(f"    Ratings: {len(ratings)}")
    print(f"    Users: {len(users)}")
    print(f"    Movies: {len(movies)}")

    # Apply k-core filtering if requested
    if min_interactions > 0:
        ratings = apply_k_core_filtering(ratings, k=min_interactions)

        # Update users and movies to only include those in filtered ratings
        valid_users = ratings['user_id'].unique()
        valid_movies = ratings['movie_id'].unique()

        users = users[users['user_id'].isin(valid_users)]
        movies = movies[movies['movie_id'].isin(valid_movies)]

        print(f"\n  After filtering:")
        print(f"    Ratings: {len(ratings)}")
        print(f"    Users: {len(users)}")
        print(f"    Movies: {len(movies)}")

    # ========================================================================
    # Create continuous ID mappings
    # ========================================================================
    user_id_map, item_id_map = create_id_mappings(users, movies)

    # Apply mappings to dataframes
    ratings['user_id'] = ratings['user_id'].map(user_id_map['original_to_new'])
    ratings['movie_id'] = ratings['movie_id'].map(item_id_map['original_to_new'])
    users['user_id'] = users['user_id'].map(user_id_map['original_to_new'])
    movies['movie_id'] = movies['movie_id'].map(item_id_map['original_to_new'])

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
    # 4. Save ID mappings
    # ========================================================================
    print(f"\nSaving ID mappings...")

    mapping_data = {
        'user_id_map': user_id_map,
        'item_id_map': item_id_map,
        'stats': {
            'num_users': len(user_id_map['new_to_original']),
            'num_items': len(item_id_map['new_to_original']),
            'num_ratings': len(ratings),
            'min_interactions': min_interactions
        }
    }

    mapping_path = output_dir / 'id_mappings.json'
    with open(mapping_path, 'w') as f:
        # Convert int keys to strings for JSON serialization
        serializable_data = {
            'user_id_map': {
                'original_to_new': {str(k): v for k, v in user_id_map['original_to_new'].items()},
                'new_to_original': {str(k): v for k, v in user_id_map['new_to_original'].items()}
            },
            'item_id_map': {
                'original_to_new': {str(k): v for k, v in item_id_map['original_to_new'].items()},
                'new_to_original': {str(k): v for k, v in item_id_map['new_to_original'].items()}
            },
            'stats': mapping_data['stats']
        }
        json.dump(serializable_data, f, indent=2)

    print(f"  Saved to {mapping_path}")

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
    print(f"  - id_mappings.json (ID mapping for posters/features)")
    print()
    print("ID Mapping:")
    print(f"  - User IDs: {len(user_id_map['new_to_original'])} users mapped to [1, {len(user_id_map['new_to_original'])}]")
    print(f"  - Item IDs: {len(item_id_map['new_to_original'])} items mapped to [1, {len(item_id_map['new_to_original'])}]")
    print(f"  - Use 'new_to_original' to map RecBole IDs back to original movie IDs for posters")
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
    parser.add_argument('--min_interactions', type=int, default=5,
                        help='Minimum interactions for k-core filtering (default: 5, 0 = no filtering)')

    args = parser.parse_args()

    prepare_recbole_data(
        ml_data_dir=args.ml_data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        min_interactions=args.min_interactions
    )


if __name__ == '__main__':
    main()
