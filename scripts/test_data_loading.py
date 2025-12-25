#!/usr/bin/env python3
"""
Test script for data loading.

This script tests the MovieLensDataLoader and prints dataset statistics.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.data_loader import MovieLensDataLoader
from src.utils.config import get_config
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main test function."""
    logger.info("Testing MovieLens 1M data loading...")

    # Load config
    try:
        config = get_config()
        data_dir = config.get('data.raw_dir')
        logger.info(f"Using data directory: {data_dir}")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        logger.info("Using default data directory...")
        data_dir = "/data/xuao/llm-knowledge-extraction-rec/data/raw/ml-1m"

    # Initialize data loader
    try:
        loader = MovieLensDataLoader(data_dir)
        logger.info(" Data loader initialized successfully")
    except Exception as e:
        logger.error(f" Failed to initialize data loader: {e}")
        return

    # Load movies
    try:
        movies = loader.load_movies()
        logger.info(f" Loaded {len(movies)} movies")
        logger.info(f"  Sample movie: {movies.iloc[0]['title']}")
    except Exception as e:
        logger.error(f" Failed to load movies: {e}")
        return

    # Load ratings
    try:
        ratings = loader.load_ratings()
        logger.info(f" Loaded {len(ratings)} ratings")
        logger.info(f"  Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
    except Exception as e:
        logger.error(f" Failed to load ratings: {e}")
        return

    # Load users
    try:
        users = loader.load_users()
        logger.info(f" Loaded {len(users)} users")
        logger.info(f"  Gender distribution: {users['gender'].value_counts().to_dict()}")
    except Exception as e:
        logger.error(f" Failed to load users: {e}")
        return

    # Check posters
    try:
        poster_ids = loader.get_available_posters()
        logger.info(f" Found {len(poster_ids)} posters")

        if len(poster_ids) > 0:
            # Try loading one poster
            sample_id = poster_ids[0]
            poster = loader.load_poster(sample_id)
            if poster:
                logger.info(f"  Sample poster (ID {sample_id}): {poster.size}")
    except Exception as e:
        logger.warning(f"! Error checking posters: {e}")

    # Get dataset statistics
    try:
        logger.info("\n" + "="*60)
        logger.info("DATASET STATISTICS")
        logger.info("="*60)

        stats = loader.get_dataset_stats()
        for key, value in stats.items():
            if key == 'rating_distribution':
                logger.info(f"{key}:")
                for rating, count in sorted(value.items()):
                    logger.info(f"  {rating} stars: {count}")
            elif key == 'year_range':
                logger.info(f"{key}: {value[0]} - {value[1]}")
            elif key == 'rating_sparsity':
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")

        logger.info("="*60)
    except Exception as e:
        logger.error(f" Failed to get statistics: {e}")

    # Filter movies with posters
    try:
        movies_with_posters = loader.filter_movies_with_posters()
        logger.info(f"\n Filtered to {len(movies_with_posters)} movies with posters")
    except Exception as e:
        logger.warning(f"! Failed to filter movies with posters: {e}")

    logger.info("\n All tests completed successfully!")


if __name__ == "__main__":
    main()
