"""
Data loader for MovieLens 1M dataset.

This module handles loading and preprocessing of:
- Movie metadata (movies.dat)
- User ratings (ratings.dat)
- User information (users.dat)
- Movie posters (posters/*.jpg)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class MovieLensDataLoader:
    """Data loader for MovieLens 1M dataset."""

    def __init__(self, data_dir: str):
        """
        Initialize data loader.

        Args:
            data_dir: Path to the ml-1m directory containing the dataset files
        """
        self.data_dir = Path(data_dir)
        self.movies_file = self.data_dir / "movies.dat"
        self.ratings_file = self.data_dir / "ratings.dat"
        self.users_file = self.data_dir / "users.dat"
        self.posters_dir = self.data_dir / "posters"

        # Validate paths
        self._validate_paths()

        # Cached data
        self._movies = None
        self._ratings = None
        self._users = None

    def _validate_paths(self):
        """Validate that all required data files exist."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        required_files = [self.movies_file, self.ratings_file, self.users_file]
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

        if not self.posters_dir.exists():
            logger.warning(f"Posters directory not found: {self.posters_dir}")

    def load_movies(self, reload: bool = False) -> pd.DataFrame:
        """
        Load movie metadata.

        Format: MovieID::Title::Genres

        Args:
            reload: Force reload from file even if cached

        Returns:
            DataFrame with columns: movie_id, title, genres, year
        """
        if self._movies is not None and not reload:
            return self._movies

        logger.info(f"Loading movies from {self.movies_file}")

        # Read with '::' separator
        movies = pd.read_csv(
            self.movies_file,
            sep='::',
            engine='python',
            header=None,
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1'
        )

        # Extract year from title (format: "Title (YEAR)")
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('Int64')

        # Split genres into list
        movies['genres_list'] = movies['genres'].str.split('|')

        logger.info(f"Loaded {len(movies)} movies")
        self._movies = movies

        return movies

    def load_ratings(self, reload: bool = False) -> pd.DataFrame:
        """
        Load user ratings.

        Format: UserID::MovieID::Rating::Timestamp

        Args:
            reload: Force reload from file even if cached

        Returns:
            DataFrame with columns: user_id, movie_id, rating, timestamp
        """
        if self._ratings is not None and not reload:
            return self._ratings

        logger.info(f"Loading ratings from {self.ratings_file}")

        # Read with '::' separator
        ratings = pd.read_csv(
            self.ratings_file,
            sep='::',
            engine='python',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )

        # Convert timestamp to datetime
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')

        logger.info(f"Loaded {len(ratings)} ratings")
        self._ratings = ratings

        return ratings

    def load_users(self, reload: bool = False) -> pd.DataFrame:
        """
        Load user information.

        Format: UserID::Gender::Age::Occupation::Zip-code

        Args:
            reload: Force reload from file even if cached

        Returns:
            DataFrame with columns: user_id, gender, age, occupation, zipcode
        """
        if self._users is not None and not reload:
            return self._users

        logger.info(f"Loading users from {self.users_file}")

        # Read with '::' separator
        users = pd.read_csv(
            self.users_file,
            sep='::',
            engine='python',
            header=None,
            names=['user_id', 'gender', 'age', 'occupation', 'zipcode']
        )

        # Age and occupation mapping
        age_map = {
            1: "Under 18",
            18: "18-24",
            25: "25-34",
            35: "35-44",
            45: "45-49",
            50: "50-55",
            56: "56+"
        }

        occupation_map = {
            0: "other",
            1: "academic/educator",
            2: "artist",
            3: "clerical/admin",
            4: "college/grad student",
            5: "customer service",
            6: "doctor/health care",
            7: "executive/managerial",
            8: "farmer",
            9: "homemaker",
            10: "K-12 student",
            11: "lawyer",
            12: "programmer",
            13: "retired",
            14: "sales/marketing",
            15: "scientist",
            16: "self-employed",
            17: "technician/engineer",
            18: "tradesman/craftsman",
            19: "unemployed",
            20: "writer"
        }

        users['age_group'] = users['age'].map(age_map)
        users['occupation_name'] = users['occupation'].map(occupation_map)

        logger.info(f"Loaded {len(users)} users")
        self._users = users

        return users

    def load_poster(self, movie_id: int) -> Optional[Image.Image]:
        """
        Load poster image for a specific movie.

        Args:
            movie_id: Movie ID

        Returns:
            PIL Image object, or None if poster not found
        """
        poster_path = self.posters_dir / f"{movie_id}.jpg"

        if not poster_path.exists():
            logger.debug(f"Poster not found for movie {movie_id}")
            return None

        try:
            image = Image.open(poster_path)
            return image
        except Exception as e:
            logger.error(f"Error loading poster for movie {movie_id}: {e}")
            return None

    def get_available_posters(self) -> List[int]:
        """
        Get list of movie IDs that have posters available.

        Returns:
            List of movie IDs with available posters
        """
        if not self.posters_dir.exists():
            return []

        poster_files = list(self.posters_dir.glob("*.jpg"))
        movie_ids = [int(p.stem) for p in poster_files]

        return sorted(movie_ids)

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data (movies, ratings, users).

        Returns:
            Tuple of (movies, ratings, users) DataFrames
        """
        movies = self.load_movies()
        ratings = self.load_ratings()
        users = self.load_users()

        return movies, ratings, users

    def get_dataset_stats(self) -> dict:
        """
        Get basic statistics about the dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        movies, ratings, users = self.load_all()

        stats = {
            'num_movies': len(movies),
            'num_users': len(users),
            'num_ratings': len(ratings),
            'num_posters': len(self.get_available_posters()),
            'avg_ratings_per_user': len(ratings) / len(users),
            'avg_ratings_per_movie': len(ratings) / len(movies),
            'rating_sparsity': 1 - (len(ratings) / (len(users) * len(movies))),
            'rating_distribution': ratings['rating'].value_counts().to_dict(),
            'year_range': (movies['year'].min(), movies['year'].max()),
        }

        return stats

    def filter_movies_with_posters(self, movies: pd.DataFrame = None) -> pd.DataFrame:
        """
        Filter movies to only those with available posters.

        Args:
            movies: Movies DataFrame (if None, loads from file)

        Returns:
            Filtered movies DataFrame
        """
        if movies is None:
            movies = self.load_movies()

        available_poster_ids = set(self.get_available_posters())
        filtered = movies[movies['movie_id'].isin(available_poster_ids)].copy()

        logger.info(f"Filtered to {len(filtered)} movies with posters (from {len(movies)})")

        return filtered


def create_train_test_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test split for ratings.

    Uses temporal split: oldest ratings for train, newest for test.

    Args:
        ratings: Ratings DataFrame
        test_ratio: Proportion of data for test set
        val_ratio: Proportion of data for validation set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train, val, test) DataFrames
    """
    # Sort by timestamp
    ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)

    n = len(ratings_sorted)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - test_size - val_size

    train = ratings_sorted[:train_size]
    val = ratings_sorted[train_size:train_size + val_size]
    test = ratings_sorted[train_size + val_size:]

    logger.info(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    return train, val, test
