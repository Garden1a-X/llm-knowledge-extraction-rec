"""
MovieLens 1M Data Loader

Loads movies, ratings, users, and poster images.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image


class MovieLensLoader:
    """Data loader for MovieLens 1M dataset."""

    def __init__(self, data_dir: str):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to ml-1m directory
        """
        self.data_dir = Path(data_dir)
        self._movies = None
        self._ratings = None
        self._users = None

    def load_movies(self) -> pd.DataFrame:
        """
        Load movie metadata.

        Returns:
            DataFrame with columns: movie_id, title, genres, year, genres_list
        """
        if self._movies is not None:
            return self._movies

        movies = pd.read_csv(
            self.data_dir / "movies.dat",
            sep='::',
            engine='python',
            header=None,
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1'
        )

        # Extract year from title
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('Int64')
        movies['genres_list'] = movies['genres'].str.split('|')

        self._movies = movies
        return movies

    def load_ratings(self) -> pd.DataFrame:
        """
        Load user ratings.

        Returns:
            DataFrame with columns: user_id, movie_id, rating, timestamp
        """
        if self._ratings is not None:
            return self._ratings

        ratings = pd.read_csv(
            self.data_dir / "ratings.dat",
            sep='::',
            engine='python',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )

        self._ratings = ratings
        return ratings

    def load_users(self) -> pd.DataFrame:
        """
        Load user demographics.

        Returns:
            DataFrame with columns: user_id, gender, age, occupation, zipcode
        """
        if self._users is not None:
            return self._users

        users = pd.read_csv(
            self.data_dir / "users.dat",
            sep='::',
            engine='python',
            header=None,
            names=['user_id', 'gender', 'age', 'occupation', 'zipcode']
        )

        self._users = users
        return users

    def load_poster(self, movie_id: int) -> Optional[Image.Image]:
        """
        Load movie poster image.

        Args:
            movie_id: Movie ID

        Returns:
            PIL Image or None if poster not found
        """
        poster_path = self.data_dir / "posters" / f"{movie_id}.jpg"

        if not poster_path.exists():
            return None

        try:
            return Image.open(poster_path)
        except Exception:
            return None

    def get_available_posters(self) -> List[int]:
        """Get list of movie IDs with available posters."""
        posters_dir = self.data_dir / "posters"
        if not posters_dir.exists():
            return []

        poster_files = list(posters_dir.glob("*.jpg"))
        return sorted([int(p.stem) for p in poster_files])

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        movies = self.load_movies()
        ratings = self.load_ratings()
        users = self.load_users()
        posters = self.get_available_posters()

        return {
            'num_movies': len(movies),
            'num_users': len(users),
            'num_ratings': len(ratings),
            'num_posters': len(posters),
            'sparsity': 1 - len(ratings) / (len(users) * len(movies)),
        }
