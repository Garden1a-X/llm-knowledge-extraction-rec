"""
Data splitter for train/validation/test split.

Supports temporal split (by timestamp) and random split.
"""

from typing import Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import pickle


class DataSplitter:
    """Split ratings data into train/val/test sets."""

    def __init__(
        self,
        ratings: pd.DataFrame,
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        split_method: str = "temporal",
        min_train_items: int = 5,
        random_seed: int = 42
    ):
        """
        Initialize data splitter.

        Args:
            ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]
            split_ratios: (train, val, test) ratios, should sum to 1.0
            split_method: "temporal" or "random"
            min_train_items: Minimum number of items per user in training set
            random_seed: Random seed for reproducibility
        """
        self.ratings = ratings.copy()
        self.split_ratios = split_ratios
        self.split_method = split_method
        self.min_train_items = min_train_items
        self.random_seed = random_seed

        # Validate split ratios
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

        # Validate columns
        required_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        assert all(col in ratings.columns for col in required_cols), \
            f"Missing required columns. Need: {required_cols}"

        # Split data
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def split(self) -> Dict[str, pd.DataFrame]:
        """
        Perform data split.

        Returns:
            Dictionary with keys 'train', 'val', 'test' containing DataFrames
        """
        if self.split_method == "temporal":
            return self._temporal_split()
        elif self.split_method == "random":
            return self._random_split()
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")

    def _temporal_split(self) -> Dict[str, pd.DataFrame]:
        """
        Split by timestamp (chronological).

        For each user, sort their ratings by timestamp and split.
        This simulates predicting future behavior.
        """
        np.random.seed(self.random_seed)

        train_list = []
        val_list = []
        test_list = []

        # Group by user
        grouped = self.ratings.groupby('user_id')

        for user_id, user_ratings in grouped:
            # Sort by timestamp
            user_ratings = user_ratings.sort_values('timestamp')
            n_items = len(user_ratings)

            # Calculate split points
            train_end = int(n_items * self.split_ratios[0])
            val_end = int(n_items * (self.split_ratios[0] + self.split_ratios[1]))

            # Ensure minimum training items
            if train_end < self.min_train_items:
                train_end = min(self.min_train_items, n_items - 2)
                val_end = min(train_end + 1, n_items - 1)

            # Split
            train = user_ratings.iloc[:train_end]
            val = user_ratings.iloc[train_end:val_end]
            test = user_ratings.iloc[val_end:]

            train_list.append(train)
            if len(val) > 0:
                val_list.append(val)
            if len(test) > 0:
                test_list.append(test)

        # Concatenate
        self.train_df = pd.concat(train_list, ignore_index=True)
        self.val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
        self.test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

        return {
            'train': self.train_df,
            'val': self.val_df,
            'test': self.test_df
        }

    def _random_split(self) -> Dict[str, pd.DataFrame]:
        """
        Random split while ensuring each user has minimum items in training.
        """
        np.random.seed(self.random_seed)

        train_list = []
        val_list = []
        test_list = []

        # Group by user
        grouped = self.ratings.groupby('user_id')

        for user_id, user_ratings in grouped:
            # Shuffle
            user_ratings = user_ratings.sample(frac=1, random_state=self.random_seed)
            n_items = len(user_ratings)

            # Calculate split points
            train_end = int(n_items * self.split_ratios[0])
            val_end = int(n_items * (self.split_ratios[0] + self.split_ratios[1]))

            # Ensure minimum training items
            if train_end < self.min_train_items:
                train_end = min(self.min_train_items, n_items - 2)
                val_end = min(train_end + 1, n_items - 1)

            # Split
            train = user_ratings.iloc[:train_end]
            val = user_ratings.iloc[train_end:val_end]
            test = user_ratings.iloc[val_end:]

            train_list.append(train)
            if len(val) > 0:
                val_list.append(val)
            if len(test) > 0:
                test_list.append(test)

        # Concatenate
        self.train_df = pd.concat(train_list, ignore_index=True)
        self.val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
        self.test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

        return {
            'train': self.train_df,
            'val': self.val_df,
            'test': self.test_df
        }

    def get_statistics(self) -> Dict:
        """Get statistics about the split."""
        if self.train_df is None:
            raise ValueError("Must call split() first")

        stats = {
            'total_ratings': len(self.ratings),
            'train_ratings': len(self.train_df),
            'val_ratings': len(self.val_df),
            'test_ratings': len(self.test_df),
            'train_ratio': len(self.train_df) / len(self.ratings),
            'val_ratio': len(self.val_df) / len(self.ratings),
            'test_ratio': len(self.test_df) / len(self.ratings),
            'n_users': self.ratings['user_id'].nunique(),
            'n_items': self.ratings['movie_id'].nunique(),
            'train_users': self.train_df['user_id'].nunique(),
            'train_items': self.train_df['movie_id'].nunique(),
            'test_users': self.test_df['user_id'].nunique(),
            'test_items': self.test_df['movie_id'].nunique(),
            'split_method': self.split_method,
        }

        return stats

    def save(self, output_dir: Path):
        """
        Save split data to disk.

        Args:
            output_dir: Directory to save split data
        """
        if self.train_df is None:
            raise ValueError("Must call split() first")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save DataFrames
        self.train_df.to_csv(output_dir / 'train.csv', index=False)
        self.val_df.to_csv(output_dir / 'val.csv', index=False)
        self.test_df.to_csv(output_dir / 'test.csv', index=False)

        # Save statistics
        stats = self.get_statistics()
        with open(output_dir / 'split_stats.pkl', 'wb') as f:
            pickle.dump(stats, f)

        print(f"Split data saved to {output_dir}")
        print(f"  - train.csv: {len(self.train_df)} ratings")
        print(f"  - val.csv: {len(self.val_df)} ratings")
        print(f"  - test.csv: {len(self.test_df)} ratings")

    @staticmethod
    def load(input_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load split data from disk.

        Args:
            input_dir: Directory containing split data

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        input_dir = Path(input_dir)

        return {
            'train': pd.read_csv(input_dir / 'train.csv'),
            'val': pd.read_csv(input_dir / 'val.csv'),
            'test': pd.read_csv(input_dir / 'test.csv')
        }


def filter_positive_ratings(
    ratings: pd.DataFrame,
    threshold: float = 4.0
) -> pd.DataFrame:
    """
    Filter ratings to keep only positive ones (rating >= threshold).

    This is commonly used for implicit feedback scenarios.

    Args:
        ratings: DataFrame with rating column
        threshold: Minimum rating to keep

    Returns:
        Filtered DataFrame
    """
    return ratings[ratings['rating'] >= threshold].copy()


def create_user_item_mapping(
    ratings: pd.DataFrame
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Create mapping between original IDs and consecutive indices.

    This is needed for PyTorch/embedding layers which require consecutive indices.

    Args:
        ratings: DataFrame with user_id and movie_id columns

    Returns:
        Tuple of (user2idx, idx2user, item2idx, idx2item) dictionaries
    """
    # Get unique IDs
    unique_users = sorted(ratings['user_id'].unique())
    unique_items = sorted(ratings['movie_id'].unique())

    # Create mappings
    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    idx2user = {idx: uid for uid, idx in user2idx.items()}

    item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
    idx2item = {idx: iid for iid, idx in item2idx.items()}

    return user2idx, idx2user, item2idx, idx2item
