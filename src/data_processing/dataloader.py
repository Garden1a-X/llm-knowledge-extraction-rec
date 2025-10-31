"""
Data Loader for Amazon VideoGames Dataset
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger


class AmazonDataLoader:
    """Loads and preprocesses Amazon VideoGames dataset."""

    def __init__(self, data_path: str, demo_mode: bool = True):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the raw data directory
            demo_mode: If True, load only a subset of data for testing
        """
        self.data_path = Path(data_path)
        self.demo_mode = demo_mode

    def load_products(self, num_products: Optional[int] = None) -> pd.DataFrame:
        """
        Load product metadata.

        Args:
            num_products: Number of products to load (None for all)

        Returns:
            DataFrame with product information
        """
        logger.info(f"Loading products from {self.data_path}")
        # TODO: Implement actual data loading logic
        # Expected columns: product_id, title, description, category, price, etc.
        raise NotImplementedError("Product loading not yet implemented")

    def load_interactions(self, num_users: Optional[int] = None) -> pd.DataFrame:
        """
        Load user-item interactions.

        Args:
            num_users: Number of users to load (None for all)

        Returns:
            DataFrame with interaction data
        """
        logger.info(f"Loading interactions from {self.data_path}")
        # TODO: Implement actual data loading logic
        # Expected columns: user_id, product_id, rating, timestamp, review_text, etc.
        raise NotImplementedError("Interaction loading not yet implemented")

    def load_demo_data(self, num_users: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load demo dataset with limited users and their interactions.

        Args:
            num_users: Number of users to include in demo

        Returns:
            Tuple of (products_df, interactions_df)
        """
        logger.info(f"Loading demo data with {num_users} users")

        # Load interactions for selected users
        interactions_df = self.load_interactions(num_users=num_users)

        # Get unique product IDs from interactions
        product_ids = interactions_df['product_id'].unique()

        # Load only the products that appear in interactions
        products_df = self.load_products()
        products_df = products_df[products_df['product_id'].isin(product_ids)]

        logger.info(
            f"Demo data loaded: {len(products_df)} products, "
            f"{len(interactions_df)} interactions"
        )

        return products_df, interactions_df

    def preprocess_products(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess product data.

        Args:
            products_df: Raw product DataFrame

        Returns:
            Preprocessed DataFrame
        """
        # TODO: Implement preprocessing logic
        # - Clean text fields
        # - Handle missing values
        # - Normalize fields
        return products_df

    def preprocess_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess interaction data.

        Args:
            interactions_df: Raw interaction DataFrame

        Returns:
            Preprocessed DataFrame with interaction types
        """
        # TODO: Implement preprocessing logic
        # - Convert ratings to interaction types (positive/negative)
        # - Filter low-quality interactions
        # - Sort by timestamp
        return interactions_df

    def split_data(
        self,
        interactions_df: pd.DataFrame,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Split interaction data into train/val/test sets.

        Args:
            interactions_df: Interaction DataFrame
            test_ratio: Ratio of test set
            val_ratio: Ratio of validation set

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        # TODO: Implement temporal split or random split
        raise NotImplementedError("Data splitting not yet implemented")
