"""
Graph builder for recommendation models.

Builds user-item bipartite graph for LightGCN and other GNN models.
"""

from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict


class BipartiteGraphBuilder:
    """Build user-item bipartite graph for recommendation."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        user2idx: Dict[int, int],
        item2idx: Dict[int, int],
        n_users: int,
        n_items: int
    ):
        """
        Initialize graph builder.

        Args:
            train_df: Training ratings DataFrame
            user2idx: Mapping from user_id to index
            item2idx: Mapping from movie_id to index
            n_users: Total number of users
            n_items: Total number of items
        """
        self.train_df = train_df
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.n_users = n_users
        self.n_items = n_items

        # Build interaction structures
        self.user_items = defaultdict(set)  # user -> set of items
        self.item_users = defaultdict(set)  # item -> set of users

        for _, row in train_df.iterrows():
            u_idx = user2idx[row['user_id']]
            i_idx = item2idx[row['movie_id']]
            self.user_items[u_idx].add(i_idx)
            self.item_users[i_idx].add(u_idx)

    def build_graph(self) -> Data:
        """
        Build PyTorch Geometric graph.

        In LightGCN, we create an undirected bipartite graph:
        - Nodes: users (0 to n_users-1) + items (n_users to n_users+n_items-1)
        - Edges: user-item interactions (bidirectional)

        Returns:
            PyTorch Geometric Data object
        """
        edge_list = []

        # Add edges (both directions for undirected graph)
        for _, row in self.train_df.iterrows():
            u_idx = self.user2idx[row['user_id']]
            i_idx = self.item2idx[row['movie_id']]

            # User node index: u_idx
            # Item node index: n_users + i_idx (offset by n_users)
            user_node = u_idx
            item_node = self.n_users + i_idx

            # Add bidirectional edges
            edge_list.append([user_node, item_node])
            edge_list.append([item_node, user_node])

        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()

        # Create graph
        graph = Data(
            edge_index=edge_index,
            num_nodes=self.n_users + self.n_items
        )

        return graph

    def get_user_items_dict(self) -> Dict[int, List[int]]:
        """
        Get user-items interaction dictionary.

        Returns:
            Dictionary mapping user index to list of item indices
        """
        return {u: list(items) for u, items in self.user_items.items()}

    def get_item_users_dict(self) -> Dict[int, List[int]]:
        """
        Get item-users interaction dictionary.

        Returns:
            Dictionary mapping item index to list of user indices
        """
        return {i: list(users) for i, users in self.item_users.items()}


class NegativeSampler:
    """Negative sampling for BPR training."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_items: Dict[int, set],
        random_seed: int = 42
    ):
        """
        Initialize negative sampler.

        Args:
            n_users: Number of users
            n_items: Number of items
            user_items: Dictionary mapping user index to set of positive items
            random_seed: Random seed
        """
        self.n_users = n_users
        self.n_items = n_items
        self.user_items = user_items
        self.rng = np.random.RandomState(random_seed)

    def sample_negative(self, user_idx: int, n_samples: int = 1) -> List[int]:
        """
        Sample negative items for a user.

        Args:
            user_idx: User index
            n_samples: Number of negative samples

        Returns:
            List of negative item indices
        """
        positive_items = self.user_items.get(user_idx, set())
        negative_items = []

        while len(negative_items) < n_samples:
            # Sample random item
            neg_item = self.rng.randint(0, self.n_items)

            # Check if it's not in positive items
            if neg_item not in positive_items and neg_item not in negative_items:
                negative_items.append(neg_item)

        return negative_items

    def sample_batch(
        self,
        user_indices: List[int],
        n_negatives_per_positive: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of (user, positive_item, negative_item) triplets.

        Args:
            user_indices: List of user indices
            n_negatives_per_positive: Number of negative samples per positive

        Returns:
            Tuple of (users, pos_items, neg_items) arrays
        """
        users = []
        pos_items = []
        neg_items = []

        for u_idx in user_indices:
            # Get positive items
            positive_items = list(self.user_items.get(u_idx, set()))

            if len(positive_items) == 0:
                continue

            # Sample positive item
            pos_item = self.rng.choice(positive_items)

            # Sample negative items
            neg_item_list = self.sample_negative(u_idx, n_negatives_per_positive)

            for neg_item in neg_item_list:
                users.append(u_idx)
                pos_items.append(pos_item)
                neg_items.append(neg_item)

        return (
            np.array(users, dtype=np.int64),
            np.array(pos_items, dtype=np.int64),
            np.array(neg_items, dtype=np.int64)
        )


def create_train_dataloader(
    train_df: pd.DataFrame,
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
    n_users: int,
    n_items: int,
    batch_size: int = 1024,
    n_negatives: int = 1,
    shuffle: bool = True,
    random_seed: int = 42
):
    """
    Create training dataloader with negative sampling.

    Args:
        train_df: Training DataFrame
        user2idx: User to index mapping
        item2idx: Item to index mapping
        n_users: Number of users
        n_items: Number of items
        batch_size: Batch size
        n_negatives: Number of negative samples per positive
        shuffle: Whether to shuffle
        random_seed: Random seed

    Yields:
        Batches of (users, pos_items, neg_items) tensors
    """
    # Build user_items dict
    user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        u_idx = user2idx[row['user_id']]
        i_idx = item2idx[row['movie_id']]
        user_items[u_idx].add(i_idx)

    # Create sampler
    sampler = NegativeSampler(n_users, n_items, user_items, random_seed)

    # Get all users with interactions
    users_with_interactions = list(user_items.keys())

    # Shuffle if needed
    rng = np.random.RandomState(random_seed)
    if shuffle:
        rng.shuffle(users_with_interactions)

    # Generate batches
    n_batches = (len(users_with_interactions) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(users_with_interactions))
        batch_users = users_with_interactions[start_idx:end_idx]

        # Sample triplets
        users, pos_items, neg_items = sampler.sample_batch(
            batch_users,
            n_negatives_per_positive=n_negatives
        )

        # Convert to tensors
        users_tensor = torch.from_numpy(users).long()
        pos_items_tensor = torch.from_numpy(pos_items).long()
        neg_items_tensor = torch.from_numpy(neg_items).long()

        yield users_tensor, pos_items_tensor, neg_items_tensor
