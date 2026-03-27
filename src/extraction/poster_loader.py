"""
Poster image loader with ID mapping support.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import random


class PosterLoader:
    """Loads movie posters using ID mappings."""

    def __init__(
        self,
        poster_dir: str,
        id_mapping_path: str,
        image_format: str = 'RGB'
    ):
        """
        Initialize poster loader.

        Args:
            poster_dir: Directory containing poster images (named by original movie_id)
            id_mapping_path: Path to id_mappings.json
            image_format: PIL image mode (default: 'RGB')
        """
        self.poster_dir = Path(poster_dir)
        self.image_format = image_format

        # Load ID mappings
        with open(id_mapping_path, 'r') as f:
            mappings = json.load(f)

        self.item_id_map = mappings['item_id_map']
        self.stats = mappings['stats']

        # Convert string keys to int for easier lookup
        self.new_to_original = {
            int(k): v for k, v in self.item_id_map['new_to_original'].items()
        }
        self.original_to_new = {
            int(k): v for k, v in self.item_id_map['original_to_new'].items()
        }

        print(f"PosterLoader initialized:")
        print(f"  Poster directory: {self.poster_dir}")
        print(f"  Total items: {self.stats['num_items']}")
        print(f"  Image format: {self.image_format}")

    def get_original_movie_id(self, recbole_id: int) -> int:
        """
        Map RecBole ID to original MovieLens movie_id.

        Args:
            recbole_id: RecBole item ID (continuous, 1-indexed)

        Returns:
            Original MovieLens movie_id

        Raises:
            KeyError: If recbole_id not found
        """
        return self.new_to_original[recbole_id]

    def get_recbole_id(self, original_movie_id: int) -> int:
        """
        Map original MovieLens movie_id to RecBole ID.

        Args:
            original_movie_id: Original MovieLens movie_id

        Returns:
            RecBole item ID

        Raises:
            KeyError: If original_movie_id not in filtered dataset
        """
        return self.original_to_new[original_movie_id]

    def get_poster_path(self, recbole_id: int) -> Path:
        """
        Get poster file path for a RecBole item ID.

        Args:
            recbole_id: RecBole item ID

        Returns:
            Path to poster image file

        Raises:
            FileNotFoundError: If poster file doesn't exist
        """
        original_id = self.get_original_movie_id(recbole_id)
        poster_path = self.poster_dir / f"{original_id}.jpg"

        if not poster_path.exists():
            raise FileNotFoundError(
                f"Poster not found for RecBole ID {recbole_id} "
                f"(original movie_id {original_id}): {poster_path}"
            )

        return poster_path

    def load_poster(
        self,
        recbole_id: int,
        max_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Load poster image for a RecBole item ID.

        Args:
            recbole_id: RecBole item ID
            max_size: Optional (width, height) to resize image

        Returns:
            PIL Image object

        Raises:
            FileNotFoundError: If poster file doesn't exist
        """
        poster_path = self.get_poster_path(recbole_id)
        image = Image.open(poster_path)

        # Convert to desired format
        if image.mode != self.image_format:
            image = image.convert(self.image_format)

        # Resize if requested
        if max_size:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

        return image

    def load_posters_batch(
        self,
        recbole_ids: List[int],
        max_size: Optional[Tuple[int, int]] = None,
        skip_missing: bool = True
    ) -> Dict[int, Image.Image]:
        """
        Load multiple posters in batch.

        Args:
            recbole_ids: List of RecBole item IDs
            max_size: Optional (width, height) to resize images
            skip_missing: If True, skip missing posters; if False, raise error

        Returns:
            Dict mapping recbole_id to PIL Image

        Raises:
            FileNotFoundError: If poster missing and skip_missing=False
        """
        images = {}
        missing = []

        for recbole_id in recbole_ids:
            try:
                images[recbole_id] = self.load_poster(recbole_id, max_size)
            except FileNotFoundError as e:
                if skip_missing:
                    missing.append(recbole_id)
                else:
                    raise e

        if missing:
            print(f"Warning: Skipped {len(missing)} missing posters")

        return images

    def sample_items(
        self,
        n: int,
        seed: Optional[int] = None
    ) -> List[int]:
        """
        Sample random RecBole item IDs.

        Args:
            n: Number of items to sample
            seed: Random seed for reproducibility

        Returns:
            List of RecBole item IDs (sorted)

        Raises:
            ValueError: If n > total items
        """
        total_items = self.stats['num_items']

        if n > total_items:
            raise ValueError(
                f"Cannot sample {n} items from {total_items} total items"
            )

        if seed is not None:
            random.seed(seed)

        # RecBole IDs are 1-indexed, range [1, num_items]
        all_ids = list(range(1, total_items + 1))
        sampled = random.sample(all_ids, n)

        return sorted(sampled)

    def get_all_item_ids(self) -> List[int]:
        """
        Get all RecBole item IDs.

        Returns:
            List of all RecBole IDs (sorted)
        """
        return list(range(1, self.stats['num_items'] + 1))

    def get_item_info(self, recbole_id: int) -> Dict:
        """
        Get information about an item.

        Args:
            recbole_id: RecBole item ID

        Returns:
            Dict with item information
        """
        original_id = self.get_original_movie_id(recbole_id)
        poster_path = self.poster_dir / f"{original_id}.jpg"

        return {
            'recbole_id': recbole_id,
            'original_movie_id': original_id,
            'poster_path': str(poster_path),
            'poster_exists': poster_path.exists()
        }


# Example usage and testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test PosterLoader')
    parser.add_argument('--poster_dir', type=str, required=True,
                        help='Directory containing posters')
    parser.add_argument('--id_mapping', type=str, required=True,
                        help='Path to id_mappings.json')
    parser.add_argument('--sample', type=int, default=5,
                        help='Number of items to sample for testing')

    args = parser.parse_args()

    # Initialize loader
    print("Initializing PosterLoader...")
    loader = PosterLoader(
        poster_dir=args.poster_dir,
        id_mapping_path=args.id_mapping
    )

    print(f"\nSampling {args.sample} random items...")
    sampled_ids = loader.sample_items(args.sample, seed=42)

    print(f"\nSampled RecBole IDs: {sampled_ids}")
    print("\nItem details:")
    for recbole_id in sampled_ids:
        info = loader.get_item_info(recbole_id)
        print(f"  RecBole ID {info['recbole_id']} -> "
              f"Movie ID {info['original_movie_id']} -> "
              f"{'✓' if info['poster_exists'] else '✗'} {info['poster_path']}")

    print(f"\nLoading posters...")
    images = loader.load_posters_batch(sampled_ids, max_size=(512, 768))
    print(f"Successfully loaded {len(images)} posters")

    for recbole_id, image in images.items():
        print(f"  RecBole ID {recbole_id}: {image.size} {image.mode}")

    print("\n✅ PosterLoader test completed!")
