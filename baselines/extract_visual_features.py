#!/usr/bin/env python3
"""
Extract visual features from movie posters using ResNet50.

This script extracts 2048-dimensional visual features from the last hidden layer
of ResNet50 pretrained on ImageNet, following the MKGAT paper approach.

Usage:
    python baselines/extract_visual_features.py \
        --poster_dir /path/to/posters \
        --item_file data/recbole/ml-1m/ml-1m.item \
        --id_mapping data/recbole/ml-1m/id_mappings.json \
        --output data/recbole/ml-1m/visual_features.npy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm


class VisualFeatureExtractor:
    """Extract visual features using ResNet50."""

    def __init__(self, device='cuda'):
        """
        Initialize ResNet50 feature extractor.

        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load pretrained ResNet50
        print(f"Loading ResNet50 (pretrained on ImageNet)...")
        resnet = models.resnet50(pretrained=True)

        # Remove the final classification layer to get features
        # ResNet50 outputs 2048-dim features from avgpool layer
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"✓ ResNet50 loaded on {self.device}")
        print(f"✓ Feature dimension: 2048")

    def extract_single(self, image_path):
        """
        Extract feature from a single image.

        Args:
            image_path: Path to image file

        Returns:
            numpy array of shape (2048,)
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Extract feature
            with torch.no_grad():
                feature = self.model(image_tensor)

            # Reshape from (1, 2048, 1, 1) to (2048,)
            feature = feature.squeeze().cpu().numpy()

            return feature

        except Exception as e:
            print(f"Warning: Failed to extract feature from {image_path}: {e}")
            return None

    def extract_batch(self, image_paths, batch_size=32):
        """
        Extract features from multiple images in batches.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (N, 2048)
        """
        features = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_indices = []

            # Load and preprocess batch
            for idx, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Warning: Skipping {path}: {e}")
                    batch_images.append(None)

            if not batch_images:
                continue

            # Stack valid images
            valid_images = [img for img in batch_images if img is not None]
            if valid_images:
                batch_tensor = torch.stack(valid_images).to(self.device)

                # Extract features
                with torch.no_grad():
                    batch_features = self.model(batch_tensor)

                # Reshape and convert to numpy
                batch_features = batch_features.squeeze().cpu().numpy()

                # Handle single image case
                if len(valid_images) == 1:
                    batch_features = batch_features.reshape(1, -1)

                features.append(batch_features)

        if features:
            return np.vstack(features)
        else:
            return np.array([])


def load_item_mapping(item_file, id_mapping_file):
    """
    Load item ID mapping.

    Args:
        item_file: Path to .item file
        id_mapping_file: Path to id_mappings.json

    Returns:
        List of (recbole_id, original_movie_id) tuples
    """
    # Load ID mapping
    with open(id_mapping_file, 'r') as f:
        mappings = json.load(f)

    new_to_original = mappings['item_id_map']['new_to_original']

    # Load item IDs from .item file
    item_ids = []
    with open(item_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            recbole_id = int(parts[0])
            item_ids.append(recbole_id)

    # Create mapping list
    mapping = []
    for recbole_id in item_ids:
        original_id = int(new_to_original[str(recbole_id)])
        mapping.append((recbole_id, original_id))

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description='Extract visual features from movie posters using ResNet50'
    )

    parser.add_argument('--poster_dir', type=str, required=True,
                        help='Directory containing poster images (named by original movie_id)')
    parser.add_argument('--item_file', type=str, required=True,
                        help='Path to .item file')
    parser.add_argument('--id_mapping', type=str, required=True,
                        help='Path to id_mappings.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .npy file for visual features')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    poster_dir = Path(args.poster_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Visual Feature Extraction (ResNet50)")
    print("="*70)
    print(f"Poster directory: {poster_dir}")
    print(f"Item file: {args.item_file}")
    print(f"Output: {output_path}")
    print()

    # Load item mapping
    print("Loading item mapping...")
    mapping = load_item_mapping(args.item_file, args.id_mapping)
    print(f"✓ Loaded {len(mapping)} items")
    print()

    # Initialize feature extractor
    extractor = VisualFeatureExtractor(device=args.device)
    print()

    # Collect poster paths
    print("Collecting poster paths...")
    poster_paths = []
    missing_posters = []

    for recbole_id, original_id in mapping:
        poster_path = poster_dir / f"{original_id}.jpg"
        if poster_path.exists():
            poster_paths.append((recbole_id, poster_path))
        else:
            missing_posters.append((recbole_id, original_id))

    print(f"✓ Found {len(poster_paths)} posters")
    if missing_posters:
        print(f"⚠ Missing {len(missing_posters)} posters")
    print()

    # Extract features
    print(f"Extracting features (batch_size={args.batch_size})...")

    # Sort by recbole_id to ensure correct order
    poster_paths.sort(key=lambda x: x[0])

    # Extract features
    paths_only = [str(path) for _, path in poster_paths]
    features = extractor.extract_batch(paths_only, batch_size=args.batch_size)

    # Create feature matrix aligned with RecBole item IDs
    # RecBole IDs are 1-indexed, so we need num_items rows
    num_items = max(recbole_id for recbole_id, _ in mapping)
    feature_dim = 2048

    feature_matrix = np.zeros((num_items, feature_dim), dtype=np.float32)

    for i, (recbole_id, _) in enumerate(poster_paths):
        if i < len(features):
            feature_matrix[recbole_id - 1] = features[i]

    # For missing posters, use zero vectors (will be handled in model)
    for recbole_id, _ in missing_posters:
        feature_matrix[recbole_id - 1] = np.zeros(feature_dim, dtype=np.float32)

    # Save features
    print()
    print(f"Saving features to {output_path}...")
    np.save(output_path, feature_matrix)

    # Summary
    print()
    print("="*70)
    print("Feature Extraction Complete")
    print("="*70)
    print(f"Total items: {num_items}")
    print(f"Features extracted: {len(poster_paths)}")
    print(f"Missing posters (zero vectors): {len(missing_posters)}")
    print(f"Feature shape: {feature_matrix.shape}")
    print(f"Output: {output_path}")
    print("="*70)


if __name__ == '__main__':
    main()
