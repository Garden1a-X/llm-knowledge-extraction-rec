#!/usr/bin/env python3
"""
Extract visual features from Video Games cover images using ResNet50.

This script adapts the ML-1M feature extraction for Amazon Video Games dataset.

Usage:
    python baselines/extract_videogames_visual_features.py \
        --image_dir data/recbole/amazon-videogames/images \
        --item_file data/recbole/amazon-videogames/amazon-videogames.item \
        --id_mapping data/recbole/amazon-videogames/mappings/item_mapping.json \
        --output data/recbole/amazon-videogames/visual_features.npy
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

            # Load and preprocess batch
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                except Exception as e:
                    # Use zero feature for missing/corrupted images
                    batch_images.append(None)

            # Process valid images
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

                # Add back None features for failed images
                feature_idx = 0
                for img in batch_images:
                    if img is not None:
                        features.append(batch_features[feature_idx])
                        feature_idx += 1
                    else:
                        features.append(np.zeros(2048, dtype=np.float32))
            else:
                # All images in batch failed
                for _ in batch_paths:
                    features.append(np.zeros(2048, dtype=np.float32))

        return np.array(features)


def load_videogames_mapping(item_file, mapping_file):
    """
    Load Video Games item ID mapping.

    Args:
        item_file: Path to amazon-videogames.item file
        mapping_file: Path to item_mapping.json

    Returns:
        List of (recbole_id, original_asin) tuples, sorted by recbole_id
    """
    # Load mapping: {"original_to_recbole": {"0700026398": 1, ...}}
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)

    original_to_recbole = mapping_data['original_to_recbole']

    # Build reverse mapping
    recbole_to_original = {v: k for k, v in original_to_recbole.items()}

    # Load item IDs from .item file
    item_ids = []
    with open(item_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                recbole_id = int(parts[0])
                item_ids.append(recbole_id)

    # Create mapping list
    mapping = []
    for recbole_id in item_ids:
        original_asin = recbole_to_original.get(recbole_id, None)
        if original_asin:
            mapping.append((recbole_id, original_asin))
        else:
            print(f"Warning: No original ASIN found for RecBole ID {recbole_id}")

    # Sort by recbole_id to ensure correct order
    mapping.sort(key=lambda x: x[0])

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description='Extract visual features from Video Games cover images using ResNet50'
    )

    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing game cover images (named by original ASIN)')
    parser.add_argument('--item_file', type=str, required=True,
                        help='Path to amazon-videogames.item file')
    parser.add_argument('--id_mapping', type=str, required=True,
                        help='Path to item_mapping.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .npy file for visual features')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Visual Feature Extraction (ResNet50) - Video Games")
    print("="*70)
    print(f"Image directory: {image_dir}")
    print(f"Item file: {args.item_file}")
    print(f"ID mapping: {args.id_mapping}")
    print(f"Output: {output_path}")
    print()

    # Load item mapping
    print("Loading item mapping...")
    mapping = load_videogames_mapping(args.item_file, args.id_mapping)
    print(f"✓ Loaded {len(mapping)} items")
    print()

    # Initialize feature extractor
    extractor = VisualFeatureExtractor(device=args.device)
    print()

    # Collect image paths
    print("Collecting image paths...")
    image_paths_ordered = []
    missing_images = []

    for recbole_id, original_asin in mapping:
        # Try different extensions
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = image_dir / f"{original_asin}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path:
            image_paths_ordered.append((recbole_id, image_path))
        else:
            missing_images.append((recbole_id, original_asin))

    print(f"✓ Found {len(image_paths_ordered)} images")
    if missing_images:
        print(f"⚠ Missing {len(missing_images)} images (will use zero vectors)")
        print(f"  First 10 missing: {[asin for _, asin in missing_images[:10]]}")
    print()

    # Extract features
    print(f"Extracting features (batch_size={args.batch_size})...")

    paths_only = [str(path) for _, path in image_paths_ordered]
    features = extractor.extract_batch(paths_only, batch_size=args.batch_size)

    # Create feature matrix aligned with RecBole item IDs
    # RecBole IDs are 1-indexed
    num_items = max(recbole_id for recbole_id, _ in mapping)
    feature_dim = 2048

    feature_matrix = np.zeros((num_items, feature_dim), dtype=np.float32)

    # Fill in extracted features
    for i, (recbole_id, _) in enumerate(image_paths_ordered):
        if i < len(features):
            feature_matrix[recbole_id - 1] = features[i]

    # Missing images already have zero vectors (initialized above)

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
    print(f"Features extracted: {len(image_paths_ordered)}")
    print(f"Missing images (zero vectors): {len(missing_images)}")
    print(f"Feature shape: {feature_matrix.shape}")
    print(f"Feature stats:")
    print(f"  Non-zero rows: {(feature_matrix.sum(axis=1) != 0).sum()}")
    print(f"  Mean magnitude: {np.linalg.norm(feature_matrix, axis=1).mean():.4f}")
    print(f"Output: {output_path}")
    print("="*70)


if __name__ == '__main__':
    main()
