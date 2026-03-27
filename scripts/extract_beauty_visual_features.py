#!/usr/bin/env python3
"""
Extract visual features from Amazon Beauty product images using ResNet50.

This script extracts 2048-dimensional visual features from ResNet50 pretrained on ImageNet.

Usage:
    python scripts/extract_beauty_visual_features.py \
        --image_dir /path/to/beauty/images \
        --output data/recbole/amazon-beauty/visual_features.npy
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
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"Loading ResNet50 (pretrained on ImageNet)...")
        resnet = models.resnet50(weights='IMAGENET1K_V1')

        # Remove the final classification layer to get 2048-dim features
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

        print(f"ResNet50 loaded on {self.device}")
        print(f"Feature dimension: 2048")

    def extract_batch(self, image_paths, batch_size=32):
        """Extract features from multiple images in batches."""
        features = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                except Exception as e:
                    batch_images.append(None)

            valid_images = [img for img in batch_images if img is not None]
            if valid_images:
                batch_tensor = torch.stack(valid_images).to(self.device)

                with torch.no_grad():
                    batch_features = self.model(batch_tensor)

                batch_features = batch_features.squeeze(-1).squeeze(-1).cpu().numpy()

                if len(valid_images) == 1:
                    batch_features = batch_features.reshape(1, -1)

                feature_idx = 0
                for img in batch_images:
                    if img is not None:
                        features.append(batch_features[feature_idx])
                        feature_idx += 1
                    else:
                        features.append(np.zeros(2048, dtype=np.float32))
            else:
                for _ in batch_paths:
                    features.append(np.zeros(2048, dtype=np.float32))

        return np.array(features)


def main():
    parser = argparse.ArgumentParser(
        description='Extract visual features from Beauty product images'
    )

    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing product images (named by ASIN)')
    parser.add_argument('--data_path', type=str, default='data/recbole/amazon-beauty',
                        help='Path to RecBole data directory')
    parser.add_argument('--output', type=str, default='data/recbole/amazon-beauty/visual_features.npy',
                        help='Output .npy file for visual features')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    data_path = Path(args.data_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Visual Feature Extraction (ResNet50) - Amazon Beauty")
    print("="*70)
    print(f"Image directory: {image_dir}")
    print(f"Data path: {data_path}")
    print(f"Output: {output_path}")
    print()

    # Load ID mapping
    mapping_file = data_path / "id_mappings.json"
    print(f"Loading ID mapping from {mapping_file}...")

    with open(mapping_file, 'r') as f:
        mappings = json.load(f)

    # Get RecBole ID -> original ASIN mapping
    recbole_to_original = mappings['item']['recbole_to_original']

    # Get number of items
    n_items = len(recbole_to_original)
    print(f"Total items: {n_items}")
    print()

    # Initialize feature extractor
    extractor = VisualFeatureExtractor(device=args.device)
    print()

    # Collect image paths in order of RecBole ID
    print("Collecting image paths...")
    image_paths_ordered = []
    missing_images = []

    for recbole_id in range(1, n_items + 1):
        original_asin = recbole_to_original.get(str(recbole_id))
        if not original_asin:
            missing_images.append(recbole_id)
            image_paths_ordered.append(None)
            continue

        # Try different extensions
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '']:
            candidate = image_dir / f"{original_asin}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path:
            image_paths_ordered.append(str(image_path))
        else:
            missing_images.append(recbole_id)
            image_paths_ordered.append(None)

    found_count = sum(1 for p in image_paths_ordered if p is not None)
    print(f"Found {found_count} images")
    print(f"Missing {len(missing_images)} images (will use zero vectors)")
    print()

    # Extract features
    print(f"Extracting features (batch_size={args.batch_size})...")

    # Filter out None paths for batch processing
    valid_paths = [(i, p) for i, p in enumerate(image_paths_ordered) if p is not None]
    valid_indices = [i for i, _ in valid_paths]
    paths_only = [p for _, p in valid_paths]

    if paths_only:
        features = extractor.extract_batch(paths_only, batch_size=args.batch_size)
    else:
        features = np.array([])

    # Create feature matrix (0-indexed for numpy, but represents 1-indexed RecBole IDs)
    feature_matrix = np.zeros((n_items, 2048), dtype=np.float32)

    for i, feat_idx in enumerate(valid_indices):
        if i < len(features):
            feature_matrix[feat_idx] = features[i]

    # Save features
    print()
    print(f"Saving features to {output_path}...")
    np.save(output_path, feature_matrix)

    # Summary
    print()
    print("="*70)
    print("Feature Extraction Complete")
    print("="*70)
    print(f"Total items: {n_items}")
    print(f"Features extracted: {found_count}")
    print(f"Missing images (zero vectors): {len(missing_images)}")
    print(f"Feature shape: {feature_matrix.shape}")
    print(f"Non-zero rows: {(feature_matrix.sum(axis=1) != 0).sum()}")
    print(f"Output: {output_path}")
    print("="*70)


if __name__ == '__main__':
    main()
