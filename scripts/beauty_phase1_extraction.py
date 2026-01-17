#!/usr/bin/env python3
"""
Beauty Phase 1: Free Exploration - Extract knowledge from product images

Extract visual knowledge from 58 Beauty products using free-form prompts.
No vocabulary constraints - let LLM discover visual patterns.

Usage:
    python scripts/beauty_phase1_extraction.py \\
        --api_key YOUR_KEY \\
        --base_url YOUR_URL \\
        --workers 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import threading
from datetime import datetime
from typing import List, Dict, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from src.extraction.mllm_interface import create_mllm
from src.extraction.prompts import PromptTemplates


def load_beauty_items(mapping_file: Path) -> List[Dict]:
    """Load Beauty items from filtered metadata."""
    with open(mapping_file, 'r') as f:
        metadata = json.load(f)

    # Convert to list with ASIN
    items = []
    for asin, meta in metadata.items():
        items.append({
            'asin': asin,
            'title': meta.get('title', 'Unknown'),
            'brand': meta.get('brand', 'Unknown')
        })

    return sorted(items, key=lambda x: x['asin'])


def load_product_image(asin: str, image_dir: Path, max_size=(1024, 1024)) -> Image.Image:
    """Load and resize product image."""
    image_path = image_dir / f"{asin}.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)

    # Resize if needed
    if max_size:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

    return image


def extract_single_product(
    asin: str,
    title: str,
    brand: str,
    image_dir: Path,
    mllm,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int
) -> Dict:
    """Extract knowledge from a single product image."""
    try:
        # Load image
        image = load_product_image(asin, image_dir, max_size=(1024, 1024))

        # Extract knowledge
        output = mllm.extract_from_image(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Parse output
        knowledge_points = PromptTemplates.parse_extraction_output(output)

        return {
            'asin': asin,
            'title': title,
            'brand': brand,
            'num_knowledge_points': len(knowledge_points),
            'knowledge_points': knowledge_points,
            'raw_output': output,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

    except Exception as e:
        return {
            'asin': asin,
            'title': title,
            'brand': brand,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def extract_concurrent(
    items: List[Dict],
    image_dir: Path,
    mllm,
    output_file: Path,
    temperature: float = 0.0,
    max_tokens: int = 800,
    max_workers: int = 10,
    save_interval: int = 10,
    verbose: bool = True
) -> List[Dict]:
    """Extract knowledge concurrently with multi-threading."""

    # Get Phase 1 prompts
    system_prompt = PromptTemplates.get_phase1_system_prompt()

    # Adapt user prompt for Beauty products
    def get_beauty_user_prompt(title: str) -> str:
        return f"""Analyze this beauty/personal care product image (Product: "{title}") and extract visual knowledge points.

**Output Format:**
Provide knowledge points as relation-entity pairs, one per line:
```
<relation>: <entity>
```

**What are Relations and Entities?**
- **Relation**: A type or category of visual feature (e.g., "product_type", "color_scheme", "packaging_style")
- **Entity**: A specific characteristic within that category (e.g., "lipstick", "warm_tones", "minimalist")

**Guidelines for Relations:**
1. Be creative - discover diverse types of visual features you observe
2. Relations can describe: product types, colors, packaging, materials, text, shapes, branding elements, or ANY visual aspect
3. Use clear, descriptive names with underscores (e.g., "dominant_color", "packaging_material", "product_category")

**CRITICAL Guidelines for Entities:**
1. Use ABSTRACT, HIGH-LEVEL terms that can apply to MULTIPLE products
2. Avoid overly specific descriptions - think in CATEGORIES, not unique details
3. Prefer COMMON visual terms over rare combinations
4. Ask yourself: "Could this entity describe other products too?"

**Good Entity Examples (abstract, reusable):**
- rose_gold, metallic, matte_finish (NOT "shiny_pink_with_gold_specks")
- tube_packaging, bottle, compact (NOT "cylindrical_container_with_twist_cap")
- minimalist, luxury, playful (NOT "simple_white_box_with_one_logo")

**Task Requirements:**
1. Extract AT LEAST 10 knowledge points (aim for 10-15)
2. Use diverse relation types - explore different visual aspects
3. Keep entities abstract and reusable
4. Focus on characteristics useful for product recommendation
5. Use underscores for multi-word terms

Now extract knowledge points from the product image:"""

    if verbose:
        print(f"\nExtraction configuration:")
        print(f"  Total products: {len(items)}")
        print(f"  Concurrent workers: {max_workers}")
        print(f"  Temperature: {temperature}")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Prompt: Phase 1 (Free Exploration)")

    results = []
    results_lock = threading.Lock()

    if verbose:
        progress_bar = tqdm(total=len(items), desc="Extracting")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_item = {
            executor.submit(
                extract_single_product,
                item['asin'],
                item['title'],
                item['brand'],
                image_dir,
                mllm,
                system_prompt,
                get_beauty_user_prompt(item['title']),
                temperature,
                max_tokens
            ): item for item in items
        }

        # Collect results
        for future in as_completed(future_to_item):
            result = future.result()

            with results_lock:
                results.append(result)

                # Periodic save
                if len(results) % save_interval == 0:
                    save_results(output_file, results)

            if verbose:
                progress_bar.update(1)
                status = "✓" if result['status'] == 'success' else "✗"
                progress_bar.set_postfix_str(
                    f"{status} {result['asin']} ({len(results)}/{len(items)})"
                )

        if verbose:
            progress_bar.close()

    # Final save
    save_results(output_file, results)

    return results


def save_results(output_file: Path, results: List[Dict]):
    """Save extraction results to JSON file (thread-safe)."""
    output_data = {
        'phase': 'beauty_phase1_free_exploration',
        'dataset': 'amazon_beauty',
        'results': results,
        'total_products': len(results),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'error'),
        'timestamp': datetime.now().isoformat()
    }

    # Atomic write
    temp_file = output_file.with_suffix('.tmp.json')
    with open(temp_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    temp_file.replace(output_file)


def main():
    parser = argparse.ArgumentParser(description='Beauty Phase 1: Free Exploration')
    parser.add_argument('--metadata', type=str,
                       default='data/recbole/amazon-beauty/mappings/filtered_metadata.json',
                       help='Path to filtered metadata JSON')
    parser.add_argument('--image_dir', type=str,
                       default='data/raw/amazon-beauty/images',
                       help='Directory containing product images')
    parser.add_argument('--output', type=str,
                       default='results/beauty_phase1_extraction.json',
                       help='Output JSON file')
    parser.add_argument('--api_key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--base_url', type=str,
                       help='Optional API base URL')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Model name (default: gpt-4o-mini)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=800,
                       help='Maximum output tokens')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of concurrent workers')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save results every N items')

    args = parser.parse_args()

    print("="*70)
    print("Beauty Phase 1: Free Exploration")
    print("="*70)

    # Load items
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        print("Please run prepare_beauty_dataset.py first")
        return

    items = load_beauty_items(metadata_path)
    print(f"\nLoaded {len(items)} products from metadata")

    # Check image directory
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        print("Please run download_beauty_images.py first")
        return

    # Create output directory
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create MLLM
    mllm = create_mllm(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model
    )
    print(f"Using model: {args.model}")

    # Extract
    results = extract_concurrent(
        items=items,
        image_dir=image_dir,
        mllm=mllm,
        output_file=output_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.workers,
        save_interval=args.save_interval,
        verbose=True
    )

    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_kps = sum(r.get('num_knowledge_points', 0) for r in results if r['status'] == 'success')

    print(f"\n{'='*70}")
    print("Extraction Complete!")
    print(f"{'='*70}")
    print(f"Total products: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"Total knowledge points: {total_kps}")
    print(f"Average KPs per product: {total_kps/success_count:.1f}" if success_count > 0 else "N/A")
    print(f"\nResults saved to: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Review results and standardize vocabulary (manual or LLM-assisted)")
    print(f"  2. Run Phase 4 extraction with standardized vocabulary")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
