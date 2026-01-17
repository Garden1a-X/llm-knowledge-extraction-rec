#!/usr/bin/env python3
"""
Phase 4: Beauty Product Knowledge Extraction

Final extraction for Beauty dataset using 13 standardized relations.
- Simplified prompt without NEW_ mechanism
- Direct extraction using vocabulary as reference
- Post-processing filters to vocabulary-only entities
- Multi-threaded for fast processing (58 products)

Usage:
    python scripts/beauty_phase4_extraction.py \
        --metadata data/recbole/amazon-beauty/mappings/filtered_metadata.json \
        --image_dir data/raw/amazon-beauty/images \
        --vocabulary results/beauty_vocabulary_standardized.json \
        --api_key YOUR_KEY \
        --base_url YOUR_URL \
        --workers 10 \
        --output results/beauty_phase4_extraction.json
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


def load_vocabulary(vocab_file: Path) -> Dict[str, List[str]]:
    """
    Load standardized vocabulary from Beauty vocabulary JSON.

    Returns:
        Dict mapping relation names to list of standard entities
    """
    with open(vocab_file, 'r') as f:
        data = json.load(f)

    vocabulary = {}

    # Extract relation names from standardized_vocabulary
    for relation_info in data['standardized_vocabulary']['relations']:
        relation_name = relation_info['standard_name']
        # For Phase 4, we don't pre-fill entities, just list the relation
        # The prompt will guide the LLM to extract appropriate entities
        vocabulary[relation_name] = []

    return vocabulary


def load_metadata(metadata_file: Path) -> List[Dict]:
    """Load filtered metadata JSON."""
    with open(metadata_file, 'r') as f:
        meta_dict = json.load(f)

    # Convert dict to list of items with ASIN
    items = []
    for asin, meta in meta_dict.items():
        item = meta.copy()
        item['asin'] = asin
        items.append(item)

    return sorted(items, key=lambda x: x['asin'])


def load_product_image(asin: str, image_dir: Path, max_size=(1024, 1024)) -> Image.Image:
    """Load and resize product image."""
    image_path = image_dir / f"{asin}.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert('RGB')

    # Resize if needed
    if max_size:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

    return image


def get_phase4_system_prompt(vocabulary: Dict[str, List[str]]) -> str:
    """
    Get system prompt for Beauty Phase 4 extraction.

    Adapted from ML-1M Phase 4 prompts but for beauty products.
    """
    # Build vocabulary description
    vocab_str = "═══════════════════════════════════════════════════════════════\n"
    vocab_str += f"STANDARD VOCABULARY ({len(vocabulary)} Relations)\n"
    vocab_str += "═══════════════════════════════════════════════════════════════\n\n"

    # Relation descriptions (from beauty_vocabulary_standardized.json)
    relation_descriptions = {
        'product_type': 'Product category/type',
        'color': 'Color-related attributes (dominant, accent, scheme, etc.)',
        'packaging': 'Packaging style and structure',
        'material': 'Material composition',
        'texture': 'Surface texture and feel',
        'design_style': 'Overall design aesthetic',
        'size': 'Product size/dimensions',
        'shape': 'Physical shape',
        'label_style': 'Label and text design',
        'target_audience': 'Intended user group',
        'product_feature': 'Key features and benefits',
        'scent': 'Fragrance/scent profile',
        'brand_element': 'Brand-related visual elements'
    }

    for idx, relation in enumerate(vocabulary.keys(), 1):
        description = relation_descriptions.get(relation, '')
        vocab_str += f"{idx}. {relation}: {description}\n"

    vocab_str += "\n"

    return f"""You are an expert in analyzing product images and extracting visual knowledge for a beauty product recommendation system.

Your task: Analyze the product image and extract visual knowledge points using ONLY the approved relations below.

{vocab_str}═══════════════════════════════════════════════════════════════
EXTRACTION GUIDELINES
═══════════════════════════════════════════════════════════════

1. **Relations**: Use ONLY the {len(vocabulary)} relations listed above
   - Examples: product_type, color, packaging, material, texture, etc.

2. **Entities**: Extract appropriate entity values for each relation
   - Keep entities simple and descriptive (lowercase, underscores)
   - Examples: "lipstick", "red", "glossy", "tube", "compact"
   - Avoid overly complex or specific entity names

3. **Output Format**: One knowledge point per line
   - Format: relation: entity
   - Example: product_type: lipstick
   - Example: color: red
   - Example: texture: glossy

4. **Quality over Quantity**:
   - Extract 8-12 most prominent visual features
   - Focus on features you're confident about
   - Only extract what you can clearly observe in the image

5. **Visual-Only**:
   - Extract ONLY what you can see in the image
   - Do NOT infer features from product name or description
   - Focus on packaging appearance, colors, shapes, textures

═══════════════════════════════════════════════════════════════

Extract knowledge points that best describe this product's visual characteristics."""


def get_phase4_user_prompt(title: str) -> str:
    """Get user prompt for Beauty Phase 4."""
    return f"""Analyze this beauty product image for "{title}" and extract 8-12 visual knowledge points.

Output format (one per line):
relation: entity

Extract the most prominent visual features now:"""


def parse_extraction_output(output: str) -> List[Dict[str, str]]:
    """Parse LLM output into structured knowledge points."""
    knowledge_points = []

    # Remove code blocks if present
    output = output.replace('```', '')

    for line in output.strip().split('\n'):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#') or line.startswith('//'):
            continue

        # Parse relation: entity format
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                relation = parts[0].strip()
                entity = parts[1].strip()

                # Remove numbering prefix (e.g., "1. relation" -> "relation")
                import re
                relation = re.sub(r'^\d+[\.)]\s*', '', relation)
                entity = re.sub(r'^\d+[\.)]\s*', '', entity)

                # Basic validation
                if relation and entity:
                    knowledge_points.append({
                        'relation': relation,
                        'entity': entity
                    })

    return knowledge_points


def extract_single_product(
    asin: str,
    title: str,
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
        knowledge_points = parse_extraction_output(output)

        return {
            'asin': asin,
            'title': title,
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
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def load_existing_results(result_file: Path) -> tuple[Dict, Set[str]]:
    """Load existing extraction results and return processed ASINs."""
    if not result_file.exists():
        return None, set()

    with open(result_file, 'r') as f:
        data = json.load(f)

    # Get successfully processed ASINs only
    processed_asins = {
        r['asin'] for r in data.get('results', [])
        if r.get('status') == 'success'
    }

    return data, processed_asins


def save_results(output_file: Path, config: Dict, results: List[Dict]):
    """Save extraction results to JSON file (thread-safe)."""
    output_data = {
        'phase': 'phase4_beauty_extraction',
        'dataset': 'amazon-beauty',
        'config': config,
        'results': results,
        'total_products': len(results),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'error')
    }

    # Atomic write
    temp_file = output_file.with_suffix('.tmp.json')
    with open(temp_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    temp_file.replace(output_file)


def extract_concurrent(
    items: List[Dict],
    image_dir: Path,
    mllm,
    output_file: Path,
    config: Dict,
    vocabulary: Dict[str, List[str]],
    existing_results: List[Dict] = None,
    skip_processed: Set[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 600,
    max_workers: int = 10,
    save_interval: int = 10,
    verbose: bool = True
) -> List[Dict]:
    """Extract knowledge concurrently with multi-threading."""
    skip_processed = skip_processed or set()

    # Filter items to process
    to_extract = [item for item in items if item['asin'] not in skip_processed]
    to_extract_set = set(item['asin'] for item in to_extract)

    # Keep existing results not being reprocessed
    if existing_results:
        results = [r for r in existing_results if r['asin'] not in to_extract_set]
    else:
        results = []

    if verbose:
        print(f"\nExtraction status:")
        print(f"  Total items: {len(items)}")
        print(f"  Already processed: {len(skip_processed)}")
        print(f"  To extract: {len(to_extract)}")
        print(f"  Concurrent workers: {max_workers}")

    if not to_extract:
        print("  ✓ All items already extracted!")
        return results

    # Get prompts (Phase 4 - simplified)
    system_prompt = get_phase4_system_prompt(vocabulary)

    if verbose:
        print(f"\n  Using {len(vocabulary)} standardized relations")
        print(f"  Prompt: Phase 4 (simplified, no NEW_ mechanism)")

    # Thread-safe results
    results_lock = threading.Lock()
    completed_count = 0

    if verbose:
        progress_bar = tqdm(total=len(to_extract), desc="Extracting")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_asin = {
            executor.submit(
                extract_single_product,
                item['asin'],
                item.get('title', 'Unknown Product'),
                image_dir,
                mllm,
                system_prompt,
                get_phase4_user_prompt(item.get('title', 'Unknown Product')),
                temperature,
                max_tokens
            ): item['asin']
            for item in to_extract
        }

        # Collect results
        for future in as_completed(future_to_asin):
            asin = future_to_asin[future]

            try:
                result = future.result()

                with results_lock:
                    results.append(result)
                    completed_count += 1

                    if verbose:
                        if result['status'] == 'success':
                            kps = result['num_knowledge_points']
                            progress_bar.write(
                                f"  ✓ {asin}: {kps} KPs"
                            )
                        else:
                            progress_bar.write(
                                f"  ✗ {asin}: {result.get('error', 'Unknown error')}"
                            )
                        progress_bar.update(1)

                    # Periodic save
                    if completed_count % save_interval == 0:
                        save_results(output_file, config, results)
                        if verbose:
                            success_rate = sum(1 for r in results if r['status'] == 'success') / len(results) * 100
                            progress_bar.write(
                                f"  💾 Checkpoint: {completed_count}/{len(to_extract)} "
                                f"(Success: {success_rate:.1f}%)"
                            )

            except Exception as e:
                if verbose:
                    progress_bar.write(f"  ✗ {asin}: Unexpected error - {e}")

    if verbose:
        progress_bar.close()

    # Final save
    save_results(output_file, config, results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: Beauty Product Knowledge Extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument('--metadata', type=str,
                       default='data/recbole/amazon-beauty/mappings/filtered_metadata.json',
                       help='Path to filtered metadata JSON')
    parser.add_argument('--image_dir', type=str,
                       default='data/raw/amazon-beauty/images',
                       help='Directory containing product images')
    parser.add_argument('--vocabulary', type=str,
                       default='results/beauty_vocabulary_standardized.json',
                       help='Path to standardized vocabulary JSON')

    # MLLM settings
    parser.add_argument('--backend', type=str, default='openai',
                       choices=['openai', 'local'],
                       help='MLLM backend')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Model name')
    parser.add_argument('--api_key', type=str, required=True,
                       help='API key')
    parser.add_argument('--base_url', type=str, default=None,
                       help='Base URL for API')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=600,
                       help='Max tokens')

    # Concurrency
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of concurrent workers')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N completions')

    # Output
    parser.add_argument('--output', type=str,
                       default='results/beauty_phase4_extraction.json',
                       help='Output JSON file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    # Initialize
    print("="*70)
    print("PHASE 4: BEAUTY PRODUCT KNOWLEDGE EXTRACTION")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Workers: {args.workers}")
    print()

    # Load vocabulary
    vocab_file = Path(args.vocabulary)
    if not vocab_file.exists():
        print(f"Error: Vocabulary file not found: {vocab_file}")
        return 1

    print(f"Loading vocabulary: {vocab_file}")
    vocabulary = load_vocabulary(vocab_file)
    print(f"  ✓ {len(vocabulary)} standardized relations")
    print()

    # Load metadata
    metadata_file = Path(args.metadata)
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        print("\nPlease run prepare_beauty_dataset.py first to create the metadata file")
        return 1

    print(f"Loading metadata: {metadata_file}")
    items = load_metadata(metadata_file)
    print(f"  ✓ Loaded {len(items)} products")

    # Check image directory
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        print("\nPlease run download_beauty_images.py first to download product images")
        return 1

    # Load existing results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing_data, skip_processed = load_existing_results(output_file)
    if skip_processed:
        print(f"\n  Found {len(skip_processed)} already processed products")

    # Initialize MLLM
    print(f"\nInitializing {args.backend} MLLM...")
    mllm_kwargs = {
        'backend': args.backend,
        'model_name': args.model,
        'api_key': args.api_key
    }
    if args.base_url:
        mllm_kwargs['base_url'] = args.base_url
        print(f"  Using custom base URL: {args.base_url}")

    mllm = create_mllm(**mllm_kwargs)
    print(f"  ✓ MLLM initialized")

    # Config
    config = {
        'backend': args.backend,
        'model': args.model,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'vocabulary_file': str(vocab_file),
        'num_relations': len(vocabulary),
        'concurrent_workers': args.workers,
        'save_interval': args.save_interval,
        'timestamp': datetime.now().isoformat()
    }

    # Extract
    print(f"\nStarting extraction...")
    print("-"*70)

    results = extract_concurrent(
        items=items,
        image_dir=image_dir,
        mllm=mllm,
        output_file=output_file,
        config=config,
        vocabulary=vocabulary,
        existing_results=existing_data.get('results', []) if existing_data else None,
        skip_processed=skip_processed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.workers,
        save_interval=args.save_interval,
        verbose=not args.quiet
    )

    # Summary
    print("-"*70)
    print("\nSUMMARY")
    print("="*70)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    total_kps = sum(r.get('num_knowledge_points', 0) for r in results if r['status'] == 'success')
    avg_kps = total_kps / successful if successful > 0 else 0

    print(f"Total extracted: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    print()
    print(f"Total knowledge points: {total_kps}")
    print(f"Average KPs/product: {avg_kps:.1f}")
    print()
    print(f"✓ Results saved to: {output_file}")
    print()
    print("="*70)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
