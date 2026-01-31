#!/usr/bin/env python3
"""
Phase 1: Amazon Beauty Knowledge Extraction (5% Exploration)

Extract visual knowledge from beauty product images.
This phase explores relation types and entity vocabulary.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import argparse
import re
import base64
from datetime import datetime
from typing import List, Dict, Set, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Try to import MLLM interface, fallback to direct API call
try:
    from src.extraction.mllm_interface import create_mllm
    HAS_MLLM = True
except ImportError:
    HAS_MLLM = False
    print("Warning: mllm_interface not found, using direct API calls")


def parse_json_response(response: str) -> List[Dict]:
    """Parse JSON from MLLM response."""
    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    code_block_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON array in text
    array_pattern = r'\[\s*\{.*?\}\s*\]'
    match = re.search(array_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not parse JSON from response", response, 0)


def create_beauty_extraction_prompt(title: str, categories: str) -> str:
    """
    Create extraction prompt for beauty products.
    Focus on visual characteristics visible from product images.
    """
    prompt = f"""You are a beauty product expert analyzing product images.

**Product Information:**
- Title: {title}
- Categories: {categories}

**Task:** Extract visual knowledge from the beauty product image.

Focus on aspects you can DIRECTLY observe in the image:

1. **Product Type**: What type of beauty product (lipstick, eyeshadow, skincare, haircare, perfume, nail polish, etc.)

2. **Color/Shade**: Main colors, color family (warm/cool), specific shades visible

3. **Packaging Design**:
   - Container type (tube, jar, bottle, compact, palette)
   - Design style (luxury, minimalist, colorful, elegant, playful)
   - Material appearance (glass, plastic, metal, matte, glossy)

4. **Brand Aesthetic**: Visual style suggesting brand positioning (high-end, drugstore, natural/organic, trendy)

5. **Product Appearance**:
   - Texture visible (creamy, powdery, liquid, gel, mousse)
   - Finish (matte, shimmer, glossy, satin, metallic)
   - Size indicator if visible

6. **Target Occasion**: Visual cues for intended use (everyday, party, professional, special occasion)

7. **Visual Theme**: Overall aesthetic (natural beauty, bold/dramatic, youthful, sophisticated, artistic)

**Important Guidelines:**
- Only extract what you can SEE in the image
- Be specific about colors and visual characteristics
- Focus on visual attributes, not ingredient claims
- Use lowercase with underscores for entities

**Output Format:**
Provide a JSON list of knowledge points:

[
  {{"relation": "product_type", "entity": "lipstick"}},
  {{"relation": "has_color", "entity": "deep_red"}},
  {{"relation": "packaging_style", "entity": "luxury_gold_tube"}},
  {{"relation": "finish_type", "entity": "matte"}},
  {{"relation": "brand_aesthetic", "entity": "high_end_luxury"}},
  {{"relation": "occasion", "entity": "evening_glamour"}}
]

**Output only the JSON array, nothing else.**"""

    return prompt


def encode_image_base64(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def extract_with_openai(
    image_path: Path,
    prompt: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> str:
    """Extract using OpenAI API directly."""
    import openai

    client = openai.OpenAI(api_key=api_key)

    # Encode image
    base64_image = encode_image_base64(image_path)

    # Determine image type
    suffix = image_path.suffix.lower()
    media_type = "image/jpeg" if suffix in ['.jpg', '.jpeg'] else "image/png"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content


def extract_beauty_knowledge(
    asin: str,
    recbole_id: int,
    title: str,
    categories: str,
    image_path: Path,
    mllm=None,
    api_key: str = None
) -> Dict:
    """Extract knowledge from a single beauty product image."""

    try:
        prompt = create_beauty_extraction_prompt(title, categories)

        # Call MLLM or API
        if mllm is not None:
            response = mllm.extract_from_image(
                image=str(image_path),
                prompt=prompt
            )
        elif api_key:
            response = extract_with_openai(image_path, prompt, api_key)
        else:
            raise ValueError("No MLLM or API key provided")

        # Parse response
        knowledge_points = parse_json_response(response)

        return {
            'asin': asin,
            'recbole_id': recbole_id,
            'title': title,
            'status': 'success',
            'knowledge_points': knowledge_points,
            'raw_response': response
        }

    except Exception as e:
        return {
            'asin': asin,
            'recbole_id': recbole_id,
            'title': title,
            'status': 'error',
            'error': str(e),
            'knowledge_points': []
        }


def load_metadata(metadata_file: Path) -> Dict:
    """Load filtered metadata."""
    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_id_mappings(mapping_file: Path) -> Dict:
    """Load ID mappings."""
    with open(mapping_file, 'r') as f:
        return json.load(f)


def save_results(output_file: Path, config: Dict, results: List[Dict]):
    """Save extraction results."""
    output_data = {
        'phase': 'phase1_exploration',
        'dataset': 'amazon-beauty',
        'config': config,
        'results': results,
        'statistics': {
            'total': len(results),
            'success': sum(1 for r in results if r['status'] == 'success'),
            'error': sum(1 for r in results if r['status'] == 'error')
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Beauty Phase 1 Knowledge Extraction')
    parser.add_argument('--data_dir', type=str,
                        default='/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-beauty',
                        help='Data directory')
    parser.add_argument('--output', type=str,
                        default='/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-beauty/phase1_results.json',
                        help='Output file')
    parser.add_argument('--sample_ratio', type=float, default=0.05,
                        help='Sample ratio (default: 0.05 = 5%)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model to use')
    parser.add_argument('--max_workers', type=int, default=5,
                        help='Max parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key and not HAS_MLLM:
        print("Error: No API key provided. Set --api_key or OPENAI_API_KEY env var")
        return

    print("="*70)
    print("Phase 1: Amazon Beauty Knowledge Extraction")
    print("="*70)

    data_dir = Path(args.data_dir)

    # Load data
    print("\nLoading data...")
    metadata = load_metadata(data_dir / "filtered_metadata.json")
    id_mappings = load_id_mappings(data_dir / "id_mappings.json")

    item_to_recbole = id_mappings['item']['original_to_recbole']

    print(f"  Total items: {len(metadata):,}")

    # Sample items
    random.seed(args.seed)
    all_asins = list(metadata.keys())
    sample_size = int(len(all_asins) * args.sample_ratio)
    sampled_asins = random.sample(all_asins, sample_size)

    print(f"  Sample size ({args.sample_ratio*100:.0f}%): {sample_size:,}")

    # Check images
    image_dir = data_dir / "images"
    items_to_process = []

    for asin in sampled_asins:
        image_path = image_dir / f"{asin}.jpg"
        if image_path.exists():
            meta = metadata[asin]
            title = meta.get('title', f'Beauty_{asin}')
            categories = '|'.join(meta.get('category', []) or meta.get('categories', []) or ['Unknown'])
            recbole_id = item_to_recbole.get(asin)

            if recbole_id:
                items_to_process.append({
                    'asin': asin,
                    'recbole_id': int(recbole_id) if isinstance(recbole_id, str) else recbole_id,
                    'title': title,
                    'categories': categories,
                    'image_path': image_path
                })

    print(f"  Items with images: {len(items_to_process):,}")

    # Create MLLM if available
    mllm = None
    if HAS_MLLM:
        try:
            mllm = create_mllm(model=args.model)
        except Exception as e:
            print(f"  Warning: Could not create MLLM: {e}")

    # Extract knowledge
    print(f"\nExtracting knowledge (model: {args.model})...")
    results = []

    # Use tqdm for progress
    for item in tqdm(items_to_process, desc="Extracting"):
        result = extract_beauty_knowledge(
            asin=item['asin'],
            recbole_id=item['recbole_id'],
            title=item['title'],
            categories=item['categories'],
            image_path=item['image_path'],
            mllm=mllm,
            api_key=api_key
        )
        results.append(result)

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    config = {
        'sample_ratio': args.sample_ratio,
        'model': args.model,
        'seed': args.seed
    }
    save_results(output_file, config, results)

    # Print summary
    success = sum(1 for r in results if r['status'] == 'success')
    errors = sum(1 for r in results if r['status'] == 'error')

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Total processed: {len(results):,}")
    print(f"  Success: {success:,} ({success/len(results)*100:.1f}%)")
    print(f"  Errors: {errors:,}")
    print(f"  Output: {output_file}")

    # Analyze relations
    if success > 0:
        all_relations = {}
        for r in results:
            if r['status'] == 'success':
                for kp in r['knowledge_points']:
                    rel = kp.get('relation', 'unknown')
                    all_relations[rel] = all_relations.get(rel, 0) + 1

        print(f"\nRelation types found ({len(all_relations)}):")
        for rel, count in sorted(all_relations.items(), key=lambda x: -x[1])[:15]:
            print(f"  {rel}: {count}")

    print(f"{'='*70}")


if __name__ == '__main__':
    import os
    main()
