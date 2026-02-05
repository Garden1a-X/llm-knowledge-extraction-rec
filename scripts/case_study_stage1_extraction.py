#!/usr/bin/env python3
"""
Generate Stage 1 (Exploratory Extraction) examples for Case Study.

This script runs the Phase 1 extraction prompt on specific items to demonstrate
the raw, unstandardized MLLM outputs before vocabulary standardization.

Usage:
    python scripts/case_study_stage1_extraction.py

Requirements:
    - OpenAI API key (OPENAI_API_KEY environment variable)
    - Poster images accessible at the configured paths
    - id_mappings.json for each dataset
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import base64
from datetime import datetime

# Check for API key
if not os.environ.get('OPENAI_API_KEY'):
    print("Error: OPENAI_API_KEY environment variable not set")
    print("Please set it: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

from openai import OpenAI
from src.extraction.prompts import PromptTemplates


def encode_image_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def extract_with_phase1_prompt(client: OpenAI, image_path: Path, title: str, model: str = "gpt-4o-mini") -> dict:
    """
    Run Phase 1 (Exploratory) extraction on a single image.

    Args:
        client: OpenAI client
        image_path: Path to the image file
        title: Item title for context
        model: Model to use

    Returns:
        dict with raw_output, knowledge_points, etc.
    """
    system_prompt = PromptTemplates.get_phase1_system_prompt()
    user_prompt = PromptTemplates.get_phase1_user_prompt(title)

    # Encode image
    image_base64 = encode_image_base64(image_path)

    # Determine image type
    suffix = image_path.suffix.lower()
    media_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }.get(suffix, 'image/jpeg')

    # Call API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000,
        temperature=0.0
    )

    raw_output = response.choices[0].message.content
    knowledge_points = PromptTemplates.parse_extraction_output(raw_output)

    return {
        'title': title,
        'image_path': str(image_path),
        'raw_output': raw_output,
        'knowledge_points': knowledge_points,
        'num_knowledge_points': len(knowledge_points),
        'timestamp': datetime.now().isoformat()
    }


def main():
    print("=" * 70)
    print("CASE STUDY: Stage 1 Exploratory Extraction Examples")
    print("=" * 70)

    client = OpenAI()
    results = {}

    # ========== ML-1M: Star Wars ==========
    print("\n" + "-" * 70)
    print("Example 1: ML-1M - Star Wars: Episode IV - A New Hope (1977)")
    print("-" * 70)

    # Configure paths for ML-1M
    ml1m_poster_dir = Path("/data/xuao/llm-knowledge-extraction-rec/data/recbole/ml-1m/posters")
    ml1m_mapping_file = Path("/data/xuao/llm-knowledge-extraction-rec/data/recbole/ml-1m/id_mappings.json")

    if ml1m_mapping_file.exists():
        with open(ml1m_mapping_file, 'r') as f:
            ml1m_mappings = json.load(f)

        # Star Wars: Episode IV has recbole_id = 242
        recbole_id = 242
        original_id = ml1m_mappings['item_id_map']['new_to_original'].get(str(recbole_id))

        if original_id:
            poster_path = ml1m_poster_dir / f"{original_id}.jpg"
            print(f"  RecBole ID: {recbole_id}")
            print(f"  Original Movie ID: {original_id}")
            print(f"  Poster path: {poster_path}")

            if poster_path.exists():
                print("  Extracting...")
                result = extract_with_phase1_prompt(
                    client, poster_path,
                    "Star Wars: Episode IV - A New Hope (1977)"
                )
                results['ml1m_star_wars'] = result

                print(f"\n  Stage 1 Raw Output:")
                print("  " + "-" * 50)
                for kp in result['knowledge_points']:
                    print(f"    ({kp['relation']}, {kp['entity']})")
                print(f"\n  Total: {result['num_knowledge_points']} knowledge points")
            else:
                print(f"  ⚠ Poster not found: {poster_path}")
        else:
            print(f"  ⚠ RecBole ID {recbole_id} not in mapping")
    else:
        print(f"  ⚠ Mapping file not found: {ml1m_mapping_file}")

    # ========== Video Games: Sonic the Hedgehog 2 ==========
    print("\n" + "-" * 70)
    print("Example 2: Video Games - Sonic the Hedgehog 2")
    print("-" * 70)

    # Configure paths for Video Games
    vg_image_dir = Path("/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-videogames/images")
    vg_item_file = Path("/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-videogames/amazon-videogames.item")

    # Find Sonic the Hedgehog 2
    sonic_asin = None
    sonic_title = None
    sonic_recbole_id = None

    if vg_item_file.exists():
        with open(vg_item_file, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    item_id = parts[0]
                    title = parts[1]
                    if 'Sonic' in title and 'Hedgehog 2' in title and 'Sega Genesis' not in title:
                        sonic_recbole_id = item_id
                        # For Amazon, the item file might have ASIN
                        sonic_title = title
                        break

    if sonic_recbole_id:
        print(f"  Found: {sonic_title}")
        print(f"  RecBole ID: {sonic_recbole_id}")

        # Try to find the image (might be named by ASIN or recbole_id)
        # Check phase1 results for the ASIN
        vg_phase1_file = Path("results/videogames/phase1_5percent_exploration.json")
        if vg_phase1_file.exists():
            with open(vg_phase1_file, 'r') as f:
                vg_phase1 = json.load(f)

            for item in vg_phase1.get('results', []):
                if 'Sonic' in item.get('title', '') and 'Hedgehog 2' in item.get('title', ''):
                    sonic_asin = item.get('asin')
                    sonic_title = item.get('title')
                    sonic_recbole_id = item.get('recbole_id')
                    break

        if sonic_asin:
            image_path = vg_image_dir / f"{sonic_asin}.jpg"
            print(f"  ASIN: {sonic_asin}")
            print(f"  Image path: {image_path}")

            if image_path.exists():
                print("  Extracting...")
                result = extract_with_phase1_prompt(client, image_path, sonic_title)
                results['videogames_sonic'] = result

                print(f"\n  Stage 1 Raw Output:")
                print("  " + "-" * 50)
                for kp in result['knowledge_points']:
                    print(f"    ({kp['relation']}, {kp['entity']})")
                print(f"\n  Total: {result['num_knowledge_points']} knowledge points")
            else:
                print(f"  ⚠ Image not found: {image_path}")
    else:
        print("  ⚠ Sonic the Hedgehog 2 not found in item file")

    # ========== Beauty: Lipstick ==========
    print("\n" + "-" * 70)
    print("Example 3: Beauty - NYX Lipstick (Alternative)")
    print("-" * 70)

    beauty_image_dir = Path("/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-beauty/images")
    beauty_phase1_file = Path("results/beauty/phase1_results.json")

    if beauty_phase1_file.exists():
        with open(beauty_phase1_file, 'r') as f:
            beauty_phase1 = json.load(f)

        # Find lipstick example
        for item in beauty_phase1.get('results', []):
            if 'lipstick' in item.get('title', '').lower() and item.get('status') == 'success':
                asin = item.get('asin')
                title = item.get('title')
                recbole_id = item.get('recbole_id')

                image_path = beauty_image_dir / f"{asin}.jpg"
                print(f"  Found: {title[:60]}...")
                print(f"  ASIN: {asin}")
                print(f"  Image path: {image_path}")

                if image_path.exists():
                    print("  Extracting...")
                    result = extract_with_phase1_prompt(client, image_path, title)
                    results['beauty_lipstick'] = result

                    print(f"\n  Stage 1 Raw Output:")
                    print("  " + "-" * 50)
                    for kp in result['knowledge_points']:
                        print(f"    ({kp['relation']}, {kp['entity']})")
                    print(f"\n  Total: {result['num_knowledge_points']} knowledge points")
                else:
                    print(f"  ⚠ Image not found: {image_path}")
                break

    # ========== Save Results ==========
    output_file = Path("results/case_study_stage1_extraction.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70)

    # ========== Print Summary for Paper ==========
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER (Copy-paste ready)")
    print("=" * 70)

    for key, result in results.items():
        print(f"\n### {result['title']}")
        print("\nStage 1 (Exploratory) Raw Knowledge Points:")
        for i, kp in enumerate(result['knowledge_points'], 1):
            print(f"  {i}. ({kp['relation']}, {kp['entity']})")


if __name__ == '__main__':
    main()
