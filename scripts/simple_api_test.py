#!/usr/bin/env python3
"""
Simple API test - copied from mllm_interface.py example usage
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
from PIL import Image

from src.extraction.mllm_interface import create_mllm
from src.extraction.prompts import PromptTemplates


def main():
    parser = argparse.ArgumentParser(description='Test MLLM interface')
    parser.add_argument('--backend', type=str, default='openai',
                        choices=['openai', 'local'],
                        help='MLLM backend')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name')
    parser.add_argument('--image', type=str, required=False,
                        help='Path to test image (optional, will create white image if not provided)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (if using OpenAI)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='Base URL for API')

    args = parser.parse_args()

    # Print configuration
    print("="*70)
    print("API Configuration Test")
    print("="*70)
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"API Key: {args.api_key}")
    print(f"Base URL: {args.base_url}")
    print()

    # Create MLLM
    print(f"Creating {args.backend} MLLM...")
    mllm_kwargs = {
        'backend': args.backend,
        'model_name': args.model,
        'api_key': args.api_key
    }
    if args.base_url:
        mllm_kwargs['base_url'] = args.base_url

    mllm = create_mllm(**mllm_kwargs)
    print("✓ MLLM created")

    # Get or create image
    if args.image:
        image_path = args.image
        print(f"Using image: {image_path}")
    else:
        # Create a simple white test image
        print("Creating test white image...")
        test_image = Image.new('RGB', (256, 256), color='white')
        image_path = test_image

    # Get prompts
    system_prompt = PromptTemplates.get_phase1_system_prompt()
    user_prompt = PromptTemplates.get_phase1_user_prompt()

    # Extract knowledge
    print(f"\nCalling API...")
    start_time = time.time()

    output = mllm.extract_from_image(
        image=image_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=1000
    )

    elapsed = time.time() - start_time

    print(f"✓ API call successful in {elapsed:.2f}s")
    print("\n" + "="*60)
    print("Model Output:")
    print("="*60)
    print(output)

    # Parse output
    knowledge_points = PromptTemplates.parse_extraction_output(output)
    print("\n" + "="*60)
    print(f"Parsed {len(knowledge_points)} knowledge points")
    print("="*60)

    return 0


if __name__ == '__main__':
    exit(main())
