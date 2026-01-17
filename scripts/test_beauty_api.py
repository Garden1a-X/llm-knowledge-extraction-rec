#!/usr/bin/env python3
"""
Debug script to test OpenAI API connection for Beauty extraction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from PIL import Image

from src.extraction.mllm_interface import create_mllm
from src.extraction.prompts import PromptTemplates


def test_api_connection(api_key: str, base_url: str = None, model: str = 'gpt-4o-mini'):
    """Test OpenAI API connection with a simple text-only request."""
    print("="*70)
    print("Testing OpenAI API Connection")
    print("="*70)
    print(f"\nModel: {model}")
    print(f"Base URL: {base_url if base_url else 'None (using default)'}")
    print(f"API Key: {api_key}")
    print(f"API Key length: {len(api_key)} characters")
    print()

    try:
        # Create MLLM
        print("Creating MLLM interface...")
        mllm = create_mllm(
            backend='openai',
            model_name=model,
            api_key=api_key,
            base_url=base_url
        )
        print("✓ MLLM interface created")

        # Test with a simple white image
        print("\nCreating test image...")
        test_image = Image.new('RGB', (256, 256), color='white')
        print("✓ Test image created (256x256 white)")

        # Simple prompts
        system_prompt = "You are a helpful assistant."
        user_prompt = "What color is this image? Reply in one word."

        print("\nCalling API...")
        output = mllm.extract_from_image(
            image=test_image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=50
        )

        print("✓ API call successful!")
        print(f"\nResponse: {output}")
        print("\n" + "="*70)
        print("SUCCESS: API connection is working!")
        print("="*70)
        return True

    except Exception as e:
        print(f"\n✗ API call failed!")
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\n" + "="*70)
        print("FAILED: Please check your API configuration")
        print("="*70)
        print("\nCommon issues:")
        print("  1. Invalid API key")
        print("  2. Incorrect base_url (check if it needs /v1 suffix)")
        print("  3. Network/proxy issues")
        print("  4. Model not available on this endpoint")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test OpenAI API connection')
    parser.add_argument('--api_key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default=None,
                       help='Optional API base URL')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Model name to test')

    args = parser.parse_args()

    success = test_api_connection(args.api_key, args.base_url, args.model)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
