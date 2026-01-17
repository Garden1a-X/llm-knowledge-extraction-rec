#!/usr/bin/env python3
"""
Direct API call debugging - shows exactly what's being sent
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI


def encode_image(image):
    """Encode PIL image to base64."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def main():
    parser = argparse.ArgumentParser(description='Direct OpenAI API test')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--base_url', type=str, default=None)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    args = parser.parse_args()

    print("="*70)
    print("Direct OpenAI API Call Test")
    print("="*70)
    print(f"\nAPI Key: {args.api_key}")
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print()

    # Create OpenAI client
    print("Creating OpenAI client...")
    client_kwargs = {'api_key': args.api_key}
    if args.base_url:
        client_kwargs['base_url'] = args.base_url

    client = OpenAI(**client_kwargs)
    print(f"✓ Client created")
    print(f"  Client base_url: {client.base_url}")
    print(f"  Client api_key: {client.api_key}")
    print()

    # Create test image
    print("Creating test image...")
    test_image = Image.new('RGB', (100, 100), color='white')
    image_b64 = encode_image(test_image)
    print(f"✓ Image encoded (base64 length: {len(image_b64)})")
    print()

    # Prepare messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image? Reply in one word."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        }
    ]

    print("Calling API...")
    print(f"  Endpoint: {client.base_url}/chat/completions")
    print(f"  Model: {args.model}")
    print(f"  Temperature: 0.0")
    print(f"  Max tokens: 50")
    print()

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )

        print("="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nResponse:")
        print(f"  Model: {response.model}")
        print(f"  Content: {response.choices[0].message.content}")
        print(f"  Tokens: {response.usage.total_tokens}")
        print()
        return 0

    except Exception as e:
        print("="*70)
        print("ERROR!")
        print("="*70)
        print(f"\nException type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print()

        # Show full exception details
        import traceback
        print("Full traceback:")
        print("-"*70)
        traceback.print_exc()
        print()
        return 1


if __name__ == '__main__':
    exit(main())
