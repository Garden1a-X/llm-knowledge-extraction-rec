#!/usr/bin/env python3
"""
Analyze Phase 3 extraction errors to identify quota issues.
"""

import json
from pathlib import Path
from collections import Counter
import re


def analyze_errors(results_file: Path):
    """Analyze extraction errors."""

    print("="*70)
    print("Phase 3 Error Analysis")
    print("="*70)
    print()

    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data['results']

    # Basic stats
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'success')
    errors = sum(1 for r in results if r['status'] == 'error')

    print(f"Total results: {total:,}")
    print(f"  Successful: {success:,} ({success/total*100:.1f}%)")
    print(f"  Errors: {errors:,} ({errors/total*100:.1f}%)")
    print()

    # Error types
    error_results = [r for r in results if r['status'] == 'error']

    error_types = Counter()
    quota_errors = []
    image_not_found = []
    json_errors = []
    other_errors = []

    for r in error_results:
        error_msg = r.get('error', '')

        if '该月用量已经达到申报限额' in error_msg or '404: 403' in error_msg:
            error_types['API Quota Exceeded'] += 1
            quota_errors.append(r)
        elif 'Image not found' in error_msg:
            error_types['Image Not Found'] += 1
            image_not_found.append(r)
        elif 'JSON parse error' in error_msg:
            error_types['JSON Parse Error'] += 1
            json_errors.append(r)
        else:
            error_types['Other'] += 1
            other_errors.append(r)

    print("Error Type Distribution:")
    print("-"*70)
    for error_type, count in error_types.most_common():
        print(f"  {error_type:<30} {count:>6,} ({count/errors*100:>5.1f}%)")
    print()

    # When did quota errors start?
    if quota_errors:
        print("API Quota Error Timeline:")
        print("-"*70)
        print(f"  First quota error at index: {results.index(quota_errors[0])}")
        print(f"  Last quota error at index: {results.index(quota_errors[-1])}")
        print(f"  Total quota errors: {len(quota_errors):,}")
        print()

        # Show first few quota errors
        print("First 3 quota errors:")
        for i, r in enumerate(quota_errors[:3], 1):
            print(f"  {i}. RecBole ID: {r['recbole_id']}, ASIN: {r['asin']}")
            print(f"     Index: {results.index(r)}")
        print()

        # Successful items before quota
        success_before_quota = sum(
            1 for r in results[:results.index(quota_errors[0])]
            if r['status'] == 'success'
        )
        print(f"Successful extractions before quota hit: {success_before_quota:,}")
        print()

    # Image not found stats
    if image_not_found:
        print(f"Image Not Found: {len(image_not_found):,} items")
        print("  Sample ASINs:")
        for r in image_not_found[:5]:
            print(f"    - {r['asin']}: {r.get('title', 'N/A')[:50]}")
        print()

    # JSON parse errors
    if json_errors:
        print(f"JSON Parse Errors: {len(json_errors):,} items")
        print()

    # Other errors
    if other_errors:
        print(f"Other Errors: {len(other_errors):,} items")
        print("  Sample errors:")
        for r in other_errors[:3]:
            print(f"    - {r.get('error', 'N/A')[:80]}")
        print()

    # Summary
    print("="*70)
    print("Summary")
    print("="*70)
    print()

    if quota_errors:
        first_quota_idx = results.index(quota_errors[0])
        print(f"✓ Extracted successfully: {success:,} / {total:,}")
        print(f"✗ API quota hit at item ~{first_quota_idx:,}")
        print(f"✗ Quota errors: {len(quota_errors):,}")
        print(f"✗ Other errors: {errors - len(quota_errors):,}")
        print()
        print("Recommendation: Contact admin to increase quota, then resume extraction.")
    else:
        print("No quota errors detected.")

    print("="*70)


if __name__ == '__main__':
    results_file = Path("results/videogames/phase3_full_extraction.json")
    analyze_errors(results_file)
