#!/usr/bin/env python3
"""
Run our method on Video Games dataset for 5 trials.
"""

import sys
import subprocess
from pathlib import Path

def main():
    seeds = [42, 2023, 2024, 2025, 12345]
    config_file = "configs/ours_full_videogames.yaml"

    print("="*70)
    print("Running Ours-Full on Video Games Dataset (5 trials)")
    print("="*70)
    print(f"Config: {config_file}")
    print(f"Seeds: {seeds}")
    print("="*70)
    print()

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'#'*70}")
        print(f"# Trial {i}/5: seed={seed}")
        print(f"{'#'*70}\n")

        cmd = [
            sys.executable,
            "scripts/train_model.py",
            "--config", config_file,
            "--seed", str(seed)
        ]

        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

        if result.returncode != 0:
            print(f"\n✗ Trial {i} (seed={seed}) failed!")
        else:
            print(f"\n✓ Trial {i} (seed={seed}) completed!")

    print("\n" + "="*70)
    print("All trials completed!")
    print("="*70)

if __name__ == '__main__':
    main()
