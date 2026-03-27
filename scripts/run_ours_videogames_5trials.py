#!/usr/bin/env python3
"""
Run our method on Video Games dataset for 5 trials.
"""

import sys
import subprocess
import yaml
import tempfile
from pathlib import Path

def main():
    seeds = [42, 2023, 2024, 2025, 12345]
    base_config_file = "configs/ours_full_videogames.yaml"

    print("="*70)
    print("Running Ours-Full on Video Games Dataset (5 trials)")
    print("="*70)
    print(f"Base config: {base_config_file}")
    print(f"Seeds: {seeds}")
    print("="*70)
    print()

    # Load base config
    with open(base_config_file, 'r') as f:
        base_config = yaml.safe_load(f)

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'#'*70}")
        print(f"# Trial {i}/5: seed={seed}")
        print(f"{'#'*70}\n")

        # Create temporary config with modified seed
        trial_config = base_config.copy()
        trial_config['train']['random_seed'] = seed

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(trial_config, f)
            temp_config_path = f.name

        try:
            cmd = [
                sys.executable,
                "scripts/train_model.py",
                "--config", temp_config_path
            ]

            result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

            if result.returncode != 0:
                print(f"\n✗ Trial {i} (seed={seed}) failed!")
            else:
                print(f"\n✓ Trial {i} (seed={seed}) completed!")
        finally:
            # Clean up temp file
            Path(temp_config_path).unlink(missing_ok=True)

    print("\n" + "="*70)
    print("All trials completed!")
    print("="*70)

if __name__ == '__main__':
    main()
