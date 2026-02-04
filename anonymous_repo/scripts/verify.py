#!/usr/bin/env python3
"""
Verify that the anonymous repo can run training end-to-end.
Runs 2 epochs on each dataset as a smoke test.

Usage (from anonymous_repo/):
    python scripts/verify.py
"""

import sys
import os
from pathlib import Path

# Ensure we're running from the anonymous_repo root
repo_root = Path(__file__).parent.parent
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))

import yaml
import tempfile
import subprocess


def check_files():
    """Check all required files exist."""
    print("=" * 60)
    print("Step 1: Checking files")
    print("=" * 60)

    required_code = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/graph_builder.py",
        "src/data/dataset.py",
        "src/model/__init__.py",
        "src/model/ours.py",
        "src/model/encoders.py",
        "src/model/losses.py",
        "src/utils/__init__.py",
        "src/utils/config.py",
        "src/utils/metrics.py",
        "scripts/train_model.py",
    ]

    required_data = {
        "ml1m": [
            "data/ml-1m/ml-1m.inter",
            "data/ml-1m/ml-1m.item.kg",
            "data/ml-1m/ml-1m.user.kg",
        ],
        "beauty": [
            "data/amazon-beauty/amazon-beauty.inter",
            "data/amazon-beauty/amazon-beauty.item.kg",
            "data/amazon-beauty/amazon-beauty.user.kg",
        ],
        "videogames": [
            "data/amazon-videogames/amazon-videogames.inter",
            "data/amazon-videogames/amazon-videogames.item.kg",
            "data/amazon-videogames/amazon-videogames.user.kg",
        ],
    }

    ok = True

    # Check code
    for f in required_code:
        exists = Path(f).exists()
        status = "OK" if exists else "MISSING"
        if not exists:
            ok = False
        print(f"  [{status}] {f}")

    # Check data
    available_datasets = []
    for dataset, files in required_data.items():
        all_exist = all(Path(f).exists() for f in files)
        if all_exist:
            available_datasets.append(dataset)
        for f in files:
            exists = Path(f).exists()
            status = "OK" if exists else "MISSING"
            if not exists:
                ok = False
            print(f"  [{status}] {f}")

    print(f"\nAvailable datasets: {available_datasets}")
    return ok, available_datasets


def run_smoke_test(dataset):
    """Run 2-epoch training as smoke test."""
    config_path = f"configs/{dataset}.yaml"
    print(f"\n{'=' * 60}")
    print(f"Step 2: Smoke test - {dataset} (2 epochs)")
    print(f"{'=' * 60}")

    # Create a temp config with 2 epochs and early stop disabled
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['train']['num_epochs'] = 2
    config['train']['early_stop_patience'] = 999
    config['train']['eval_every'] = 1
    config['train']['save_every'] = 999
    config['train']['device'] = 'cuda'

    tmp_config = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False, dir='/tmp'
    )
    yaml.dump(config, tmp_config)
    tmp_config.close()

    try:
        result = subprocess.run(
            ['python', 'scripts/train_model.py', '--config', tmp_config.name],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode == 0:
            # Extract final metrics
            for line in result.stdout.split('\n'):
                if 'NDCG@10' in line or 'Recall@10' in line or 'Training Complete' in line:
                    print(f"  {line.strip()}")
            print(f"\n  [PASS] {dataset} training completed successfully!")
            return True
        else:
            print(f"  [FAIL] {dataset} training failed!")
            # Print last 20 lines of output
            lines = (result.stdout + result.stderr).strip().split('\n')
            for line in lines[-20:]:
                print(f"  {line}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] {dataset} timed out (>300s)")
        return False
    finally:
        os.unlink(tmp_config.name)


def main():
    print("Anonymous Repo Verification")
    print(f"Working directory: {os.getcwd()}\n")

    ok, datasets = check_files()

    if not ok:
        print("\nSome files are missing. Fix before proceeding.")
        return

    results = {}
    for ds in datasets:
        results[ds] = run_smoke_test(ds)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for ds, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {ds}")

    if all(results.values()):
        print("\nAll smoke tests passed!")
    else:
        failed = [ds for ds, p in results.items() if not p]
        print(f"\nFailed: {failed}")


if __name__ == '__main__':
    main()
