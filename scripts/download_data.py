"""
Download MovieLens 1M dataset with posters from GitHub repository.

This script downloads the ml-1m dataset with movie posters from:
https://github.com/11Li11/Li/tree/master/ml-1m
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))


def clone_ml1m_data():
    """
    Clone the ml-1m dataset with posters from GitHub repository.
    """
    print("=" * 60)
    print("Downloading MovieLens 1M Dataset with Posters")
    print("=" * 60)

    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    ml1m_dir = data_dir / "ml-1m"

    # Check if data already exists
    if ml1m_dir.exists() and (ml1m_dir / "ratings.dat").exists():
        print(f"\n✓ Dataset already exists at: {ml1m_dir}")
        print("If you want to re-download, please delete the directory first.")
        return

    print(f"\nTarget directory: {data_dir}")
    print("\nDownloading from: https://github.com/11Li11/Li")

    # Use git sparse-checkout to download only the ml-1m folder
    try:
        os.chdir(data_dir)

        # Initialize a new git repo
        if not (data_dir / ".git").exists():
            print("\n[1/4] Initializing git repository...")
            subprocess.run(["git", "init"], check=True, capture_output=True)

            print("[2/4] Adding remote...")
            subprocess.run(
                ["git", "remote", "add", "-f", "origin", "https://github.com/11Li11/Li.git"],
                check=True,
                capture_output=True
            )

            print("[3/4] Configuring sparse checkout...")
            subprocess.run(["git", "config", "core.sparseCheckout", "true"], check=True, capture_output=True)

            # Specify the ml-1m folder
            sparse_checkout_file = data_dir / ".git" / "info" / "sparse-checkout"
            sparse_checkout_file.parent.mkdir(parents=True, exist_ok=True)
            with open(sparse_checkout_file, 'w') as f:
                f.write("ml-1m/*\n")

            print("[4/4] Downloading ml-1m folder...")
            subprocess.run(["git", "pull", "origin", "master"], check=True)

        print("\n✓ Dataset downloaded successfully!")

        # Verify the downloaded files
        print("\nVerifying downloaded files:")
        required_files = ["ratings.dat", "users.dat", "movies.dat"]
        for file in required_files:
            file_path = ml1m_dir / file
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"  ✓ {file}: {file_size / 1024:.2f} KB")
            else:
                print(f"  ✗ {file}: NOT FOUND")

        # Check posters directory
        posters_dir = ml1m_dir / "posters"
        if posters_dir.exists():
            num_posters = len(list(posters_dir.glob("*.jpg")))
            print(f"  ✓ posters/: {num_posters} poster images")
        else:
            print(f"  ✗ posters/: NOT FOUND")

        print("\n" + "=" * 60)
        print("Dataset Information:")
        print("=" * 60)
        print("MovieLens 1M Dataset contains:")
        print("  - 6,040 users")
        print("  - ~3,900 movies")
        print("  - 1,000,209 ratings")
        print("  - Movie posters (if available)")
        print("\nData location:", ml1m_dir)
        print("=" * 60)

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("Please make sure git is installed and you have internet connection.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
    finally:
        os.chdir(project_root)


def create_gitkeep_files():
    """Create .gitkeep files in empty data directories."""
    data_subdirs = ["raw", "processed", "graphs"]
    for subdir in data_subdirs:
        dir_path = project_root / "data" / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        gitkeep = dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "MovieLens 1M Dataset Downloader" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    create_gitkeep_files()
    clone_ml1m_data()

    print("\n✓ All done! You can now proceed to data processing.")
    print("  Next step: python scripts/process_data.py\n")
