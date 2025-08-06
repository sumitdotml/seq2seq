#!/usr/bin/env python3
"""
Download pre-trained seq2seq model and tokenizers from Hugging Face Hub
This allows users to try inference without training from scratch
"""

from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url, local_path, description="Downloading"):
    """Download a file with progress bar"""

    # Create directory if it doesn't exist
    local_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(local_path, "wb") as file, tqdm(
        desc=description,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            pbar.update(size)

    print(f"✓ Downloaded: {local_path}")


def download_pretrained_model():
    """Download pre-trained model and tokenizers"""

    print("=" * 60)
    print("Downloading Pre-trained Seq2Seq German-English Model")
    print("=" * 60)
    print()

    base_url = (
        "https://huggingface.co/sumitdotml/seq2seq-de-en/resolve/main"
    )

    # Files to download
    files_to_download = [
        {
            "url": f"{base_url}/best_model.pt",
            "path": Path("checkpoints/best_model.pt"),
            "description": "Best model checkpoint",
        },
        {
            "url": f"{base_url}/german_tokenizer.pkl",
            "path": Path("data/german_tokenizer.pkl"),
            "description": "German tokenizer",
        },
        {
            "url": f"{base_url}/english_tokenizer.pkl",
            "path": Path("data/english_tokenizer.pkl"),
            "description": "English tokenizer",
        },
    ]

    # Check if files already exist
    existing_files = []
    missing_files = []

    for file_info in files_to_download:
        if file_info["path"].exists():
            existing_files.append(file_info["path"])
        else:
            missing_files.append(file_info)

    if existing_files:
        print("Found existing files:")
        for path in existing_files:
            print(f"  ✓ {path}")
        print()

        if not missing_files:
            print("All files already downloaded! Ready to run inference.")
            return

        response = input("Download missing files? (y/n): ").lower().strip()
        if response != "y":
            print("Skipping download.")
            return

    # Download missing files
    print(f"Downloading {len(missing_files)} files...")
    print()

    for file_info in missing_files:
        try:
            download_file(file_info["url"], file_info["path"], file_info["description"])
        except requests.RequestException as e:
            print(f"❌ Failed to download {file_info['path']}: {e}")
            print("\nPlease check:")
            print("1. Internet connection")
            print("2. Hugging Face repository URL is correct")
            print("3. Files exist in the repository")
            return

    print()
    print("=" * 60)
    print("✅ Download Complete!")
    print("=" * 60)
    print()
    print("You can now run inference:")
    print("  python scripts/inference.py")
    print("  python scripts/inference.py --interactive")
    print("  python scripts/inference.py --sentence 'Hallo Welt!'")


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ["torch", "requests", "tqdm"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print()
        print("Install with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download pre-trained seq2seq model")
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if files exist"
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        exit(1)

    # Force re-download by removing existing files
    if args.force:
        files_to_remove = [
            Path("checkpoints/best_model.pt"),
            Path("data/german_tokenizer.pkl"),
            Path("data/english_tokenizer.pkl"),
        ]

        for path in files_to_remove:
            if path.exists():
                path.unlink()
                print(f"Removed: {path}")
        print()

    download_pretrained_model()
