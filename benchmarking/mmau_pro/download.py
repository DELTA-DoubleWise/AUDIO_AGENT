#!/usr/bin/env python3
"""
Download MMAU-Pro dataset including audio files.

The MMAU-Pro dataset audio files are stored in a large data.zip file (~47GB).
This script downloads and extracts them.

⚠️ IMPORTANT: This is a gated dataset. You must:
1. Accept terms at: https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro
2. Login with: huggingface-cli login
   Or set: export HF_TOKEN="your-token"

Usage:
    # Download metadata only (faster, audio downloaded on-demand during benchmark)
    python -m benchmarking.mmau_pro.download --output-dir /lihaoyu/datasets/MMAU-Pro
    
    # Download metadata + all audio files (requires ~47GB disk space)
    python -m benchmarking.mmau_pro.download --output-dir /lihaoyu/datasets/MMAU-Pro --download-audio
    
    # Download subset for testing
    python -m benchmarking.mmau_pro.download --output-dir /lihaoyu/datasets/MMAU-Pro --download-audio --max-audio-files 100
"""

import argparse
import os
import zipfile
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import requests


def download_file(url: str, local_path: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f:
        if total_size > 0:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path.name[:40]) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def download_mmau_pro(
    output_dir: str,
    cache_dir: str | None = None,
    download_audio: bool = False,
    max_audio_files: int | None = None,
):
    """
    Download MMAU-Pro dataset.
    
    Args:
        output_dir: Directory to save the dataset.
        cache_dir: Cache directory for HuggingFace.
        download_audio: Whether to download and extract audio files (requires ~47GB).
        max_audio_files: Maximum number of audio files to keep (for testing).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  MMAU-Pro Dataset Download")
    print("=" * 60)
    print()
    print("IMPORTANT: This is a gated dataset on HuggingFace.")
    print("You must accept terms at:")
    print("  https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro")
    print()
    print("And login with: huggingface-cli login")
    print("Or set: export HF_TOKEN=\"your-token\"")
    print("=" * 60)
    print()
    
    print(f"Output directory: {output_path}")
    
    # Download test split metadata
    print("\nLoading test split metadata...")
    dataset = load_dataset(
        "gamma-lab-umd/MMAU-Pro",
        split="test",
        cache_dir=cache_dir,
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create data directory for audio files
    data_dir = output_path / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Print dataset statistics
    categories = {}
    task_types = {}
    length_types = {}
    
    for sample in dataset:
        cat = sample.get("category", "unknown")
        task = sample.get("task_classification", "unknown")
        length = sample.get("length_type", "unknown")
        
        categories[cat] = categories.get(cat, 0) + 1
        task_types[task] = task_types.get(task, 0) + 1
        length_types[length] = length_types.get(length, 0) + 1
    
    print("\nDataset Statistics:")
    print("-" * 40)
    print(f"Total samples: {len(dataset)}")
    
    print("\nBy Category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print("\nBy Task Type:")
    for task, count in sorted(task_types.items(), key=lambda x: x[0] or ""):
        task_display = task if task else "unknown"
        print(f"  {task_display}: {count}")
    
    print("\nBy Length Type:")
    for length, count in sorted(length_types.items(), key=lambda x: x[0] or ""):
        length_display = length if length else "unknown"
        print(f"  {length_display}: {count}")
    
    # Download and extract audio files if requested
    if download_audio:
        print("\n" + "=" * 60)
        print("Downloading audio files...")
        print("=" * 60)
        print("Note: Audio files are in data.zip (~47GB)")
        print("This will take significant time and disk space.")
        print("=" * 60)
        
        zip_path = output_path / "data.zip"
        
        # Download data.zip if not exists
        if not zip_path.exists():
            url = "https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro/resolve/main/data.zip"
            print(f"\nDownloading data.zip from HuggingFace...")
            print(f"URL: {url}")
            try:
                download_file(url, zip_path)
                print(f"Downloaded to: {zip_path}")
            except Exception as e:
                print(f"Error downloading data.zip: {e}")
                return dataset
        else:
            print(f"\ndata.zip already exists at: {zip_path}")
        
        # Extract audio files
        print(f"\nExtracting audio files to: {data_dir}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files in zip
                file_list = zip_ref.namelist()
                wav_files = [f for f in file_list if f.endswith('.wav')]
                
                if max_audio_files:
                    wav_files = wav_files[:max_audio_files]
                    print(f"Extracting first {max_audio_files} audio files...")
                else:
                    print(f"Extracting all {len(wav_files)} audio files...")
                
                # Extract files
                for wav_file in tqdm(wav_files, desc="Extracting"):
                    # Check if already exists
                    target_path = data_dir / Path(wav_file).name
                    if target_path.exists():
                        continue
                    
                    # Extract file
                    zip_ref.extract(wav_file, output_path)
                    
                    # Move to data directory if nested
                    extracted_path = output_path / wav_file
                    if extracted_path != target_path and extracted_path.exists():
                        extracted_path.rename(target_path)
                        # Clean up empty directories
                        try:
                            (output_path / wav_file.split('/')[0]).rmdir()
                        except:
                            pass
                
                print(f"\nExtraction complete!")
                
                # Remove zip file if requested (optional, for space)
                # if not max_audio_files:
                #     print(f"\nRemoving data.zip to save space...")
                #     zip_path.unlink()
                
        except Exception as e:
            print(f"Error extracting data.zip: {e}")
            print("You may need to manually extract the file.")
    else:
        print("\n" + "=" * 60)
        print("Skipping audio file download.")
        print("Audio files will be downloaded on-demand during benchmarking")
        print("from the data.zip file on HuggingFace (slower, requires internet).")
        print("=" * 60)
        print(f"\nTo download all audio files for offline use, run:")
        print(f"  python -m benchmarking.mmau_pro.download --download-audio")
    
    print(f"\nDataset ready at: {output_path}")
    print(f"Audio files location: {data_dir}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download MMAU-Pro dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/lihaoyu/datasets/MMAU-Pro",
        help="Directory to save dataset (default: /lihaoyu/datasets/MMAU-Pro)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets",
    )
    parser.add_argument(
        "--download-audio",
        action="store_true",
        help="Download and extract audio files (~47GB)",
    )
    parser.add_argument(
        "--max-audio-files",
        type=int,
        default=None,
        help="Maximum number of audio files to extract (for testing)",
    )
    
    args = parser.parse_args()
    
    download_mmau_pro(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        download_audio=args.download_audio,
        max_audio_files=args.max_audio_files,
    )


if __name__ == "__main__":
    main()
