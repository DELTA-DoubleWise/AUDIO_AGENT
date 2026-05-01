#!/usr/bin/env python3
"""
Download MMAR dataset metadata and audio files.

The MMAR dataset audio files are stored in a tar.gz archive (~3GB).
This script downloads and extracts them.

Usage:
    # Download metadata only (audio downloaded on-demand during benchmark)
    python -m benchmarking.mmar.download --output-dir /lihaoyu/datasets/MMAR
    
    # Download metadata + all audio files (~3GB)
    python -m benchmarking.mmar.download --output-dir /lihaoyu/datasets/MMAR --download-audio
    
    # Download subset for testing
    python -m benchmarking.mmar.download --output-dir /lihaoyu/datasets/MMAR --download-audio --max-audio-files 100
"""

import argparse
import json
import tarfile
from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm


def download_mmar(
    output_dir: str,
    cache_dir: str | None = None,
    download_audio: bool = False,
    max_audio_files: int | None = None,
):
    """
    Download MMAR dataset metadata.
    
    Args:
        output_dir: Directory to save the dataset.
        cache_dir: Cache directory for HuggingFace.
        download_audio: Whether to download audio files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  MMAR Dataset Download")
    print("=" * 60)
    print()
    
    print(f"Output directory: {output_path}")
    
    # Download test split metadata
    print("\nLoading MMAR dataset from HuggingFace...")
    dataset = load_dataset(
        "BoJack/MMAR",
        split="test",
        cache_dir=cache_dir,
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create metadata directory
    metadata_dir = output_path / "metadata"
    
    # Save in HuggingFace format
    print(f"\nSaving metadata to: {metadata_dir}")
    dataset.save_to_disk(str(metadata_dir))
    print("Saved HuggingFace datasets format")
    
    # Also save as JSON for easy inspection
    json_path = output_path / "metadata.json"
    print(f"Saving JSON to: {json_path}")
    data = [dict(item) for item in dataset]
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved JSON format")
    
    # Print dataset statistics
    modalities = {}
    categories = {}
    sub_categories = {}
    languages = {}
    
    for sample in dataset:
        mod = sample.get("modality", "unknown")
        cat = sample.get("category", "unknown")
        sub = sample.get("sub-category", "unknown")
        lang = sample.get("language", "unknown")
        
        modalities[mod] = modalities.get(mod, 0) + 1
        categories[cat] = categories.get(cat, 0) + 1
        sub_categories[sub] = sub_categories.get(sub, 0) + 1
        languages[lang] = languages.get(lang, 0) + 1
    
    print("\nDataset Statistics:")
    print("-" * 40)
    print(f"Total samples: {len(dataset)}")
    
    print("\nBy Modality:")
    for mod, count in sorted(modalities.items()):
        print(f"  {mod}: {count}")
    
    print("\nBy Category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print("\nBy Sub-Category:")
    for sub, count in sorted(sub_categories.items()):
        print(f"  {sub}: {count}")
    
    print("\nBy Language:")
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count}")
    
    # Create audio directory
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    # Download and extract audio files if requested
    if download_audio:
        print("\n" + "=" * 60)
        print("Downloading audio files...")
        print("=" * 60)
        print("Note: Audio files are in mmar-audio.tar.gz (~3GB)")
        print("This will take significant time and disk space.")
        print("=" * 60)
        
        tar_path = output_path / "mmar-audio.tar.gz"
        
        # Download tar.gz if not exists
        if not tar_path.exists():
            print(f"\nDownloading mmar-audio.tar.gz from HuggingFace...")
            print("Using: hf download BoJack/MMAR mmar-audio.tar.gz")
            
            # Use hf CLI which handles authentication better
            import subprocess
            try:
                result = subprocess.run(
                    [
                        "hf", "download", "BoJack/MMAR", "mmar-audio.tar.gz",
                        "--repo-type", "dataset",
                        "--local-dir", str(output_path)
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"Downloaded to: {tar_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading mmar-audio.tar.gz: {e}")
                print("\nMake sure you are authenticated with HuggingFace:")
                print("  hf auth login")
                print("Or set HF_TOKEN environment variable.")
                print("\nYou can also manually download from:")
                print("  https://huggingface.co/datasets/BoJack/MMAR/tree/main")
                return dataset
            except FileNotFoundError:
                print("Error: 'hf' CLI not found. Please install it:")
                print("  pip install huggingface-hub[cli]")
                print("\nAlternatively, manually download:")
                print("  hf download BoJack/MMAR mmar-audio.tar.gz --repo-type dataset")
                return dataset
        else:
            print(f"\nmmar-audio.tar.gz already exists at: {tar_path}")
        
        # Extract audio files
        print(f"\nExtracting audio files to: {audio_dir}")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar_ref:
                # Get list of files in tar
                file_list = tar_ref.getmembers()
                wav_files = [f for f in file_list if f.name.endswith('.wav')]
                
                if max_audio_files:
                    wav_files = wav_files[:max_audio_files]
                    print(f"Extracting first {max_audio_files} audio files...")
                else:
                    print(f"Extracting all {len(wav_files)} audio files...")
                
                # Extract files
                for wav_file in tqdm(wav_files, desc="Extracting"):
                    # Check if already exists
                    target_path = audio_dir / Path(wav_file.name).name
                    if target_path.exists():
                        continue
                    
                    # Extract file
                    tar_ref.extract(wav_file, audio_dir)
                
                print(f"\nExtraction complete!")
                print(f"Audio files saved to: {audio_dir}")
                
        except Exception as e:
            print(f"Error extracting mmar-audio.tar.gz: {e}")
            print("You may need to manually extract the file.")
    else:
        print("\n" + "=" * 60)
        print("Skipping audio file download.")
        print("Audio files will be downloaded on-demand during benchmarking")
        print("from the mmar-audio.tar.gz on HuggingFace (slower, requires internet).")
        print("=" * 60)
        print(f"\nTo download all audio files for offline use, run:")
        print(f"  python -m benchmarking.mmar.download --download-audio")
    
    print(f"\nDataset ready at: {output_path}")
    print(f"  - metadata/: HuggingFace datasets format")
    print(f"  - metadata.json: JSON format for inspection")
    print(f"  - audio/: Audio files (extracted if --download-audio)")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download MMAR dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/lihaoyu/datasets/MMAR",
        help="Directory to save dataset (default: /lihaoyu/datasets/MMAR)",
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
        help="Download and extract audio files (~3GB)",
    )
    parser.add_argument(
        "--max-audio-files",
        type=int,
        default=None,
        help="Maximum number of audio files to extract (for testing)",
    )
    
    args = parser.parse_args()
    
    download_mmar(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        download_audio=args.download_audio,
        max_audio_files=args.max_audio_files,
    )


if __name__ == "__main__":
    main()
