#!/usr/bin/env python3
"""
Download and prepare MMAU test-mini dataset.

The MMAU dataset audio files are embedded as binary data inside Parquet files.
This script downloads the metadata, extracts audio to disk, and saves a
metadata-only dataset for fast loading during benchmarking.

Usage:
    # Download metadata + extract all audio files
    python -m benchmarking.mmau.download --output-dir /lihaoyu/datasets/MMAU

    # Download metadata only (audio extracted on-demand during benchmark)
    python -m benchmarking.mmau.download --output-dir /lihaoyu/datasets/MMAU --skip-audio

    # Download subset for testing
    python -m benchmarking.mmau.download --output-dir /lihaoyu/datasets/MMAU --max-audio-files 100
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def download_mmau(
    output_dir: str,
    cache_dir: str | None = None,
    skip_audio: bool = False,
    max_audio_files: int | None = None,
):
    """
    Download and prepare MMAU test-mini dataset.

    Args:
        output_dir: Directory to save the dataset.
        cache_dir: Cache directory for HuggingFace Hub downloads.
        skip_audio: If True, do not extract audio files.
        max_audio_files: Maximum number of audio files to extract (for testing).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MMAU Dataset Download")
    print("=" * 60)
    print()
    print(f"Output directory: {output_path}")

    # Discover test-mini parquet files from HuggingFace Hub
    print("\nDiscovering test-mini parquet files on HuggingFace Hub...")
    from huggingface_hub import list_repo_files, hf_hub_download

    files = list(list_repo_files("lmms-lab/mmau", repo_type="dataset"))
    mini_files = sorted([f for f in files if "test_mini" in f])
    print(f"Found {len(mini_files)} parquet files")

    # Download and read all parquet files
    all_tables = []
    for f in tqdm(mini_files, desc="Downloading parquet files"):
        path = hf_hub_download(
            repo_id="lmms-lab/mmau",
            filename=f,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        table = pq.read_table(path)
        all_tables.append(table)

    # Concatenate into one table
    full_table = pa.concat_tables(all_tables)
    total_samples = len(full_table)
    print(f"\nLoaded {total_samples} samples")

    # Prepare directories
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    metadata_dir = output_path / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    # Extract text columns
    text_columns = [
        "id", "question", "choices", "answer",
        "dataset", "task", "split", "category", "sub-category", "difficulty"
    ]

    # Extract audio files
    audio_paths: list[str] = []

    if not skip_audio:
        print("\nExtracting audio files...")
        num_to_extract = max_audio_files if max_audio_files else total_samples

        for i in tqdm(range(num_to_extract), desc="Extracting audio"):
            audio_id = full_table.column("id")[i].as_py()
            audio_path = audio_dir / f"{audio_id}.wav"

            if not audio_path.exists():
                audio_struct = full_table.column("audio")[i].as_py()
                wav_bytes = audio_struct["bytes"]
                with open(audio_path, "wb") as f:
                    f.write(wav_bytes)

            audio_paths.append(str(audio_path))

        # Pad remaining paths if max_audio_files was set
        for i in range(num_to_extract, total_samples):
            audio_id = full_table.column("id")[i].as_py()
            audio_paths.append(str(audio_dir / f"{audio_id}.wav"))
    else:
        for i in range(total_samples):
            audio_id = full_table.column("id")[i].as_py()
            audio_paths.append(str(audio_dir / f"{audio_id}.wav"))

    # Build metadata table with audio_path column
    metadata_table = full_table.select(text_columns)
    metadata_table = metadata_table.append_column(
        "audio_path",
        pa.array(audio_paths, type=pa.string())
    )

    # Save as JSON for inspection
    json_path = output_path / "metadata.json"
    print(f"\nSaving metadata JSON to: {json_path}")
    df = metadata_table.to_pandas()
    records = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Save as HuggingFace datasets format for fast loading
    print(f"Saving metadata dataset to: {metadata_dir}")
    from datasets import Dataset

    metadata_dataset = Dataset.from_pandas(df)
    metadata_dataset.save_to_disk(str(metadata_dir))

    # Print dataset statistics
    print("\nDataset Statistics:")
    print("-" * 40)
    print(f"Total samples: {total_samples}")

    def _print_breakdown(column_name: str, label: str):
        values = df[column_name].value_counts().to_dict()
        print(f"\nBy {label}:")
        for value, count in sorted(values.items(), key=lambda x: x[1], reverse=True):
            print(f"  {value}: {count}")

    _print_breakdown("category", "Category")
    _print_breakdown("sub-category", "Sub-Category")
    _print_breakdown("task", "Task")
    _print_breakdown("difficulty", "Difficulty")
    _print_breakdown("dataset", "Source Dataset")

    print(f"\nDataset ready at: {output_path}")
    print(f"  - metadata/: HuggingFace datasets format")
    print(f"  - metadata.json: JSON format for inspection")
    print(f"  - audio/: Audio files ({len([p for p in audio_paths if Path(p).exists()])} extracted)")

    if skip_audio:
        print("\nAudio files were skipped. They will be extracted on-demand during benchmarking.")
        print("To pre-extract all audio files, run:")
        print("  python -m benchmarking.mmau.download --output-dir /lihaoyu/datasets/MMAU")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare MMAU test-mini dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/lihaoyu/datasets/MMAU",
        help="Directory to save dataset (default: /lihaoyu/datasets/MMAU)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace Hub downloads",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip extracting audio files (extract on-demand during benchmarking)",
    )
    parser.add_argument(
        "--max-audio-files",
        type=int,
        default=None,
        help="Maximum number of audio files to extract (for testing)",
    )

    args = parser.parse_args()

    download_mmau(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        skip_audio=args.skip_audio,
        max_audio_files=args.max_audio_files,
    )


if __name__ == "__main__":
    main()
