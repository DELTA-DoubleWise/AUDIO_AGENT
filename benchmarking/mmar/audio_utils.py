"""
Audio preprocessing utilities for MMAR benchmark.

Handles downsampling and downmixing of multi-channel high-sample-rate audio
to mono 16kHz WAV, which is the standard format expected by most audio
understanding models and API endpoints.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional


def preprocess_audio_path(
    audio_path: str,
    cache_dir: Optional[str] = None,
    target_sr: int = 16000,
    target_channels: int = 1,
    verbose: bool = True,
) -> str:
    """
    Preprocess an audio file to target format (mono, target sample rate).

    Uses librosa to load and resample, and soundfile to write the output.
    Results are cached to avoid re-processing the same file.

    Args:
        audio_path: Path to the original audio file.
        cache_dir: Directory to cache preprocessed audio files. If None,
            uses ``<original_dir>/.preprocessed_audio``.
        target_sr: Target sampling rate (default: 16000).
        target_channels: Target number of channels (default: 1).
        verbose: Whether to print preprocessing info.

    Returns:
        Path to the preprocessed audio file.
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(
            "Audio preprocessing requires librosa and soundfile. "
            "Install with: uv pip install librosa soundfile"
        ) from e

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Determine cache directory
    if cache_dir:
        cache_root = Path(cache_dir)
    else:
        cache_root = path.parent / ".preprocessed_audio"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Unique cache key based on path + params
    params_str = f"{os.path.abspath(audio_path)}:{target_sr}:{target_channels}"
    cache_key = hashlib.md5(params_str.encode()).hexdigest()
    cache_path = cache_root / f"{cache_key}_{path.stem}.wav"

    if cache_path.exists():
        return str(cache_path)

    # Load and preprocess
    audio, sr = librosa.load(
        audio_path,
        sr=target_sr,
        mono=(target_channels == 1),
    )

    sf.write(str(cache_path), audio, target_sr)

    if verbose:
        original_size = path.stat().st_size
        processed_size = cache_path.stat().st_size
        print(
            f"[AudioPreprocess] {path.name}: "
            f"{original_size / 1024 / 1024:.2f}MB -> {processed_size / 1024 / 1024:.2f}MB "
            f"({target_sr}Hz/{target_channels}ch)"
        )

    return str(cache_path)


def preprocess_audio_paths(
    audio_paths: list[str],
    cache_dir: Optional[str] = None,
    target_sr: int = 16000,
    target_channels: int = 1,
    verbose: bool = True,
) -> list[str]:
    """
    Preprocess multiple audio files.

    Args:
        audio_paths: List of paths to original audio files.
        cache_dir: Directory to cache preprocessed audio files.
        target_sr: Target sampling rate.
        target_channels: Target number of channels.
        verbose: Whether to print preprocessing info.

    Returns:
        List of paths to preprocessed audio files.
    """
    return [
        preprocess_audio_path(p, cache_dir, target_sr, target_channels, verbose)
        for p in audio_paths
    ]
