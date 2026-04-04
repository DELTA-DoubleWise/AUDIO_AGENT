#!/usr/bin/env python3
"""
Minimal FireRedVAD inference test script.
Usage: python test_vad.py <audio_path>
"""
import sys
sys.path.insert(0, '/lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/audio_agent/tools/catalog/fireredvad')
from model import ModelWrapper

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_vad.py <audio_path>")
        print("Example: python test_vad.py /path/to/audio.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    wrapper = ModelWrapper()
    result = wrapper.predict(audio_path)
    print(f"FireRedVAD Result: {result.to_json()}")
