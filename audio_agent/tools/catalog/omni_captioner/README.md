# Omni-Captioner Tool

Audio captioning tool using Qwen3-Omni-30B-A3B-Captioner model for detailed audio description.

## ⚠️ Setup Required

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Download Model
```bash
audio-agent-download-models --models omni-captioner
```

**Note**: This is a large model (~60GB). Ensure you have sufficient disk space and bandwidth.

Or download all models:
```bash
audio-agent-download-models --all
```

### 3. Setup Environment
```bash
python -m audio_agent.tools.catalog.setup_tool omni_captioner
python -m audio_agent.tools.catalog.setup_tool omni_captioner --verify
```

If you need to recreate the environment:
```bash
python -m audio_agent.tools.catalog.setup_tool omni_captioner --force
```

## Usage

### Tool: caption_audio

Generate a detailed caption/description of an audio file. Analyzes speech, environmental sounds, music, and other audio content to produce a comprehensive textual description.

**Input:**
```json
{
  "audio_path": "/path/to/audio.wav",
  "max_length": 8192
}
```

**Output:**
Detailed text description covering:
- Speech content and speaker characteristics
- Speaker emotions and multilingual expressions
- Environmental sounds and ambient atmospheres
- Music and cinematic sound effects
- Overall scene description

**Example Output:**
```
The audio features a conversation between two speakers in an outdoor setting. 
Speaker A, sounding excited and enthusiastic, describes a recent hiking trip 
in the mountains. In the background, birds are chirping and light wind can 
be heard rustling through trees. The atmosphere is peaceful and natural...
```

**Parameters:**
- `audio_path` (required): Path to the audio file
- `max_length` (optional): Maximum length of generated caption (default: 8192)

**Best Practices:**
- Use audio clips up to 30 seconds for optimal detail
- The model works best with clear, non-overlapping audio
- For longer audio, consider segmenting into shorter clips

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to Qwen3-Omni-Captioner model | `/lihaoyu/workspace/AUDIO_AGENT/models/Qwen3-Omni-30B-A3B-Captioner` |
| `DEVICE` | Device to use (auto/cpu/cuda) | `auto` |
| `MAX_LENGTH` | Default maximum caption length | `8192` |

## Model Information

- **Model**: Qwen/Qwen3-Omni-30B-A3B-Captioner
- **Description**: Fine-grained audio analysis model for comprehensive audio description
- **Capabilities**:
  - Speech understanding (multiple speakers, emotions, languages)
  - Environmental sound recognition
  - Music and audio effect analysis
  - Scene context understanding
- **Size**: ~60GB
- **VRAM Required**: ~30GB

## Troubleshooting

### Environment not found
Run setup: `python -m audio_agent.tools.catalog.setup_tool omni_captioner`

### Model download fails
Check HuggingFace access: `huggingface-cli login`

### Out of memory
The model requires significant VRAM. Try:
- Using a GPU with more memory
- Reducing batch size (if applicable)
- Using CPU (very slow): Set `DEVICE=cpu`

### Import errors in server
Ensure dependencies are installed:
```bash
python -m audio_agent.tools.catalog.setup_tool omni_captioner --force
```

## References

- [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni)
- [Model on HuggingFace](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)
- [Cookbook](https://github.com/QwenLM/Qwen3-Omni/blob/main/cookbooks/omni_captioner.ipynb)
