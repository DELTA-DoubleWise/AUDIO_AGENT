# autochord Tool

Automatic chord recognition for the Audio Agent Framework using [autochord](https://github.com/cjbayron/autochord).

## Model

- **Architecture**: BiLSTM-CRF
- **Vocabulary**: 25 classes — N (no chord) + 12 major + 12 minor triads
- **Input**: Audio file (any format librosa can read)
- **Output**: Time-localized chord segments with start/end times

## Setup

```bash
./setup.sh
```

## Tools

- `recognize_chords`: Recognize chord progression in an audio file
- `healthcheck`: Check runtime availability

## Limitations

- Does not recognize 7th chords, extended chords, or inversions
- Major/minor triads only
- Model is downloaded from Google Drive on first import (~3MB)
