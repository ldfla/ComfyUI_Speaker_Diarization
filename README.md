# ComfyUI Pyannote Speaker Diarizer

Custom node for ComfyUI that performs speaker diarization with chronological output ordering.

## Features

- Up to 5 speaker outputs
- Chronological mapping:
  - speaker_1_audio = first speaker to talk
  - speaker_2_audio = second speaker to talk
  - speaker_3_audio = third speaker to talk
  - speaker_4_audio = fourth speaker to talk
  - speaker_5_audio = fifth speaker to talk
- Useful for telephony / call-center / IVR / conference audio
- Returns:
  - 5 isolated speaker audio tracks
  - human-readable summary
  - JSON with diarization segments

## Requirements

- ComfyUI
- Python environment with:
  - pyannote.audio >= 3.1
  - torch
  - torchaudio
- Hugging Face token with access to:
  - pyannote/speaker-diarization-3.1

## Installation

Clone inside your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ldfla/ComfyUI-Speaker-Diarization.git
install requirements:

cd ComfyUI/custom_nodes/ComfyUI-Speaker-Diarization
pip install -r requirements.txt


