
# Audio LM Implementation

A modular, simplified implementation of a multimodal LLM inspired by Audio Language Modeling. This codebase fuses an Audio Encoder (Whisper) with an LLM (Sarvam-2B) using a projector.

## Directory Structure

- `config.py`: Configuration for Model and Training.
- `data.py`: Dataset and DataCollator implementation.
- `model.py`: Core model architecture (AudioEncoder + Projector + LLM).
- `train.py`: Training script using Hugging Face Trainer.
- `inference.py`: Inference script for testing the model.

## Setup

1. Install dependencies:
   ```bash
   pip install torch torchaudio transformers peft accelerate datasets
   ```

2. Dataset:
   The project is configured to use Hugging Face datasets (default: `fixie-ai/common_voice_17_0`).
   You can change the dataset settings in `config.py`.

## Usage

### Training

1. Configure `config.py` if needed (batch size, learning rate, paths).
2. Run training:
   ```bash
   python -m audio_lm.train
   ```

### Inference

Run inference on an audio file:
```bash
python -m audio_lm.inference /path/to/audio.wav
```

## Model Architecture

The `AudioLM` consists of:
- **Audio Encoder**: `openai/whisper-small` (frozen).
- **Projector**: A 2-layer MLP (`Linear` -> `GELU` -> `Linear`) maps audio features to LLM embedding space.
- **LLM**: `sarvamai/sarvam-2b-v0.5` with LoRA (fine-tuned).


## Dataset Guide

To train a model capable of real-time voice interaction (especially for Indic languages using Sarvam), you need a mix of datasets served in two main stages:

### Stage 1: Alignment (ASR & Continuation)
The goal is to teach the projector to map audio to the LLM's embedding space.
*   **Data Type**: `<Audio> -> <Transcription>`
*   **Recommended Datasets**:
    *   **Indic/Multilingual**: [CommonVoice 17.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) (Hindi, Tamil, etc.), [IndicVoices](https://huggingface.co/datasets/AI4Bharat/IndicVoices), [Kathbath](https://huggingface.co/datasets/AI4Bharat/Kathbath).
    *   **English**: [LibriSpeech](https://huggingface.co/datasets/librispeech_asr), [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech).
*   **Task**: The model is given audio and forced to output the exact transcription.

### Stage 2: Instruction Tuning (Speech-to-Text Chat)
The goal is to teach the model to *understand* speech and *respond* intelligently.
*   **Data Type**: `<Audio Instruction> -> <Text Response>`
*   **Strategy**:
    1.  **Synthetic**: Take a text-only instruction dataset (e.g., Alpaca, OpenAssist). Use a TTS (Text-to-Speech) model to generate audio for the "User Instruction". Train the model to output the "Assistant Response".
    2.  **Continuation**: Split a spoken sentence in half. Feed the first half as Audio. Train LLM to complete the text of the second half.
    3.  **Cross-Modal**: Use mixed datasets like [CoVoST 2](https://huggingface.co/datasets/covost2) (Speech Translation).

### Formatting for this Codebase
The codebase uses `datasets.load_dataset`. See `data.py` for details on how different datasets are mapped to `audio` and `text` fields.
