# REVE

Training code for **REVE**, a lightweight event-verification method described in a manuscript submitted to ICASSP 2027. The manuscript is under review and has not been accepted by the conference. REVE reuses pre-projector audio-encoder states from a large audio-language model. The encoder and decoder are not updated during readout training.

This repository contains training and training-feature extraction code only. It does not include evaluation scripts, baseline implementations, model weights, AudioSet/DESED audio, or generated captions.

## Training stages

1. Fit a global linear readout from mean-pooled encoder states.
2. Fit a linear readout from the concatenated means of four ordered, equal-length frame intervals.
3. Apply the global readout to valid encoder frames. For each class, fit a three-input logistic readout on the clip logit, the frame-logit standard deviation, and the frame-logit maximum.
4. Fit a class-aware fusion model from the two branch probabilities and caption-mention labels on the development split. This last stage does not use the test split.

The first three stages use class-balanced AudioSet training labels. For DESED adaptation, the same readout-training script can be used with features extracted from the training mixtures and their event labels.

## Setup

Use Python 3.10 or newer and a CUDA-capable PyTorch installation for feature extraction. Install the remaining dependencies with:

```bash
python -m pip install -r requirements.txt
```

Set `QWEN2_AUDIO_MODEL` to a local `Qwen/Qwen2-Audio-7B-Instruct` checkpoint. The feature extractor uses the model's Whisper audio tower and discards padded encoder frames before pooling.

```bash
export QWEN2_AUDIO_MODEL=/path/to/Qwen2-Audio-7B-Instruct
```

## AudioSet readouts

The input directory must contain the AudioSet balanced-training parquet shards. Each row needs an `audio` entry with encoded audio bytes and a `human_labels` list of AudioSet class IDs. Class order is determined once from the training split and saved with the readout.

```bash
python training/extract_features.py \
  --parquet-dir /path/to/audioset/bal_train \
  --output artifacts/audioset_base.npz

python training/train_readouts.py \
  --stage base \
  --features artifacts/audioset_base.npz \
  --output checkpoints/audioset_readouts.pkl

python training/extract_features.py \
  --parquet-dir /path/to/audioset/bal_train \
  --base-readout checkpoints/audioset_readouts.pkl \
  --output artifacts/audioset_statistics.npz

python training/train_readouts.py \
  --stage statistics \
  --features artifacts/audioset_statistics.npz \
  --output checkpoints/audioset_readouts.pkl
```

The two feature-extraction passes avoid storing full frame-state tensors for the entire training set. The second pass must use the same audio shards as the first pass.

## Class-aware fusion

Prepare a development-only `.npz` file with one row per mapped caption event mention:

| Array | Shape | Meaning |
| --- | --- | --- |
| `frame_scores` | `[N]` or `[N, K]` | Frame-statistics branch probabilities |
| `segment_scores` | `[N]` or `[N, K]` | Four-segment branch probabilities |
| `class_ids` | `[N]` | Class indices in the saved readout order |
| `labels` | `[N]` | Binary event-presence labels |

The fusion design uses the two branch logits, a class-specific offset, and a class-specific slope on the mean branch logit. The regularization value `C=0.1` follows the development-set selection used for the paper.

```bash
python training/fit_fusion.py \
  --features artifacts/development_mentions.npz \
  --output checkpoints/reve_fusion.pkl
```

Threshold selection and all held-out measurements are evaluation steps, so they are not part of this training-only release. Generated caches and checkpoints belong in `artifacts/` and `checkpoints/`; both directories are ignored by Git.

## Code organization

- `training/qwen_encoder.py`: Qwen2-Audio encoder-state extraction and four-way pooling.
- `training/extract_features.py`: AudioSet training labels and cached feature arrays.
- `training/train_readouts.py`: global, segment, and frame-statistics training.
- `training/fit_fusion.py`: development-only class-aware fusion training.
- `training/common.py`: audio decoding and shared readout utilities.

Citation information will be updated after the review process. The repository does not redistribute third-party model checkpoints or datasets; obtain them under their original licenses.
