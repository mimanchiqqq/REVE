"""Extract valid pre-projector states from Qwen2-Audio."""

from __future__ import annotations

import os
import gc
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


ROOT = Path(os.environ.get("REVE_ROOT", Path(__file__).resolve().parents[1]))
MODEL_DIR = os.environ.get(
    "QWEN2_AUDIO_MODEL",
    str(ROOT / "models" / "Qwen2-Audio-7B-Instruct"),
)
SAMPLE_RATE = 16_000


class QwenAudioEncoder:
    def __init__(self, model_dir: str = MODEL_DIR, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_dir)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        self.encoder = model.audio_tower.to(self.device).eval()
        del model
        gc.collect()

    @torch.inference_mode()
    def encode(self, waveforms: list[np.ndarray], batch_size: int = 8) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
        for start in range(0, len(waveforms), batch_size):
            chunk = waveforms[start : start + batch_size]
            inputs = self.processor.feature_extractor(
                chunk,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(self.device)
            feature_mask = inputs.attention_mask.to(self.device)
            feature_lengths, output_lengths = self.encoder._get_feat_extract_output_lengths(
                feature_mask.sum(-1)
            )
            max_length = (feature_mask.shape[-1] - 1) // 2 + 1
            positions = torch.arange(max_length, device=self.device)[None]
            padding = positions >= feature_lengths[:, None]
            mask = padding[:, None, None, :].expand(
                len(chunk), 1, max_length, max_length
            )
            attention_mask = torch.zeros(
                mask.shape,
                dtype=self.encoder.conv1.weight.dtype,
                device=self.device,
            )
            attention_mask[mask] = float("-inf")
            states = self.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state.float()
            for index, length in enumerate(output_lengths.tolist()):
                outputs.append(states[index, : int(length)].cpu().numpy())
        return outputs


def pool_segments(states: np.ndarray, n_segments: int = 4) -> np.ndarray:
    """Mean-pool valid states into ordered contiguous segments."""

    boundaries = np.rint(np.linspace(0, len(states), n_segments + 1)).astype(int)
    pooled = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        right = max(right, left + 1)
        pooled.append(states[left:right].mean(axis=0))
    return np.stack(pooled).astype(np.float32)


def frame_statistics(states: np.ndarray, readout: dict) -> np.ndarray:
    """Return clip logit, frame-score standard deviation, and maximum."""
    from common import class_score_statistics

    return class_score_statistics(states, readout)
