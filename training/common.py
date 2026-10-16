"""Shared data and linear-readout helpers."""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SAMPLE_RATE = 16_000


def decode_audio(blob: bytes) -> np.ndarray:
    import soundfile as sf

    waveform, sample_rate = sf.read(io.BytesIO(blob), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        import librosa

        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE
        )
    return np.asarray(waveform, dtype=np.float32)


def fit_ovr_readout(features: np.ndarray, labels: np.ndarray) -> dict:
    """Fit one balanced logistic readout for every event class."""

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)
    scaler = StandardScaler().fit(features)
    transformed = scaler.transform(features)
    coefficients = np.zeros((labels.shape[1], features.shape[1]), np.float32)
    intercepts = np.zeros(labels.shape[1], np.float32)
    for class_index in range(labels.shape[1]):
        target = labels[:, class_index]
        if target.min() == target.max():
            continue
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
        )
        model.fit(transformed, target)
        coefficients[class_index] = model.coef_[0]
        intercepts[class_index] = model.intercept_[0]
    return {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
        "coefficients": coefficients,
        "intercepts": intercepts,
    }


def class_score_statistics(states: np.ndarray, readout: dict) -> np.ndarray:
    """Return [clip logit, valid-frame logit std, valid-frame logit max]."""
    values = (states - readout["mean"]) / np.maximum(readout["scale"], 1e-6)
    frame_logits = values @ readout["coefficients"].T + readout["intercepts"]
    return np.stack(
        (frame_logits.mean(axis=0), frame_logits.std(axis=0), frame_logits.max(axis=0)),
        axis=-1,
    ).astype(np.float32)


def predict_readout(model: dict, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    values = (features - model["mean"]) / np.maximum(model["scale"], 1e-6)
    logits = values @ model["coefficients"].T + model["intercepts"]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def save_pickle(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)
