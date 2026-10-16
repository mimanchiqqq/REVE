"""Fit the class-aware fusion layer for the two REVE readouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from common import save_pickle


def design(
    frame_scores: np.ndarray,
    segment_scores: np.ndarray,
    classes: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    if frame_scores.ndim == 2:
        rows = np.arange(len(classes))
        frame_scores = frame_scores[rows, classes]
    if segment_scores.ndim == 2:
        rows = np.arange(len(classes))
        segment_scores = segment_scores[rows, classes]
    frame_scores = logit(np.clip(frame_scores, 1e-5, 1 - 1e-5))
    segment_scores = logit(np.clip(segment_scores, 1e-5, 1 - 1e-5))
    base = np.column_stack((frame_scores, segment_scores))
    one_hot = np.zeros((len(classes), num_classes), dtype=np.float32)
    one_hot[np.arange(len(classes)), classes] = 1.0
    interactions = one_hot * base.mean(axis=1, keepdims=True)
    return np.column_stack((base, one_hot, interactions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=527)
    args = parser.parse_args()

    package = np.load(args.features)
    x = design(
        package["frame_scores"],
        package["segment_scores"],
        package["class_ids"].astype(int),
        args.num_classes,
    )
    model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=79,
    ).fit(x, package["labels"].astype(np.uint8))
    save_pickle(args.output, {"model": model})
    print(f"saved fusion model to {args.output}")


if __name__ == "__main__":
    main()
