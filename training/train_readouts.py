"""Fit REVE's base, four-segment, and frame-statistics readouts."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import fit_ovr_readout, save_pickle


def train_segment_readout(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 30,
    batch_size: int = 512,
    learning_rate: float = 2e-3,
) -> dict:
    """Train the linear four-segment branch with class-balanced BCE."""
    features = np.asarray(features, np.float32)
    labels = np.asarray(labels, np.float32)
    rng = np.random.RandomState(0)
    order = rng.permutation(len(features))
    n_val = max(1, int(round(0.1 * len(features))))
    val_idx, fit_idx = order[:n_val], order[n_val:]
    if not len(fit_idx):
        raise ValueError("at least two training examples are required")
    mean = features[fit_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features[fit_idx].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    x = ((features - mean) / scale).astype(np.float32)
    positives = labels[fit_idx].sum(axis=0)
    pos_weight = (len(fit_idx) - positives) / np.maximum(positives, 1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = torch.nn.Linear(x.shape[1], labels.shape[1]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(pos_weight.astype(np.float32)).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = rng.permutation(fit_idx)
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start : start + batch_size]
            prediction = model(torch.from_numpy(x[batch]).to(device))
            loss = loss_fn(prediction, torch.from_numpy(labels[batch]).to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(x[val_idx]).to(device))
            validation_loss = float(
                loss_fn(prediction, torch.from_numpy(labels[val_idx]).to(device)).item()
            )
        if validation_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6:
                break
    assert best_state is not None
    return {
        "mean": mean,
        "scale": scale,
        "weight": best_state["weight"].numpy().astype(np.float32),
        "bias": best_state["bias"].numpy().astype(np.float32),
        "best_epoch": best_epoch,
    }


def train_statistics_readout(statistics: np.ndarray, labels: np.ndarray) -> dict:
    """Fit one three-input logistic calibrator per ontology class."""
    if statistics.shape != (len(labels), labels.shape[1], 3):
        raise ValueError("statistics must have shape [clips, classes, 3]")
    classes = labels.shape[1]
    means = np.zeros((classes, 3), np.float32)
    scales = np.ones((classes, 3), np.float32)
    weights = np.zeros((classes, 3), np.float32)
    biases = np.zeros(classes, np.float32)
    for class_index in range(classes):
        target = labels[:, class_index]
        if target.min() == target.max():
            continue
        scaler = StandardScaler().fit(statistics[:, class_index, :])
        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs"
        ).fit(scaler.transform(statistics[:, class_index, :]), target)
        means[class_index] = scaler.mean_
        scales[class_index] = scaler.scale_
        weights[class_index] = model.coef_[0]
        biases[class_index] = model.intercept_[0]
    return {"mean": means, "scale": scales, "weight": weights, "bias": biases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("base", "statistics"), required=True)
    args = parser.parse_args()

    with np.load(args.features, allow_pickle=False) as package:
        labels = package["labels"].astype(np.uint8)
        classes = package["classes"].tolist()
        if args.stage == "base":
            readouts = {
                "classes": classes,
                "base": fit_ovr_readout(package["means"], labels),
                "segment": train_segment_readout(
                    package["segments"].reshape(len(labels), -1), labels
                ),
            }
        else:
            with args.output.open("rb") as handle:
                readouts = pickle.load(handle)
            if classes != readouts["classes"]:
                raise ValueError("class order differs from the base readout")
            readouts["statistics"] = train_statistics_readout(
                package["statistics"].astype(np.float32), labels
            )
    save_pickle(args.output, readouts)
    print(f"saved {args.stage} readout to {args.output}")


if __name__ == "__main__":
    main()
