"""Extract valid encoder summaries for readout training."""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import decode_audio
from qwen_encoder import QwenAudioEncoder, frame_statistics, pool_segments


def collect_classes(paths: list[Path], top_k: int) -> list[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        frame = pd.read_parquet(path, columns=["human_labels"])
        for labels in frame["human_labels"]:
            counts.update(list(labels))
    return [name for name, _ in counts.most_common(top_k)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=527)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-readout", type=Path)
    args = parser.parse_args()

    paths = sorted(args.parquet_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files in {args.parquet_dir}")
    if args.base_readout:
        with args.base_readout.open("rb") as handle:
            readouts = pickle.load(handle)
        classes = readouts["classes"]
        base_readout = readouts["base"]
    else:
        classes = collect_classes(paths, args.top_k)
        base_readout = None
    class_index = {name: index for index, name in enumerate(classes)}
    encoder = QwenAudioEncoder()
    means, segments, labels, statistics = [], [], [], []

    for path in paths:
        frame = pd.read_parquet(path)
        waveforms = [decode_audio(item["bytes"]) for item in frame["audio"]]
        states = encoder.encode(waveforms, batch_size=args.batch_size)
        for row, hidden in zip(frame.itertuples(index=False), states):
            means.append(hidden.mean(axis=0))
            segments.append(pool_segments(hidden))
            if base_readout is not None:
                statistics.append(frame_statistics(hidden, base_readout))
            target = np.zeros(len(classes), dtype=np.uint8)
            for name in list(row.human_labels):
                if name in class_index:
                    target[class_index[name]] = 1
            labels.append(target)
        print(f"processed {path.name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "means": np.asarray(means, dtype=np.float32),
        "segments": np.asarray(segments, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.uint8),
        "classes": np.asarray(classes),
    }
    if base_readout is not None:
        payload["statistics"] = np.asarray(statistics, dtype=np.float32)
    np.savez_compressed(args.output, **payload)
    print(f"saved features to {args.output}")


if __name__ == "__main__":
    main()
