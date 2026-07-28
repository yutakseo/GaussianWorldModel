#!/usr/bin/env python3
"""Continuously overwrite one loss plot from GaussianWM JSONL metrics."""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_losses(metrics_path):
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    if not metrics_path.is_file():
        return train_steps, train_losses, val_steps, val_losses

    with metrics_path.open(encoding="utf-8") as metrics_file:
        for line in metrics_file:
            try:
                record = json.loads(line)
                loss = record["metrics"]["total_loss"]
                step = record["step"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if record.get("split") == "train":
                train_steps.append(step)
                train_losses.append(loss)
            elif record.get("split") == "validation":
                val_steps.append(step)
                val_losses.append(loss)
    return train_steps, train_losses, val_steps, val_losses


def moving_average(values, window):
    if len(values) < window:
        return np.asarray(values), 1
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid"), window


def render(metrics_path, output_path, smooth_window):
    train_steps, train_losses, val_steps, val_losses = read_losses(metrics_path)
    if not train_steps and not val_steps:
        return False

    fig, axis = plt.subplots(figsize=(11, 6))
    if train_steps:
        axis.plot(
            train_steps,
            train_losses,
            color="#4c78a8",
            alpha=0.18,
            linewidth=0.8,
            label="Train loss (raw)",
        )
        smoothed, effective_window = moving_average(train_losses, smooth_window)
        smooth_steps = train_steps[effective_window - 1 :]
        axis.plot(
            smooth_steps,
            smoothed,
            color="#1f5a94",
            linewidth=2,
            label=f"Train loss (moving avg, n={effective_window})",
        )
    if val_steps:
        axis.plot(
            val_steps,
            val_losses,
            color="#e45756",
            marker="o",
            markersize=4,
            linewidth=2,
            label="Validation loss",
        )

    axis.set_title("GaussianWM Training and Validation Loss")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Total loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.png")
    fig.savefig(temporary_path, dpi=140)
    plt.close(fig)
    os.replace(temporary_path, output_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--watch-pid", type=int)
    args = parser.parse_args()

    while True:
        render(args.metrics, args.output, args.smooth_window)
        if args.watch_pid is not None and not process_exists(args.watch_pid):
            render(args.metrics, args.output, args.smooth_window)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
