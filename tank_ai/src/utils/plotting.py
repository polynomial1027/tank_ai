from __future__ import annotations

import csv
import os
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


def save_training_log_csv(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_summary(scores: Sequence[float], avg_scores: Sequence[float], losses: Sequence[float], path: str, title: str) -> None:
    if not scores:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    episodes = list(range(1, len(scores)+1))
    plt.figure(figsize=(12, 8))
    plt.subplot(3,1,1)
    plt.plot(episodes, scores)
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True)
    plt.subplot(3,1,2)
    plt.plot(episodes, avg_scores)
    plt.ylabel("Avg100")
    plt.grid(True)
    plt.subplot(3,1,3)
    plt.plot(episodes, losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
