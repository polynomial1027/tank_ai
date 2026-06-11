from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class EpisodeTiming:
    episode: int
    elapsed_seconds: float
    total_elapsed_seconds: float


class TrainingTimer:
    def __init__(self) -> None:
        self.training_start = time.time()
        self.episode_start = time.time()

    def start_episode(self) -> None:
        self.episode_start = time.time()

    def end_episode(self, episode: int) -> EpisodeTiming:
        now = time.time()
        return EpisodeTiming(
            episode=episode,
            elapsed_seconds=now - self.episode_start,
            total_elapsed_seconds=now - self.training_start,
        )

    @staticmethod
    def format_seconds(seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
