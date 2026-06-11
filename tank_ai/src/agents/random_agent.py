from __future__ import annotations

import random


class RandomAgent:
    def __init__(self, action_size: int = 6) -> None:
        self.action_size = action_size

    def get_action(self, state=None, training: bool = False) -> int:
        return random.randint(0, self.action_size - 1)
