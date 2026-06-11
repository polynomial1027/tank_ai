from __future__ import annotations

from typing import Callable, Dict


DEFAULT_REWARD_VALUES = {
    "balanced": {
        "win": 20.0,
        "lose": -20.0,
        "hit_enemy": 8.0,
        "got_hit": -8.0,
        "fire": -0.03,
        "move": -0.01,
        "wall_bump": -0.2,
        "timeout": -2.0,
    },
    "aggressive": {
        "win": 20.0,
        "lose": -20.0,
        "hit_enemy": 12.0,
        "got_hit": -6.0,
        "fire": -0.01,
        "move": -0.005,
        "wall_bump": -0.15,
        "timeout": -1.0,
    },
    "survival": {
        "win": 25.0,
        "lose": -25.0,
        "hit_enemy": 6.0,
        "got_hit": -12.0,
        "fire": -0.04,
        "move": -0.005,
        "wall_bump": -0.3,
        "timeout": 0.0,
    },
    "hit_reward": {
        "win": 10.0,
        "lose": -10.0,
        "hit_enemy": 15.0,
        "got_hit": -5.0,
        "fire": -0.02,
        "move": 0.0,
        "wall_bump": -0.1,
        "timeout": 0.0,
    },
}


def calculate_reward(reward_mode: str, events: dict) -> float:
    """
    Calculate one-agent reward from environment events.

    Expected events:
        won, lost, hit_enemy, got_hit, fired, moved, wall_bump, timeout
    """
    if reward_mode not in DEFAULT_REWARD_VALUES:
        raise ValueError(
            f"Unknown reward mode: {reward_mode}. "
            f"Available: {', '.join(DEFAULT_REWARD_VALUES.keys())}"
        )

    values = DEFAULT_REWARD_VALUES[reward_mode]
    reward = 0.0

    if events.get("won", False):
        reward += values["win"]
    if events.get("lost", False):
        reward += values["lose"]
    if events.get("hit_enemy", False):
        reward += values["hit_enemy"]
    if events.get("got_hit", False):
        reward += values["got_hit"]
    if events.get("fired", False):
        reward += values["fire"]
    if events.get("moved", False):
        reward += values["move"]
    if events.get("wall_bump", False):
        reward += values["wall_bump"]
    if events.get("timeout", False):
        reward += values["timeout"]

    return float(reward)


def get_available_reward_modes() -> list[str]:
    return list(DEFAULT_REWARD_VALUES.keys())
