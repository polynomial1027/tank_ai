from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TankGameConfig:
    width: int = 640
    height: int = 480
    block_size: int = 32
    speed: int = 30
    max_steps: int = 700
    wall_count: int = 28
    bullet_speed_blocks: int = 1
    bullet_max_life: int = 30
    tank_max_health: int = 3
    reward_mode: str = "balanced"


@dataclass
class CNNDQNConfig:
    input_channels: int = 6
    output_size: int = 6
    learning_rate: float = 0.0005
    gamma: float = 0.95
    batch_size: int = 128
    memory_size: int = 80_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update_every: int = 20
    max_episodes: int = 500
    render_every: int = 50
    save_every: int = 50
