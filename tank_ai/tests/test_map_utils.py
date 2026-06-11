from __future__ import annotations

from tank_ai.src.env.tank_env import TankEnv
from tank_ai.src.utils.map_utils import load_map


def test_load_default_map() -> None:
    config = load_map("tank_ai/maps/classic.json")
    assert config.cols == 20
    assert config.rows == 15
    assert config.p1_spawn is not None
    assert config.p2_spawn is not None


def test_env_uses_custom_map() -> None:
    env = TankEnv(render_mode=False, map_path="tank_ai/maps/classic.json")
    assert env.cols == 20
    assert env.rows == 15
    assert len(env.walls) > 0
    env.close()
