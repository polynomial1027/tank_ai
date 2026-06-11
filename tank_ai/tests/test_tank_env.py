from __future__ import annotations

import numpy as np

from tank_ai.src.env.tank_env import TankEnv


def test_reset_state_shape():
    env = TankEnv(render_mode=False)
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (6, env.rows, env.cols)


def test_step_returns_values():
    env = TankEnv(render_mode=False)
    env.reset()
    r1, r2, done, info = env.step(0, 0)
    assert isinstance(r1, float)
    assert isinstance(r2, float)
    assert isinstance(done, bool)
    assert "p1_health" in info


def test_random_actions_run():
    env = TankEnv(render_mode=False)
    env.reset()
    for _ in range(10):
        r1, r2, done, info = env.step(5, 0)
        if done:
            env.reset()
