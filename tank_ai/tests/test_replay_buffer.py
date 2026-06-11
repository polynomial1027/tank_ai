from __future__ import annotations

import numpy as np

from tank_ai.src.utils.replay_buffer import ReplayBuffer


def test_replay_buffer_sample():
    rb = ReplayBuffer(10)
    for i in range(6):
        s = np.zeros((6, 15, 20), dtype=np.float32)
        ns = np.ones((6, 15, 20), dtype=np.float32)
        rb.push(s, i % 6, float(i), ns, False)
    states, actions, rewards, next_states, dones = rb.sample(4)
    assert states.shape == (4, 6, 15, 20)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
